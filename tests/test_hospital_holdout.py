"""Synthetic-only protocol tests; no patient data, IDs, videos or weights."""
import ast
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_tools import build_dataset as parser
from experiments.hospital_holdout import prepare as data
from experiments.hospital_holdout import video_data as video
from experiments.hospital_holdout.train import public_config, weights_for
from experiments.hospital_holdout.unit_evidence import eaf_check


@pytest.mark.parametrize("token,unit,value",[("1000","seconds",1000),
                         ("1000","milliseconds",1),("1.25","seconds",1.25),
                         ("00:20:01.5",None,1201.5)])
def test_time_explicit(token,unit,value):
    assert parser.parse_time(token,unit)==value


@pytest.mark.parametrize("token,unit",[("1000",None),("1.5",None),("nan","seconds"),
                         ("inf","seconds"),("00:60:00",None),("00:00:60",None)])
def test_ambiguous_or_invalid_time_rejected(token,unit):
    with pytest.raises(ValueError): parser.parse_time(token,unit)


def tsv(start="1.0",end="2.0"):
    return f"default\t1\t00:00:01.000\t{start}\t00:00:02.000\t{end}\t1\tx\t消毒"


@pytest.mark.parametrize("unit,start,end",[("seconds","1.0","2.0"),
                                          ("milliseconds","1000","2000")])
def test_tsv_clock_evidence(unit,start,end):
    text=tsv(start,end)
    assert data.paired_tsv_unit(text)==unit
    assert parser.parse_tsv_rows(text,unit)[0]["start_sec"]==1


def test_tsv_unit_mismatch():
    assert data.paired_tsv_unit(tsv("9","10")) is None
    with pytest.raises(ValueError):parser.parse_tsv_rows(tsv(),"milliseconds")
    with pytest.raises(ValueError):parser.parse_tsv_rows(tsv())


def test_wide_keeps_blanks_and_no_magnitude_heuristic():
    ranges=["1000-1002","1003-1004","1005-1006"]
    text="default "+"消毒".ljust(20)+"".ljust(20)+"进针"+"\nTC      "+"".join(s.ljust(20) for s in ranges)
    with pytest.raises(ValueError):parser.parse_wide(text)
    events=parser.parse_wide(text,"seconds")
    assert [e["label_id"] for e in events]==[0,None,1]
    assert events[0]["start_sec"]==1000
    assert parser.parse_wide(text,"milliseconds")[0]["start_sec"]==1


def test_blocks_and_raw_provenance():
    text="消毒\n\n00:00:01.5 - 00:00:02.5\n进针\n3 - 4"
    assert parser.detect_format(text)=="label_time_blocks"
    with pytest.raises(ValueError):parser.parse_blocks(text)
    result=parser.parse_blocks(text,"seconds")
    assert result[0]["raw_start_time"]=="00:00:01.5"
    assert result[1]["start_sec"]==3


def test_unit_map_evidence_not_guess():
    record={"input_format":"elan_wide","source_txt_sha256":"a"*64}
    assert data.verified_unit(record,"",{})[0] is None
    with pytest.raises(ValueError):data.verified_unit(record,"",{"a"*64:{"unit":"seconds"}})
    evidence={"a"*64:{"unit":"seconds","evidence_type":"data_owner_confirmation",
                      "evidence":"Synthetic owner confirmed export units"}}
    assert data.verified_unit(record,"",evidence)==("seconds","data_owner_confirmation")
    record["input_format"]="tsv_rows"
    evidence["a"*64]["unit"]="milliseconds"
    with pytest.raises(ValueError):data.verified_unit(record,tsv(),evidence)


def partition():
    def subset(prefix,source):
        return [{"clip_id":f"{prefix}-clip{i}","record_id":f"{prefix}-record{i}",
                 "group_id":f"{prefix}-group{i}","video_path":f"{prefix}-video{i}",
                 "source_collection":source,"label_id":i,"clip_start_sec":1,"clip_end_sec":2}
                for i in range(7)]
    return subset("train","FSN"),subset("test","menzhen")


def test_partition_count_coverage_and_hospital():
    train,test=partition()
    data.validate_partition(train,test,{"train":7,"val":7})
    with pytest.raises(ValueError):data.validate_partition(train,test)
    train[0]["source_collection"]="menzhen"
    with pytest.raises(ValueError):data.validate_partition(train,test,{"train":7,"val":7})


@pytest.mark.parametrize("field",["clip_id","record_id","group_id","video_path"])
def test_no_identity_or_video_crossing(field):
    train,test=partition();test[0][field]=train[0][field]
    with pytest.raises(ValueError):data.validate_partition(train,test,{"train":7,"val":7})


def test_actual_pts_uniform_and_short_clip():
    pts=np.stack([np.arange(100)/25,(np.arange(100)+1)/25],axis=1)
    selected=video.select_indices(pts,.5,2)
    assert len(selected)==16 and np.all((pts.mean(1)[selected]>=.5)&(pts.mean(1)[selected]<2))
    short=video.select_indices(pts,.01,.031)
    assert len(set(short))==1
    with pytest.raises(ValueError):video.select_indices(pts,.025,.03)


@pytest.mark.parametrize("pts,start,end",[(np.array([[1,0]]),0,1),
                 (np.array([[0,.1],[0,.05]]),0,1),
                 (np.array([[0,float('nan')]]),0,1),
                 (np.array([[0,.1]]),-1,1),(np.array([[0,.1]]),1,1)])
def test_invalid_pts(pts,start,end):
    with pytest.raises(ValueError):video.select_indices(pts,start,end)


def test_normalization_and_declared_protocol():
    frames=np.zeros((16,224,224,3),dtype=np.uint8)
    value=video.normalise_rgb(frames)
    assert value.shape==(3,16,224,224)
    torch.testing.assert_close(value[:,0,0,0],-torch.tensor(video.PREPROCESS["mean"])/torch.tensor(video.PREPROCESS["std"]))
    c=public_config()
    assert c["epochs_total"]==105 and c["optimizer_steps_total"]==12075
    assert c["selection"]=="best_finetune_Menzhen888_macro_f1"
    assert c["early_stopping"] is False and "patience" not in c
    assert video.PREPROCESS["official_full_training_or_multiview_recipe"] is False


def official_modules():
    source=Path(os.environ.get("FSN_OFFICIAL_VIDEO_SOURCE",Path(__file__).resolve().parents[2].parent/"reference_VideoMAEv2"))
    if not source.is_dir():pytest.skip("pinned upstream source not provided for spatial parity")
    from experiments.hospital_holdout.run import SPATIAL_FILES
    for p,h in SPATIAL_FILES.items():assert data.sha(source/p)==h
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module);return module
    function=load("upstream_functional",source/"dataset/functional.py")
    volume=load("upstream_volume",source/"dataset/volume_transforms.py")
    # Execute the exact upstream crop class, avoiding unrelated augmentation imports.
    tree=ast.parse((source/"dataset/video_transforms.py").read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="CenterCrop")
    scope={"FF":function,"np":np,"numbers":__import__("numbers"),"PIL":__import__("PIL")}
    exec(compile(ast.Module(body=[node],type_ignores=[]),"upstream_CenterCrop","exec"),scope)
    return function,volume,scope["CenterCrop"]


@pytest.mark.parametrize("shape",[(2,300,480,3),(2,480,300,3),(2,256,301,3),(2,224,224,3)])
def test_exact_official_spatial_output(shape):
    function,volume,crop=official_modules()
    frames=np.random.default_rng(42).integers(0,256,size=shape,dtype=np.uint8)
    expected=crop((224,224))(function.resize_clip(list(frames),256))
    np.testing.assert_array_equal(video.spatial_rgb(frames),np.stack(expected))
    expected=function.normalize(volume.ClipToTensor()(expected),video.PREPROCESS["mean"],video.PREPROCESS["std"])
    torch.testing.assert_close(video.normalise_rgb(video.spatial_rgb(frames)),expected,rtol=0,atol=0)


def test_train_weight_counts_only_train():
    train,_=partition()
    class SyntheticDataset:
        raw_rows=train
        def __len__(self):return 7
    value=weights_for(SyntheticDataset(),torch.device("cpu"))
    assert torch.isfinite(value).all()


def test_paired_eaf_replay_and_wrong_unit(tmp_path):
    path=tmp_path/"synthetic.eaf"
    path.write_text('<ANNOTATION_DOCUMENT><HEADER TIME_UNITS="milliseconds"/>'
        '<TIME_ORDER><TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="1000"/>'
        '<TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="2000"/></TIME_ORDER>'
        '<TIER TIER_ID="default"><ANNOTATION><ALIGNABLE_ANNOTATION TIME_SLOT_REF1="ts1" '
        'TIME_SLOT_REF2="ts2"><ANNOTATION_VALUE>消毒</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION>'
        '</ANNOTATION></TIER></ANNOTATION_DOCUMENT>')
    eaf_check(path,parser.parse_blocks("消毒\n1 - 2","seconds"))
    with pytest.raises(ValueError):eaf_check(path,parser.parse_blocks("消毒\n1 - 2","milliseconds"))


def test_video_missing_historical_size_is_explicit(tmp_path,monkeypatch):
    path=tmp_path/"synthetic.mp4";path.write_bytes(b"synthetic")
    probe={"file_size_bytes":9,"duration_sec":10.,"width":320,"height":240}
    monkeypatch.setattr(parser,"parse_video_metadata",lambda *a,**k:probe)
    result=data.audit_video(path,{"file_size_bytes":0,"metadata_reader":"iso_bmff_fallback_no_moov"})
    assert result["historical_size_missing"] is True
    assert result["historical_video_sha_available"] is False
    assert result["size_check"]=="missing_no_moov_sentinel_reprobed"
    with pytest.raises(ValueError):data.audit_video(path,{"file_size_bytes":8})
    with pytest.raises(ValueError):data.audit_video(path,{"file_size_bytes":0})
    probe["duration_sec"]=None
    with pytest.raises(ValueError):data.audit_video(path,{"file_size_bytes":9,"duration_sec":10})


def test_authorized_tail_correction_keeps_interval_and_rejects_large_errors():
    assert data.capped_terminal_end(0,43,42.563,42.57)==42.563
    assert data.capped_terminal_end(36.425,39.845,39.836667,39.83)==39.83
    with pytest.raises(ValueError):data.capped_terminal_end(0,45,42.563,42.57)
    with pytest.raises(ValueError):data.capped_terminal_end(43,44,42.563,42.57)
    with pytest.raises(ValueError):data.capped_terminal_end(0,40,42.563,42.57)
    with pytest.raises(ValueError):data.capped_terminal_end(0,43,42.563,float('nan'))


def test_preparation_read_only_and_no_old_test(tmp_path,monkeypatch):
    monkeypatch.setattr(parser,"parse_video_metadata",lambda path,**k:
                        {"file_size_bytes":path.stat().st_size,"duration_sec":10.,"width":320,"height":240})
    manifest=tmp_path/"old_manifests";manifest.mkdir()
    records,events=[],[]
    for split,source in [("train","FSN"),("val","menzhen")]:
        rid=split+"-synthetic-record"
        text="\n".join(f"{label}\n00:00:0{i+1} - 00:00:0{i+2}" for i,label in enumerate(parser.CANONICAL_LABELS))
        txt=tmp_path/f"{split}.txt";txt.write_text(text)
        raw_video=tmp_path/f"{split}.mp4";raw_video.write_bytes(split.encode())
        records.append({"record_id":rid,"source_collection":source,"source_txt_path":str(txt),
               "source_txt_sha256":data.sha(txt),"text_encoding":"utf-8","input_format":"label_time_blocks",
               "video_metadata":{"duration_sec":10,"file_size_bytes":raw_video.stat().st_size}})
        clips=[]
        for i,e in enumerate(parser.parse_blocks(text)):
            eid=f"{rid}-e{i:04d}";events.append({**e,"event_id":eid})
            clips.append({"clip_id":f"{split}-clip{i}","record_id":rid,"event_id":eid,
                 "split":split,"source_collection":source,"group_id":rid,"label_id":i,
                 "clip_start_sec":i+1,"clip_end_sec":i+2,"clip_duration_sec":1,"video_path":str(raw_video)})
        (manifest/f"{split}.jsonl").write_bytes(b"".join(data.encoded(c) for c in clips))
    # A trap to ensure this audit never reads an extra old test manifest.
    (manifest/"test.jsonl").write_text("not valid JSON; never read")
    record_file=tmp_path/"records.jsonl";event_file=tmp_path/"events_all.jsonl"
    record_file.write_bytes(b"".join(data.encoded(r) for r in records))
    event_file.write_bytes(b"".join(data.encoded(e) for e in events))
    monkeypatch.setattr(data,"OLD_SHA",{s:data.sha(manifest/f"{s}.jsonl") for s in ['train','val']})
    monkeypatch.setattr(data,"OLD_COUNTS",{"train":7,"val":7})
    monkeypatch.setattr(data,"EXPECTED_COUNTS",{"train":7,"val":7})
    summary,detail,derived=data.prepare(manifest,record_file,event_file,tmp_path)
    assert summary["status"]=="verified" and summary["files_created"] is False
    assert len(derived["train"])==7 and len(derived["val"])==7
    assert detail["protocol"]["independent_test_evaluated"] is False
    assert detail["protocol"]["selection"]=="best_finetune_Menzhen888_macro_f1"
