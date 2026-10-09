"""Read-only plan by default; sealed five-variant waves on 1–4 independent GPUs."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

from experiments.direction_gate import run as historical_video
from experiments.paper_validation import legacy
from experiments.relation.data import sha256_file
from experiments.videomae_tcn.io import ROOT, append_line, atomic_json, repository_state, write_json, write_text
from experiments.videomae_tcn.run import fresh_output, wait_released_gpu
from .model import VARIANTS, AnchorReferenceTCN
from experiments.relation.tcn_baseline import TCNConfig

CONFIG_PATH = Path(__file__).with_name("config.json")
CONFIG = json.loads(CONFIG_PATH.read_text())


def code_hashes():
    paths = subprocess.check_output(["git","ls-files","-z"],cwd=ROOT).decode().split("\0")
    return {n:sha256_file(ROOT/n) for n in paths if n and (
        (n.startswith("experiments/") and n.endswith((".py",".json"))) or
        (n.startswith("tests/") and n.endswith(".py")))}


def evidence(backbone, index, source, historical_run):
    if backbone == "adafocus":
        return legacy.prepare(index,source,historical_run)
    from experiments.videomae_tcn.evidence import prepare
    return prepare(index,source)


def waves(variants, gpus, uuids):
    if len(gpus) not in (1,2,3,4) or len(set(gpus)) != len(gpus) or len(uuids)!=len(gpus) \
            or len(set(uuids)) != len(uuids) or any(g<0 for g in gpus):
        raise ValueError("use 1–4 unique physical GPUs with one distinct UUID each")
    if list(variants) != list(VARIANTS):
        raise ValueError("do not silently remove registered controls")
    return [[{"variant":v,"gpu":gpus[i],"gpu_uuid":uuids[i]} for i,v in enumerate(variants[start:start+len(gpus)])]
            for start in range(0,len(variants),len(gpus))]


def plan(args):
    if not re.fullmatch(r"[0-9a-f]{40}",args.expected_commit) or repository_state() != {"commit":args.expected_commit,"clean":True}:
        raise ValueError("require exact clean committed independent worktree")
    index,source,out,base = [Path(x).resolve() for x in (args.index,args.source_protocol_dir,args.output,args.historical_run)]
    fresh_output(out,index,source)
    if out.is_relative_to(base) or base.is_relative_to(out):
        raise ValueError("new private output overlaps historical run")
    python=Path(args.python).resolve()
    if not python.is_file(): raise FileNotFoundError(python)
    gpus=[int(x) for x in args.gpus.split(",")]
    uuids=args.gpu_uuids.split(",")
    if any(not re.fullmatch(r"GPU-[a-fA-F0-9-]{8,}",u) for u in uuids):
        raise ValueError("provide full GPU UUIDs measured on the actual server")
    layout=waves(VARIANTS,gpus,uuids)
    _,_,info=evidence(args.backbone,index,source,base)
    if args.backbone=="videomae2":
        old=historical_video.plan(SimpleNamespace(index=str(index),source_protocol_dir=str(source),
            output=str(out),base_run=str(base),expected_commit=args.expected_commit,gpu=gpus[0],
            gpu_uuid=uuids[0],python=str(python)))
        assets={str(base/n):sha256_file(base/n) for n in historical_video.BASE_FILES}
        historical_f1=old["base_result"]["best_val_macro_f1"]
    else:
        assets={str(base/"tcn"/n):sha256_file(base/"tcn"/n) for n in (
            "best.pt","private_protocol.json","private_val_predictions.jsonl")}
        historical_f1=.7959464061963992
    return {"schema":CONFIG["schema"],"source_commit":args.expected_commit,"source_sha256":code_hashes(),
        "config_sha256":sha256_file(CONFIG_PATH),"training":deepcopy(CONFIG),"backbone":args.backbone,
        "index":str(index),"source_protocol_dir":str(source),"historical_run":str(base),
        "output":str(out),"python":str(python),"evidence":info,"historical_assets":assets,
        "historical_TCN_F1_reference_only":historical_f1,"variants":list(VARIANTS),"waves":layout,
        "parameters":{v:AnchorReferenceTCN(TCNConfig(**info["model_config"]),v).parameter_counts() for v in VARIANTS},
        "historical_TCN_is_replayed_but_not_used_as_initialization":True,
        "plan_only_does_not_probe_GPU_or_create_output":True,"contains_private_paths":True,
        "ddp":False,"worker_device":"cuda:0_single_UUID","test_read":False,
        "validation_role":"val823_repeated_development_not_independent_test",
        "evidence_limit":"in_sample_frozen_A_train_predictions_are_not_cross_fitted",
        "budget_caveat":"fresh_native_heads_not_the_historical_residual_TCN_recipe"}


def unchanged(p):
    if repository_state()!={"commit":p["source_commit"],"clean":True} or code_hashes()!=p["source_sha256"] \
            or sha256_file(CONFIG_PATH)!=p["config_sha256"]:
        raise RuntimeError("fixed source/config changed")
    if any(sha256_file(f)!=h for f,h in p["historical_assets"].items()):
        raise RuntimeError("historical assets changed")
    index,source=Path(p["index"]),Path(p["source_protocol_dir"])
    files={"index":index,"export_protocol":index.parent/"protocol.json","source_audit":source/"audit.json"}
    files.update({n:source/n for n in ("edges.jsonl","chains_train.jsonl","chains_val.jsonl","manifest_metadata.jsonl")})
    if {n:sha256_file(f) for n,f in files.items()}!=p["evidence"]["input_sha256"]:
        raise RuntimeError("sealed data protocol changed")


def execute(p):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("unset outer CUDA_VISIBLE_DEVICES; launcher assigns physical UUIDs")
    unchanged(p)
    slots=p["waves"][0]
    for slot in slots: wait_released_gpu(slot["gpu"],slot["gpu_uuid"])
    out=Path(p["output"]);fresh_output(out,Path(p["index"]),Path(p["source_protocol_dir"]))
    out.mkdir(mode=0o700,exist_ok=False)
    write_text(out/".launch_once",str(os.getpid())+"\n")
    write_text(out/"run_commit.txt",p["source_commit"]+"\n")
    write_json(out/"run_protocol.json",{**p,"launcher_pid":os.getpid()})
    logs=out/"launcher_logs";logs.mkdir(mode=0o700)
    write_text(logs/"pids.tsv","stage\tvariant\tpid\tphysical_gpu\tgpu_uuid\n")
    write_text(logs/"exit_codes.tsv","stage\tvariant\texit_code\n")
    exits={}

    def launch(stage,slot=None):
        unchanged(p)
        if slot is not None: wait_released_gpu(slot["gpu"],slot["gpu_uuid"])
        variant=slot["variant"] if slot else "all"
        if stage=="unit_tests":
            cmd=[p["python"],"-m","unittest","tests.test_anchor_reference","tests.test_relation_tcn_baseline","-q"]
        else:
            cmd=[p["python"],"-m","experiments.anchor_reference.worker","--stage",stage,
                 "--run-protocol",str(out/"run_protocol.json")]
            if slot:cmd += ["--variant",variant]
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",CUDA_DEVICE_ORDER="PCI_BUS_ID",
                 CUDA_VISIBLE_DEVICES="" if slot is None else slot["gpu_uuid"])
        tag=stage+"_"+variant
        stream=(logs/(tag+".log")).open("x")
        try: child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        finally:stream.close()
        append_line(logs/"pids.tsv",f"{stage}\t{variant}\t{child.pid}\t{slot['gpu'] if slot else 'cpu'}\t{slot['gpu_uuid'] if slot else 'none'}")
        print(json.dumps({"stage":stage,"variant":variant,"pid":child.pid,
                          "gpu":slot["gpu"] if slot else "cpu"}),flush=True)
        return tag,variant,child

    def collect(children):
        # Wait for all already-started workers. Never kill an unaffected peer.
        failure=[]
        for tag,variant,child in children:
            rc=child.wait();exits[tag]=rc;append_line(logs/"exit_codes.tsv",f"{tag.rsplit('_'+variant,1)[0]}\t{variant}\t{rc}")
            if rc:failure.append((tag,rc))
        if failure:raise RuntimeError(f"stage failed {failure}; preserve output and stop later waves")

    def group(stage,items):
        children=[]
        try:
            for slot in items:children.append(launch(stage,slot))
        except BaseException:
            collect(children)
            raise
        collect(children)

    try:
        atomic_json(out/"status.json",{"status":"running","stage":"unit_tests","training_completed":False})
        collect([launch("unit_tests")]);collect([launch("audit")])
        # Every registered model passes real CUDA smoke before any training.
        for i,wave in enumerate(p["waves"]):
            atomic_json(out/"status.json",{"status":"running","stage":"smoke","wave":i+1,"training_completed":False})
            group("smoke",wave)
            for slot in wave:
                if json.loads((out/"smoke"/(slot["variant"]+".json")).read_text()).get("all_passed") is not True:
                    raise RuntimeError("real CUDA smoke did not pass")
        for i,wave in enumerate(p["waves"]):
            atomic_json(out/"status.json",{"status":"running","stage":"train","wave":i+1,"training_completed":False})
            group("train",wave)
            for slot in wave:
                r=json.loads((out/slot["variant"]/"result.json").read_text())
                if not r.get("training_completed") or r["epochs_completed"]!=CONFIG["epochs"]:
                    raise RuntimeError("incomplete training despite exit0")
        evaluation_slot={**slots[0],"variant":"all"}
        group("evaluate",[evaluation_slot])
        report=json.loads((out/"evaluation/public_safe_summary.json").read_text())
        if not report.get("training_completed") or set(report["models"])!=set(VARIANTS):
            raise RuntimeError("aggregate evaluation incomplete")
        write_json(out/"suite_result.json",{"status":"complete","training_completed":True,"exit_codes":exits,"test_read":False})
        atomic_json(out/"status.json",{"status":"completed","training_completed":True,"exit_code":0})
    except BaseException as error:
        atomic_json(out/"status.json",{"status":"failed","training_completed":False,"exit_codes":exits,
                    "error_type":type(error).__name__,"error":str(error)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ("index","source-protocol-dir","historical-run","output","expected-commit","gpu-uuids"):
        parser.add_argument("--"+n,required=True)
    parser.add_argument("--backbone",choices=("videomae2","adafocus"),required=True)
    parser.add_argument("--gpus",default="0,1,2,3")
    parser.add_argument("--python",default=sys.executable)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args();p=plan(args)
    if args.execute:execute(p)
    else: print(json.dumps(p,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":main()
