"""Read-only plan by default; explicit execution is a one-GPU sealed pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.hospital_holdout import prepare as data
from experiments.hospital_holdout.train import public_config
from experiments.hospital_holdout.video_data import PREPROCESS
from experiments.modern_baselines import videomae2 as base

REPO = Path(__file__).resolve().parents[2]
SPATIAL_FILES = {
    "dataset/video_transforms.py": "6a96a18e6ba4288bd5ca995b654a9b08f9151e0258224cff5e282df28e02fcaf",
    "dataset/volume_transforms.py": "a08ba6e014cf9d2ba3e9ac54538d586a6b94c9c1cbb344221e1d508f1092ea81",
    "dataset/functional.py": "f5656b83581538af329596284717d7e2e979cead5667959f4817f7f1f71fd08f",
}


def verify_code(expected):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],cwd=REPO,text=True).strip()
    dirty = subprocess.check_output(["git","status","--porcelain","--untracked-files=all"],
                                    cwd=REPO,text=True).strip()
    if commit != expected or dirty:
        raise ValueError("fixed checkout commit mismatch or working tree not clean")
    return commit


def available_gpu():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"],text=True)
    rows = [r.split(",") for r in text.splitlines() if r.strip()]
    if len(rows) != 1:
        raise ValueError("this launcher is registered for exactly one physical GPU")
    index, uuid, name, memory, util = [s.strip() for s in rows[0]]
    apps = subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                                    "--format=csv,noheader,nounits"],text=True).strip()
    if index != "0" or "RTX 3090" not in name or apps or int(memory)>64 or int(util)>5:
        raise ValueError("GPU0 must be an idle RTX3090 without compute processes; do not stop others")
    # A small driver-only reservation (often 1MiB) is not a running experiment.
    return {"physical_index":0, "uuid":uuid,"name":name,"idle_memory_mib":int(memory),
            "internal_device":"cuda:0","DDP":False}


def pipeline(args):
    commit = verify_code(args.expected_commit)
    base.verify_official_source(args.official_source)
    checkpoint = base.verify_official_checkpoint(args.pretrained)
    for path, digest in SPATIAL_FILES.items():
        if data.sha(args.official_source/path) != digest:
            raise ValueError("pinned official spatial preprocessing source changed")
    if not Path(sys.executable).resolve().samefile(args.python.resolve()):
        raise ValueError("launcher must itself run using the selected CUDA Python")
    output = args.output.resolve()
    protected = [REPO,args.source_manifest_dir.resolve(),args.records.resolve().parent]
    if output.exists() or any(output==p or p in output.parents or output in p.parents for p in protected):
        raise ValueError("output must be new and outside code/source manifests")
    summary, detail, derived = data.prepare(args.source_manifest_dir,args.records,args.events,
                                           args.data_root,args.unit_map,args.clip_terminal_overrun)
    plan = {"read_only":not args.execute,"code_commit":commit,"data":summary,
            "config":public_config(),"preprocessing":PREPROCESS,"pretrained":checkpoint,
            "GPU_plan":"physical GPU0, UUID resolved only at execution, independent single GPU",
            "output":str(output),"Menzhen_used_for_selection":True,"independent_test_evaluated":False,
            "R_D_enabled":False,
            "stages":["unit_tests","seal_data","train_cache","val_cache","CUDA_three_steps","train105","dev_best_replay"]}
    print(json.dumps(plan,ensure_ascii=False),flush=True)
    if summary["status"] != "verified":
        raise ValueError("time/data audit blocked; no output created, no guessed units or training")
    if not args.execute:
        return
    gpu = available_gpu()
    output.mkdir(parents=True,exist_ok=False)
    (output/".launch_once").open("xb").close()
    logs = output/"launcher_logs"
    logs.mkdir()
    identity = {**plan,"read_only":False,"launcher_pid":os.getpid(),"GPU":gpu,
                "python":str(args.python),"input_sha256":{
                "records":data.sha(args.records),"events":data.sha(args.events),
                "unit_map":None if args.unit_map is None else data.sha(args.unit_map)}}
    (output/"run_protocol.json").write_bytes(data.encoded(identity))
    env = {**os.environ,"PYTHONDONTWRITEBYTECODE":"1","CUDA_VISIBLE_DEVICES":gpu["uuid"],
           "FSN_OFFICIAL_VIDEO_SOURCE":str(args.official_source)}

    def stage(name, command):
        print(json.dumps({"stage":name,"event":"starting","launcher_pid":os.getpid()}),flush=True)
        with (logs/f"{name}.log").open("xb") as log:
            child = subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            with (logs/"pids.tsv").open("a") as stream:
                stream.write(f"{name}\t{child.pid}\t{gpu['uuid']}\n")
            (output/"status.json").write_bytes(data.encoded({"phase":name,"pid":child.pid,
                        "launcher_pid":os.getpid(),"GPU":gpu,"training_completed":False}))
            code = child.wait()
        (logs/f"{name}.exit_code").write_text(str(code)+"\n")
        with (logs/"exit_codes.tsv").open("a") as stream:
            stream.write(f"{name}\t{code}\n")
        if code:
            raise RuntimeError(f"stage {name} exited {code}; preserve files, do not retry")

    py = str(args.python)
    try:
        stage("unit_tests",[py,"-m","pytest","-q","tests/test_hospital_holdout.py",
                           "tests/test_build_dataset_duplicates.py","tests/test_videomae2_baseline.py",
                           "tests/test_metrics.py","-p","no:cacheprovider"])
        protocol = output/"protocol"
        data.write_prepared(protocol,args.source_manifest_dir,args.records,detail,derived)
        (logs/"seal_data.exit_code").write_text("0\n")
        def cache(split):
            stage(f"{split}_cache",[py,"-m","experiments.hospital_holdout.video_data",
                  "--protocol-dir",str(protocol),"--split",split,"--output",str(output/f"{split}_cache"),"--execute"])
        cache("train")
        cache("val")
        # No other job may have claimed the card during lengthy CPU preparation.
        if available_gpu()["uuid"] != gpu["uuid"]:
            raise ValueError("GPU mapping changed during preparation")
        def action(name,mode,split="train",extra=()):
            stage(name,[py,"-m","experiments.hospital_holdout.train","--stage",mode,
                  "--protocol-dir",str(protocol),"--cache-dir",str(output/f"{split}_cache"),
                  "--official-source",str(args.official_source),"--pretrained",str(args.pretrained),
                  "--output",str(output/name),"--expected-commit",commit,*extra])
        action("smoke","smoke")
        report=json.loads((output/"smoke/cuda_smoke.json").read_text())
        if report.get("all_passed") is not True:
            raise ValueError("CUDA three-step smoke did not pass")
        action("train","train",extra=("--val-cache",str(output/"val_cache")))
        result=json.loads((output/"train/result.json").read_text())
        if result.get("training_completed") is not True or result.get("optimizer_steps")!=12075:
            raise ValueError("training exited without completing the registered budget")
        action("evaluation","evaluate","val",("--train-dir",str(output/"train")))
        (output/"suite_result.json").write_bytes(data.encoded({"status":"complete",
               "training_completed":True,"evaluation_completed":True,"GPU":gpu}))
        (output/"status.json").write_bytes(data.encoded({"status":"complete",
               "training_completed":True,"evaluation_completed":True}))
    except BaseException as error:
        (output/"status.json").write_bytes(data.encoded({"status":"failed",
              "error_type":type(error).__name__,"training_completed":(output/"train/result.json").exists(),
              "automatic_retry":False}))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest-dir",type=Path,required=True)
    p.add_argument("--records",type=Path,required=True)
    p.add_argument("--events",type=Path,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--unit-map",type=Path)
    p.add_argument("--official-source",type=Path,required=True)
    p.add_argument("--pretrained",type=Path,required=True)
    p.add_argument("--python",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--expected-commit",required=True)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--clip-terminal-overrun",action="store_true",
                   help="explicit owner authorization for the four audited video-tail overruns")
    pipeline(p.parse_args())


if __name__=="__main__":
    main()
