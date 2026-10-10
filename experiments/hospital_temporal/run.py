"""Read-only plan; explicit execution of six independently trained T/T′ models.

Never change/restart A. A pending run is a dependency, not an available frozen
feature source. Waiting is opt-in; any failed dependency/stage stops this suite.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.hospital_holdout.prepare import encoded, sha
from experiments.hospital_holdout.run import available_gpu, verify_code
from experiments.modern_baselines import videomae2 as visual
from experiments.modern_baselines.train_videomae2 import _atomic_json
from .features import completed_A
from .model import CONFIG, VARIANTS
from .prepare import audit

REPO = Path(__file__).resolve().parents[2]
EXPECTED_A_COMMIT = "af93d380fc8b9b64c34a36f7dcd2ca48de64e104"
EXPECTED_PROTOCOL_SHA = "45797864698785a38ff484d8173b399814e97444280756153071272b782b1d84"


def dependency_state(a_run):
    """Read sidecars and verify the live dependency; never probe/claim its GPU."""
    a_run = Path(a_run).resolve()
    protocol = json.loads((a_run/"run_protocol.json").read_text())
    if protocol.get("code_commit") != EXPECTED_A_COMMIT:
        raise ValueError("A dependency is not the registered fresh hospital baseline")
    if sha(a_run/"protocol/protocol.json") != EXPECTED_PROTOCOL_SHA:
        raise ValueError("registered corrected hospital manifest/protocol seal changed")
    status = json.loads((a_run/"status.json").read_text())
    if status.get("status") == "failed":
        raise ValueError("A failed: preserve it, do not train temporal models or restart A")
    external_exit = Path(str(a_run)+".launcher.exit_code")
    if external_exit.exists() and external_exit.read_text().strip() != "0":
        raise ValueError("A launcher exited abnormally")
    suite = a_run/"suite_result.json"
    if suite.exists():
        completed_A(a_run)
        return {"ready":True,"phase":"complete"}
    pid = protocol.get("launcher_pid")
    if not isinstance(pid,int) or isinstance(pid,bool) or pid<=0:
        raise ValueError("A has neither completed sidecars nor a verified live launcher")
    cmd = subprocess.run(["ps","-p",str(pid),"-o","args="],capture_output=True,text=True,check=True).stdout
    if "experiments.hospital_holdout.run" not in cmd or str(a_run) not in cmd:
        raise ValueError("A launcher is no longer the registered live dependency")
    return {"ready":False,"phase":status.get("phase","unknown"),"launcher_pid":pid}


def fresh_output(output, a_run, official_source):
    output = Path(output).resolve()
    roots = (REPO,Path(a_run).resolve(),Path(official_source).resolve())
    if output.exists() or not output.parent.is_dir() or output==Path(output.anchor) \
            or any(output==p or p in output.parents or output in p.parents for p in roots):
        raise ValueError("fresh private output with existing parent outside all input/code trees required")
    return output


def plan(args):
    commit = verify_code(args.expected_commit)
    output = fresh_output(args.output,args.a_run,args.official_source)
    if not Path(sys.executable).resolve().samefile(Path(args.python).resolve()):
        raise ValueError("run the launcher itself with the registered CUDA Python")
    visual.verify_official_source(args.official_source)
    weight = visual.verify_official_checkpoint(args.pretrained)
    data,_ = audit(Path(args.a_run)/"protocol")
    state = dependency_state(args.a_run)
    from .model import TemporalVariant
    return {"read_only":not args.execute,"code_commit":commit,"config":CONFIG,
        "config_sha256":sha(REPO/"experiments/hospital_temporal/config.json"),
        "A_run":str(Path(args.a_run).resolve()),"A_fixed_code_commit":EXPECTED_A_COMMIT,
        "A_dependency":state,"protocol_sha256":data["protocol_sha256"],
        "manifest_sha256":data["manifest_sha256"],"chain_counts":data["splits"],
        "pretrained":weight,"output":str(output),
        "GPU_plan":{"expected_uuid":args.gpu_uuid,"physical_index":0,"DDP":False,
                    "probed_at_execution_only":True},
        "variants":{v:TemporalVariant(v).parameter_counts() for v in VARIANTS},
        "steps_per_variant":CONFIG["training"]["epochs"]*((data["splits"]["train"]["segments"]+15)//16),
        "Menzhen_role":"development_validation_for_checkpoint_and_model_selection",
        "independent_test_evaluated":False,"R_D_enabled":False,
        "stages":["unit_tests","new_A_features","CUDA_three_steps",*VARIANTS,"evaluation"],
        "temporal_training_started":False}


def await_dependency(args, root):
    deadline = time.monotonic()+args.wait_hours*3600
    previous = None
    while True:
        state = dependency_state(args.a_run)
        if state["ready"]: return
        if not args.wait_for_A:
            raise ValueError("A is incomplete; explicit --wait-for-A required for a CPU-only queue")
        if time.monotonic()>=deadline:
            raise TimeoutError("dependency wait deadline exceeded; A remains unchanged")
        if state!=previous:
            _atomic_json(root/"status.json",{"status":"waiting_for_A","A_dependency":state,
                "launcher_pid":os.getpid(),"GPU_claimed":False,"training_completed":False})
            print(json.dumps({"status":"waiting_for_A","A_dependency":state}),flush=True)
            previous = state
        time.sleep(30)


def execute(args, description):
    root = fresh_output(args.output,args.a_run,args.official_source)
    if not description["A_dependency"]["ready"] and not args.wait_for_A:
        raise ValueError("A is incomplete: no output or temporal job created")
    root.mkdir(exist_ok=False)
    (root/".launch_once").open("xb").close()
    logs = root/"launcher_logs"; logs.mkdir()
    protocol = {**description,"read_only":False,"launcher_pid":os.getpid(),
        "python":str(args.python),"waiting_allowed":args.wait_for_A,
        "automatic_retry":False,"training_completed":False}
    _atomic_json(root/"run_protocol.json",protocol)
    env = {**os.environ,"PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"4",
           "FSN_OFFICIAL_VIDEO_SOURCE":str(args.official_source)}
    gpu = None

    def gpu_check():
        value = available_gpu()
        if value["uuid"]!=args.gpu_uuid:
            raise ValueError("physical GPU UUID differs; do not take another GPU")
        return value

    def stage(name, command, use_gpu=True):
        verify_code(args.expected_commit)
        if sha(REPO/"experiments/hospital_temporal/config.json")!=description["config_sha256"]:
            raise ValueError("runtime config changed")
        if sha(Path(args.a_run)/"protocol/protocol.json")!=description["protocol_sha256"]:
            raise ValueError("A data dependency changed")
        if use_gpu: gpu_check()
        child_env = {**env,"CUDA_VISIBLE_DEVICES":args.gpu_uuid if use_gpu else ""}
        with (logs/f"{name}.log").open("xb") as stream:
            child = subprocess.Popen(command,cwd=REPO,env=child_env,stdout=stream,stderr=subprocess.STDOUT)
            with (logs/"pids.tsv").open("a") as pid_file:
                pid_file.write(f"{name}\t{child.pid}\t{args.gpu_uuid if use_gpu else 'CPU'}\n")
            _atomic_json(root/"status.json",{"status":"running","phase":name,"pid":child.pid,
                "launcher_pid":os.getpid(),"GPU":gpu if use_gpu else None,"training_completed":False})
            print(json.dumps({"stage":name,"pid":child.pid,"GPU":gpu if use_gpu else None}),flush=True)
            code = child.wait()
        (logs/f"{name}.exit_code").write_text(str(code)+"\n")
        with (logs/"exit_codes.tsv").open("a") as stream: stream.write(f"{name}\t{code}\n")
        if code: raise RuntimeError(f"stage {name} exited {code}; no automatic retry or subsequent variants")

    py = str(args.python)
    try:
        await_dependency(args,root)
        result,checkpoint_sha = completed_A(args.a_run)
        gpu = gpu_check()
        time.sleep(2)
        gpu_check()
        protocol.update(GPU=gpu,A_checkpoint_sha256=checkpoint_sha,A_best_epoch=result["best_epoch"])
        _atomic_json(root/"run_protocol.json",protocol)
        stage("unit_tests",[py,"-m","pytest","-q","tests/test_hospital_temporal.py",
            "tests/test_hospital_temporal_models.py","tests/test_hospital_temporal_pipeline.py",
            "tests/test_metrics.py","-p","no:cacheprovider"],False)
        feature_dir = root/"features"
        stage("new_A_features",[py,"-m","experiments.hospital_temporal.features",
            "--a-run",str(args.a_run),"--official-source",str(args.official_source),
            "--pretrained",str(args.pretrained),"--output",str(feature_dir),
            "--expected-commit",args.expected_commit])
        def action(name,mode,output,extra=()):
            stage(name,[py,"-m","experiments.hospital_temporal.worker","--stage",mode,
                "--a-run",str(args.a_run),"--features",str(feature_dir),"--output",str(output),
                "--expected-commit",args.expected_commit,*extra])
        action("smoke","smoke",root/"smoke")
        if json.loads((root/"smoke/cuda_smoke.json").read_text()).get("all_passed") is not True:
            raise ValueError("real frozen-feature CUDA three-step checks failed")
        for variant in VARIANTS:
            action(variant,"train",root/variant,("--variant",variant))
            item = json.loads((root/variant/"result.json").read_text())
            if item.get("training_completed") is not True or item.get("epochs_completed")!=100:
                raise ValueError("stage exit0 alone is not a completed registered temporal run")
        action("evaluation","evaluate",root)
        public = json.loads((root/"public_safe_summary.json").read_text())
        if set(public["variants"])!=set(VARIANTS):
            raise ValueError("aggregate evaluation incomplete")
        _atomic_json(root/"suite_result.json",{"status":"complete","training_completed":True,
            "evaluation_completed":True,"code_commit":args.expected_commit,"GPU":gpu})
        _atomic_json(root/"status.json",{"status":"complete","training_completed":True,
            "evaluation_completed":True})
    except BaseException as error:
        _atomic_json(root/"status.json",{"status":"failed","error_type":type(error).__name__,
            "training_completed":False,"automatic_retry":False,"A_modified":False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("a-run","official-source","pretrained","python","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--gpu-uuid",required=True)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--wait-for-A",action="store_true",
        help="explicit CPU-only dependency queue, never train/export until new A completes")
    parser.add_argument("--wait-hours",type=float,default=96)
    args = parser.parse_args(argv)
    if not 0<args.wait_hours<=96: parser.error("wait deadline must be in (0,96] hours")
    description = plan(args)
    print(json.dumps(description,ensure_ascii=False),flush=True)
    if args.execute: execute(args,description)


if __name__=="__main__":
    main()
