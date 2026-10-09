"""Default read-only plan; one sealed independent CUDA job with --execute."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from experiments.context_evidence import run as context_run
from experiments.direction_gate import run as base_run
from experiments.temporal_heads import run as heads_run
from experiments.relation.data import sha256_file
from experiments.videomae_tcn.io import ROOT, write_json, write_text, atomic_json, append_line, repository_state
from experiments.videomae_tcn.run import fresh_output, wait_released_gpu
from . import legacy


def source_hashes():
    names = [str(f.relative_to(ROOT)) for f in (ROOT / "experiments/paper_validation").glob("*.py")]
    names += ["tests/test_paper_validation.py"]
    return {**context_run.hashes(), **heads_run.hashes(), **{n:sha256_file(ROOT/n) for n in names}}


def completed(directory):
    s = json.loads((directory / "suite_result.json").read_text())
    if s.get("status") != "complete" or not s.get("training_completed") or not s.get("exit_codes") or any(s["exit_codes"].values()):
        raise ValueError("existing suite is not fully completed")


def audit_entries(args, evidence):
    root = Path(args.multiseed_root).resolve()
    completed(root)
    master = json.loads((root / "run_protocol.json").read_text())
    directories = [(42, Path(master["seed42_run"]))] + [(c["training"]["seed"], Path(c["output"])) for c in master["children"]]
    entries, assets = [], [root / "run_protocol.json", root / "suite_result.json"]
    seen = set()
    for seed, directory in directories:
        completed(directory)
        p = json.loads((directory / "run_protocol.json").read_text())
        if p["evidence"] != evidence or p["training"]["seed"] != seed:
            raise ValueError("paired suite is not identical frozen evidence")
        assets += [directory / "run_protocol.json", directory / "suite_result.json"]
        for v in ("legacy_continue", "context_delta_continue"):
            if v not in p["variants"] or (seed,v) in seen: continue
            seen.add((seed,v)); folder = directory / v
            r = json.loads((folder / "result.json").read_text())
            if not r["training_completed"] or r["epochs_completed"] != 30 or r["optimizer_steps"] != 5910 or \
                    sha256_file(folder / "best.pt") != r["best_sha256"]:
                raise ValueError("completed paired checkpoint/budget mismatch")
            entries.append({"name": f"seed{seed}_{v}", "kind": "context", "seed": seed,
                            "variant": v, "directory": str(directory)})
            assets += [folder/n for n in ("best.pt", "result.json", "history.json")]
            assets += [directory / "evaluation" / (v+"_private_val_predictions.jsonl")]
    if seen != {(s,v) for s in (42,10,1217) for v in ("legacy_continue","context_delta_continue")}:
        raise ValueError("three paired seeds incomplete")
    directory = Path(args.mature_run).resolve()
    p = json.loads((directory / "run_protocol.json").read_text())
    if p["evidence"] != evidence: raise ValueError("mature head evidence mismatch")
    exits = dict(line.split("\t") for line in (directory/"launcher_logs/exit_codes.tsv").read_text().splitlines()[1:])
    for v in ("plain_tcn_fresh", "mstcnpp"):
        r = json.loads((directory/v/"result.json").read_text())
        if exits.get("train_"+v) != "0" or not r["training_completed"] or r["epochs_completed"] != 30 or \
                r["optimizer_steps"] != 5910 or sha256_file(directory/v/"best.pt") != r["best_sha256"]:
            raise ValueError("mature head not independently completed")
        entries.append({"name": v, "kind": "head", "variant": v, "directory": str(directory), "seed": 42})
        assets += [directory/v/n for n in ("best.pt", "result.json", "history.json")]
    assets += [directory/"run_protocol.json", directory/"launcher_logs/exit_codes.tsv"]
    return entries, assets


def plan(args):
    if not re.fullmatch(r"[a-f0-9]{40}", args.expected_commit) or repository_state() != {"commit":args.expected_commit,"clean":True}:
        raise ValueError("require exact clean committed independent worktree")
    index, source, out = map(lambda p:Path(p).resolve(), (args.index,args.source_protocol_dir,args.output))
    fresh_output(out,index,source)
    base=Path(args.base_run).resolve()
    if out.is_relative_to(base) or base.is_relative_to(out):
        raise ValueError("new output overlaps completed base")
    if args.gpu < 0 or not re.fullmatch(r"GPU-[a-fA-F0-9-]{8,}",args.gpu_uuid) or not Path(args.python).is_file():
        raise ValueError("real physical GPU UUID and Python required")
    if args.mode == "audit":
        p = base_run.plan(args)
        entries, assets = audit_entries(args,p["evidence"])
        stages = ["audit"]
    else:
        _, _, info = legacy.prepare(index,source,args.base_run)
        assets = [Path(args.base_run)/"tcn"/n for n in ("best.pt","private_protocol.json","private_val_predictions.jsonl")]
        from experiments.context_evidence.run import TRAINING
        p = {"evidence":info,"training":{**TRAINING,"augmentation":"none_unchanged_anchor_second_view"},
             "variants":[args.variant],"base_run":str(Path(args.base_run).resolve()),
             "base_sha256":{f.name:sha256_file(f) for f in assets}}
        entries=[]; stages=["legacy_audit","legacy_smoke","train_"+args.variant,"legacy_evaluate"]
    assets += [Path(args.base_run)/n for n in base_run.BASE_FILES] if args.mode == "audit" else []
    commands = {"unit_tests":[args.python,"-m","unittest","tests.test_relation_tcn_baseline",
                 "tests.test_context_evidence","tests.test_paper_validation","-q"]}
    commands.update({s:[args.python,"-m","experiments.paper_validation.worker","--stage",s,
                       "--run-protocol",str(out/"run_protocol.json")] for s in stages})
    return {**p,"source_commit":args.expected_commit,"source_sha256":source_hashes(),
        "index":str(index),"source_protocol_dir":str(source),"output":str(out),"python":args.python,
        "gpu_physical_index":args.gpu,"gpu_uuid":args.gpu_uuid,"mode":args.mode,"entries":entries,
        "artifacts":{str(f.resolve()):sha256_file(f) for f in assets},"commands":commands,
        "test_read":False,"validation_role":"val823_development_not_independent_test",
        "plan_only_does_not_probe_gpu":True,"ddp":False,"A_trained":False}


def unchanged(p):
    if repository_state() != {"commit":p["source_commit"],"clean":True} or source_hashes() != p["source_sha256"]:
        raise RuntimeError("immutable code changed")
    if any(sha256_file(Path(f)) != h for f,h in p["artifacts"].items()):
        raise RuntimeError("completed source artifacts changed")
    index, source = Path(p["index"]),Path(p["source_protocol_dir"])
    paths={"index":index,"export_protocol":index.parent/"protocol.json","source_audit":source/"audit.json"}
    paths.update({n:source/n for n in ("edges.jsonl","chains_train.jsonl","chains_val.jsonl","manifest_metadata.jsonl")})
    if {n:sha256_file(f) for n,f in paths.items()} != p["evidence"]["input_sha256"]:
        raise RuntimeError("input seals changed")


def execute(p):
    if "CUDA_VISIBLE_DEVICES" in os.environ: raise RuntimeError("unset outer CUDA_VISIBLE_DEVICES")
    unchanged(p); wait_released_gpu(p["gpu_physical_index"],p["gpu_uuid"])
    out=Path(p["output"]); fresh_output(out,Path(p["index"]),Path(p["source_protocol_dir"]))
    out.mkdir(mode=0o700,exist_ok=False)
    write_text(out/".launch_once",str(os.getpid())+"\n")
    write_json(out/"run_protocol.json",{**p,"launcher_pid":os.getpid()})
    logs=out/"launcher_logs";logs.mkdir(mode=0o700)
    write_text(logs/"pids.tsv","stage\tpid\tphysical_gpu\tgpu_uuid\n")
    write_text(logs/"exit_codes.tsv","stage\texit_code\n")
    exits={}
    try:
        for stage,cmd in p["commands"].items():
            unchanged(p)
            if stage!="unit_tests":wait_released_gpu(p["gpu_physical_index"],p["gpu_uuid"])
            mapping="cpu" if stage=="unit_tests" else str(p["gpu_physical_index"])
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="" if stage=="unit_tests" else p["gpu_uuid"])
            with (logs/(stage+".log")).open("x") as stream:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                append_line(logs/"pids.tsv",f"{stage}\t{child.pid}\t{mapping}\t{p['gpu_uuid']}")
                atomic_json(out/"status.json",{"status":"running","stage":stage,"launcher_pid":os.getpid(),"worker_pid":child.pid,"physical_gpu":mapping,"gpu_uuid":p["gpu_uuid"]})
                print(json.dumps({"stage":stage,"pid":child.pid,"gpu":mapping}),flush=True)
                exits[stage]=child.wait()
            append_line(logs/"exit_codes.tsv",f"{stage}\t{exits[stage]}")
            if exits[stage] != 0:raise RuntimeError(f"{stage} failed; no automatic retries")
            if stage == "legacy_smoke" and not json.loads((out/"cuda_smoke.json").read_text()).get("all_passed"):
                raise RuntimeError("real CUDA smoke did not pass")
            if stage.startswith("train_") and not json.loads((out/stage[6:]/"result.json").read_text()).get("training_completed"):
                raise RuntimeError("formal result not complete despite exit0")
        unchanged(p)
        write_json(out/"suite_result.json",{"status":"complete","training_completed":p["mode"]=="legacy","audit_completed":True,"exit_codes":exits,"test_read":False})
        atomic_json(out/"status.json",{"status":"completed","exit_code":0})
    except BaseException as e:
        atomic_json(out/"status.json",{"status":"failed","exit_codes":exits,"error":str(e)})
        raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ("index","source-protocol-dir","base-run","output","expected-commit","gpu-uuid"):
        ap.add_argument("--"+n,required=True)
    ap.add_argument("--mode",choices=("audit","legacy"),required=True)
    ap.add_argument("--multiseed-root");ap.add_argument("--mature-run")
    ap.add_argument("--variant",choices=("legacy_continue","context_delta_continue"))
    ap.add_argument("--gpu",type=int,required=True);ap.add_argument("--python",default=sys.executable)
    ap.add_argument("--execute",action="store_true")
    a=ap.parse_args()
    if a.mode=="audit" and (not a.multiseed_root or not a.mature_run):ap.error("audit requires completed multi-seed and mature-head directories")
    if a.mode=="legacy" and not a.variant:ap.error("legacy requires one paired variant")
    p=plan(a)
    if a.execute:execute(p)
    else:print(json.dumps(p,sort_keys=True,indent=2))


if __name__=="__main__":main()
