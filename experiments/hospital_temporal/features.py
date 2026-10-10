"""Export/load only the completed NEW hospital A, never the old FSN evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.hospital_holdout.prepare import EXPECTED_COUNTS, sha
from experiments.hospital_holdout.train import SCHEMA as A_SCHEMA, TOTAL_UPDATES, SELECTION
from experiments.hospital_holdout.video_data import HospitalDataset, manifest_rows
from experiments.metrics import compute_classification_metrics
from experiments.modern_baselines import videomae2 as visual
from experiments.modern_baselines.train_videomae2 import _atomic_json, _atomic_torch_save, _seed_all
from experiments.relation.videomae2_followup import FrozenVideoMAE2Evidence
from .prepare import audit

SCHEMA = "fsn-hospital-A-frozen-pooled-features-v1"


def completed_A(a_run):
    a_run = Path(a_run).resolve()
    suite = json.loads((a_run/"suite_result.json").read_text())
    result = json.loads((a_run/"train/result.json").read_text())
    replay = json.loads((a_run/"evaluation/result.json").read_text())
    if suite.get("status") != "complete" or suite.get("training_completed") is not True \
            or suite.get("evaluation_completed") is not True \
            or result.get("training_completed") is not True or result.get("optimizer_steps") != TOTAL_UPDATES \
            or result.get("selection") != SELECTION or replay.get("dev_best_replayed") is not True:
        raise ValueError("new A must complete all registered training and best-checkpoint replay")
    for stage in ("unit_tests", "seal_data", "train_cache", "val_cache", "smoke", "train", "evaluation"):
        if (a_run/f"launcher_logs/{stage}.exit_code").read_text().strip() != "0":
            raise ValueError("an A stage has no verified exit0")
    digest = sha(a_run/"train/best.pt")
    if digest != result.get("best_checkpoint_sha256") or digest != replay.get("best_checkpoint_sha256"):
        raise ValueError("selected new A checkpoint changed")
    # Full protocol, timestamp and institution-partition checks, not just a count.
    audit(a_run/"protocol")
    return result, digest


def export(a_run, official_source, pretrained, output):
    a_run, output = Path(a_run).resolve(), Path(output).resolve()
    result, digest = completed_A(a_run)
    repo = Path(__file__).resolve().parents[2]
    if output.exists() or any(p == output or p in output.parents or output in p.parents
                             for p in (a_run, repo)):
        raise ValueError("fresh feature output outside code and A run required")
    payload = torch.load(a_run/"train/best.pt", map_location="cpu", weights_only=True)
    if payload.get("schema") != A_SCHEMA or payload.get("selection") != SELECTION \
            or payload.get("checkpoint_epoch") != result.get("best_epoch") \
            or payload["identity"]["protocol_sha256"] != sha(a_run/"protocol/protocol.json"):
        raise ValueError("checkpoint is not the new hospital A")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one mapped CUDA GPU is required")
    _seed_all(42)
    model, load = visual.build_pretrained_model(official_source, pretrained)
    model.load_state_dict(payload["model"], strict=True)
    model._fsn_finetuned_checkpoint_sha256 = digest
    # Reuse only the audited token-capture helper; no R or D stage is called.
    encoder = FrozenVideoMAE2Evidence(model).to("cuda:0").eval()
    output.mkdir(parents=True, exist_ok=False)
    matrix, labels, identifiers, a_metrics = [], [], [], {}
    for split in ("train", "val"):
        dataset = HospitalDataset(a_run/"protocol", a_run/f"{split}_cache", split)
        scores, targets = [], []
        for number, batch in enumerate(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4,
                                                  pin_memory=True)):
            video = batch["video"].to("cuda:0")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                value = encoder(video)
                independent_logits = encoder.model(video)
            if not torch.equal(value["logits"], independent_logits):
                raise ValueError("every exported clip must replay the same A logits independently")
            vector = torch.cat((value["global_tokens"].float().mean(1),
                                value["local_tokens"].float().mean(1),
                                value["logits"].float()), -1).cpu()
            if vector.shape != (len(batch["label"]), 1543) or not torch.isfinite(vector).all():
                raise ValueError("invalid pooled VideoMAE mean/max/raw-logit features")
            matrix.append(vector); labels.append(batch["label"]); identifiers.extend(batch["clip_id"])
            scores.append(value["logits"].float().cpu()); targets.append(batch["label"])
            if (number+1) % 100 == 0:
                print(json.dumps({"split": split, "batch": number+1, "clips_exported": sum(len(v) for v in targets)}), flush=True)
        a_metrics[split] = compute_classification_metrics(torch.cat(scores), torch.cat(targets), dataset.raw_rows)
    if a_metrics["val"]["all"] != result["best_metrics"]["all"]:
        raise ValueError("new888 export does not reproduce A's selected development metrics")
    _atomic_torch_save(output/"private_features.pt", {"schema": SCHEMA, "features": torch.cat(matrix),
                       "labels": torch.cat(labels), "clip_ids": identifiers})
    seal = {"schema": SCHEMA, "status": "complete", "counts": EXPECTED_COUNTS,
            "protocol_sha256": sha(a_run/"protocol/protocol.json"), "A_checkpoint_sha256": digest,
            "features_sha256": sha(output/"private_features.pt"), "A_metrics": a_metrics,
            "A_frozen": True, "every_clip_independent_logit_replay": True,
            "input": "temporal_mean_spatial_mean768+temporal_mean_spatial_max768+raw_A_logits7",
            "pretrained_load": load, "R_D_enabled": False, "independent_test_evaluated": False}
    _atomic_json(output/"seal.json", seal)
    print(json.dumps({"status": "complete", "counts": EXPECTED_COUNTS,
                      "A_train_macro_f1": a_metrics["train"]["all"]["macro_f1"],
                      "A_development_macro_f1": a_metrics["val"]["all"]["macro_f1"]}), flush=True)


def load(a_run, feature_dir):
    a_run, feature_dir = Path(a_run), Path(feature_dir)
    result, digest = completed_A(a_run)
    seal = json.loads((feature_dir/"seal.json").read_text())
    if seal.get("schema") != SCHEMA or seal.get("status") != "complete" \
            or seal.get("counts") != EXPECTED_COUNTS or seal.get("A_frozen") is not True \
            or seal.get("every_clip_independent_logit_replay") is not True \
            or seal.get("R_D_enabled") is not False or seal.get("A_checkpoint_sha256") != digest \
            or seal.get("protocol_sha256") != sha(a_run/"protocol/protocol.json") \
            or seal.get("features_sha256") != sha(feature_dir/"private_features.pt"):
        raise ValueError("frozen evidence does not belong to the completed new hospital A")
    payload = torch.load(feature_dir/"private_features.pt", map_location="cpu", weights_only=True)
    records = manifest_rows(a_run/"protocol", "train")[0] + manifest_rows(a_run/"protocol", "val")[0]
    matrix, labels = payload["features"], payload["labels"]
    count = sum(EXPECTED_COUNTS.values())
    if payload.get("schema") != SCHEMA or matrix.shape != (count,1543) or matrix.dtype != torch.float32 \
            or labels.shape != (count,) or labels.dtype != torch.int64 or not torch.isfinite(matrix).all() \
            or payload["clip_ids"] != [r["clip_id"] for r in records] \
            or labels.tolist() != [r["label_id"] for r in records]:
        raise ValueError("feature contents/order/labels differ from sealed manifests")
    for split in ("train", "val"):
        ids = [i for i, r in enumerate(records) if r["split"] == split]
        metrics = compute_classification_metrics(matrix[ids,-7:],labels[ids],[records[i] for i in ids])
        if metrics != seal["A_metrics"][split]:
            raise ValueError("numeric frozen A metrics differ from the feature seal")
        if split == "val" and metrics["all"] != result["best_metrics"]["all"]:
            raise ValueError("frozen development metrics do not replay the selected new A")
    return matrix, labels, records, seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("a-run", "official-source", "pretrained", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-commit",required=True)
    args = parser.parse_args()
    from .run import verify_code
    verify_code(args.expected_commit)
    export(args.a_run,args.official_source,args.pretrained,args.output)


if __name__ == "__main__":
    main()
