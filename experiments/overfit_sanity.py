#!/usr/bin/env python3
"""Verify that each comparison model can learn one real cached FSN clip."""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch

from experiments.model_wrappers import build_model, load_shared_adafocus_weights
from experiments.pilot_data import PilotClipDataset


MODELS = ("adafocus_original", "adafocus_fsn", "mvit_v2_s_reference")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def final_ce(model: torch.nn.Module, video: torch.Tensor, target: torch.Tensor) -> tuple[float, int]:
    model.eval()
    logits = model(video)["logits"]
    return float(torch.nn.functional.cross_entropy(logits, target)), int(logits.argmax(dim=1)[0])


def run(steps: int, lr: float, seed: int, output: Path) -> dict:
    torch.set_num_threads(min(6, __import__("os").cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PilotClipDataset(
        "experiments/pilot_manifests/train.jsonl",
        "experiments/pilot_cache",
        num_frames=8,
        crop_size=224,
        verify_all=True,
    )
    item = dataset[0]
    video = item["video"].unsqueeze(0).to(device)
    target = torch.tensor([item["label"]], device=device)

    seed_all(seed)
    baseline = build_model("adafocus_original", device)
    baseline_state = {key: value.detach().cpu().clone() for key, value in baseline.state_dict().items()}
    del baseline
    gc.collect()

    result = {"steps": steps, "lr": lr, "seed": seed, "target": int(target), "models": {}}
    for name in MODELS:
        print(f"overfit {name}", flush=True)
        seed_all(seed)
        model = build_model(name, device)
        if name == "adafocus_original":
            model.load_state_dict(baseline_state)
        elif name == "adafocus_fsn":
            load_shared_adafocus_weights(model, baseline_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        initial_ce, initial_prediction = final_ce(model, video, target)
        native_losses = []
        for _ in range(steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            model_output = model(video)
            loss = model.compute_loss(model_output, target)
            loss.backward()
            optimizer.step()
            native_losses.append(float(loss.detach()))
        final_loss, final_prediction = final_ce(model, video, target)
        gates = {}
        if name == "adafocus_fsn":
            gates = {
                "adapter_alpha": float(model.core.local_CNN.local_adapter.alpha.detach()),
                "interaction_beta": float(model.core.fsn_interaction.beta.detach()),
                "interaction_gamma": float(model.core.fsn_interaction.gamma.detach()),
            }
        result["models"][name] = {
            "initial_final_ce": initial_ce,
            "final_final_ce": final_loss,
            "ce_ratio": final_loss / max(initial_ce, 1e-12),
            "initial_prediction": initial_prediction,
            "final_prediction": final_prediction,
            "native_loss_first": native_losses[0],
            "native_loss_last": native_losses[-1],
            "gates": gates,
        }
        del model, optimizer
        gc.collect()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/overfit_sanity.json"))
    args = parser.parse_args()
    if args.steps <= 0 or args.lr <= 0:
        parser.error("steps and lr must be positive")
    print(json.dumps(run(args.steps, args.lr, args.seed, args.output), ensure_ascii=False, indent=2))
