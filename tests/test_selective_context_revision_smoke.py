"""Fixtures for the historical-unit gate smoke (not a real CUDA certificate)."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from experiments.selective_context.revision_smoke import (
    choose_informative_multi, main, parser, run_checks, run_smoke, select_smoke_units,
)


class _OldTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 7, 3, padding=1)
        self.dropout = nn.Dropout(.5)

    def forward(self, features, a_logits):
        return a_logits + self.conv(self.dropout(features).T[None]).squeeze(0).T


class _ConditionedTCN(nn.Module):
    def forward(self, features, a_logits):
        return a_logits + features[:, :1]


class HistoricalUnitSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(212)
        self.prepared = {
            "base": _OldTCN().eval().requires_grad_(False),
            "bundle": SimpleNamespace(index=SimpleNamespace(global_dim=1, local_dim=2)),
            "config": {
                "seeds": [42], "module": {"gate_dim": 8},
                "optimization": {"lr": .001, "weight_decay": 0., "grad_clip": 5.},
            },
        }
        self.units = {
            "multi": (torch.randn(3, 3), torch.randn(3, 7), torch.tensor([0, 1, 2])),
            "singleton": (torch.randn(1, 3), torch.randn(1, 7), torch.tensor([4])),
        }

    def test_selects_only_complete_historical_units(self):
        chain = {"ordered_clip_ids": ["a", "b", "c", "d"],
                 "eligible": [True, False, False]}
        standalone = {"ordered_clip_ids": ["e"], "eligible": []}
        bundle = SimpleNamespace(chains={"train": [chain, standalone]})
        segmented = select_smoke_units(bundle, "eligible_segments")
        self.assertEqual(segmented["multi"], (chain, 0, 2))
        self.assertIn(segmented["singleton"], ((chain, 2, 3), (chain, 3, 4)))
        full = select_smoke_units(bundle, "full_chain")
        self.assertEqual(full["multi"], (chain, 0, 4))
        self.assertEqual(full["singleton"], (standalone, 0, 1))

    def test_absent_singleton_fails_closed(self):
        chain = {"ordered_clip_ids": ["a", "b"], "eligible": [True]}
        with self.assertRaisesRegex(ValueError, "singleton"):
            select_smoke_units(SimpleNamespace(chains={"train": [chain]}), "full_chain")

    def test_zero_difference_unit_is_replaced_by_real_informative_unit(self):
        first = {"ordered_clip_ids": ["a", "b"], "eligible": [True]}
        second = {"ordered_clip_ids": ["c", "d", "e"], "eligible": [True, True]}
        prepared = {"base": _ConditionedTCN(), "parity": {"chain_layout": "full_chain"},
                    "bundle": SimpleNamespace(chains={"train": [first, second]})}
        initial = (torch.zeros(2, 3), torch.zeros(2, 7), torch.tensor([0, 1]))
        informative = (torch.ones(3, 3), torch.zeros(3, 7), torch.tensor([0, 1, 2]))
        with patch("experiments.selective_context.revision_smoke._read_unit",
                   side_effect=lambda bundle, unit, device:
                       informative if unit[0] is second else initial):
            selected, difference, attempts = choose_informative_multi(prepared, "cpu", initial)
        self.assertIs(selected, informative)
        self.assertGreater(difference, 0)
        self.assertGreaterEqual(attempts, 2)

    def test_each_gate_updates_three_steps_and_preserves_old_weights(self):
        for variant in ("scalar", "class_conditioned", "logits_only", "visual_logits"):
            with self.subTest(variant=variant):
                report = run_checks(self.prepared, self.units, variant, "cpu")
                self.assertEqual(report["steps"], 3)
                self.assertTrue(report["unit_matches_old_exactly"])
                self.assertTrue(report["zero_matches_A_exactly"])
                self.assertTrue(report["singleton_exercised"])
                self.assertTrue(report["old_TCN_unchanged"])
                self.assertEqual(len(report["CE_losses"]), 3)
                self.assertTrue(all(report["updated_gate_tensors"]))
                self.assertNotIn("real_cuda_tcn_revision_smoke_completed", report)

    def test_cpu_fixture_cannot_certify_formal_preflight(self):
        with self.assertRaisesRegex(RuntimeError, "real CUDA"):
            run_smoke(self.prepared, "cpu")

    def test_cli_failure_writes_false_report_and_never_loads_inputs(self):
        args = ["--feature-index", "absent", "--source-protocol-dir", "absent",
                "--legacy-config", "absent", "--legacy-checkpoint", "absent",
                "--historical-predictions", "absent", "--chain-layout", "full_chain",
                "--config", "absent", "--device", "cpu"]
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "smoke.json"
            with self.assertRaisesRegex(RuntimeError, "real CUDA"):
                main([*args, "--output", str(report)])
            stored = json.loads(report.read_text(encoding="utf-8"))
            self.assertIs(stored["all_passed"], False)
            self.assertIs(stored["real_cuda_tcn_revision_smoke_completed"], False)
            self.assertIs(stored["formal_training_started"], False)
            self.assertEqual(stored["error"]["type"], "RuntimeError")

    def test_launcher_passed_optional_historical_flags_are_accepted(self):
        parsed = parser().parse_args([
            "--feature-index", "f", "--source-protocol-dir", "s",
            "--legacy-config", "lc", "--legacy-checkpoint", "ck",
            "--historical-predictions", "hp", "--chain-layout", "eligible_segments",
            "--config", "c", "--output", "o",
            "--historical-column-map", "hm", "--historical-logits-key", "old_logits",
        ])
        self.assertEqual(parsed.historical_column_map, "hm")
        self.assertEqual(parsed.historical_logits_key, "old_logits")


if __name__ == "__main__":
    unittest.main()
