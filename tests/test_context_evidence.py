import copy
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from experiments.context_evidence.model import ContextEvidenceTCN, VARIANTS
from experiments.direction_gate.model import train_statistics
from experiments.relation.tcn_baseline import FrozenEvidenceTCN, TCNConfig
from experiments.relation.tcn_baseline import Segment
from experiments.context_evidence import run, worker
from experiments.direction_gate.worker import replacement_plan


class ContextEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(42)
        self.config = TCNConfig(4, 5, 8, 3, .2)
        self.old = FrozenEvidenceTCN(self.config).eval()
        with torch.no_grad():
            self.old.classifier.weight.normal_(0, .2)
            self.old.classifier.bias.normal_()
        self.f = torch.randn(3, 5, 16)
        self.a = self.f[:, :, -7:].clone()
        self.valid = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
        self.stats = train_statistics(self.a.reshape(-1, 7), self.a.reshape(-1, 7))

    def model(self, variant):
        m = ContextEvidenceTCN(self.config, variant, self.stats)
        if variant != "clip_only_matched":
            m.load_old(self.old.state_dict())
        return m.eval()

    def test_full_stream_matches_actual_historical_model(self):
        for v in ("legacy_continue", "legacy_aug"):
            m = self.model(v)
            torch.testing.assert_close(m(self.f, self.a, self.valid), self.old(self.f, self.a, self.valid), atol=2e-6, rtol=2e-6)

    def test_shared_mask_full_stream_keeps_the_original_dropout_schedule(self):
        m = self.model("context_delta_continue").train()
        old = copy.deepcopy(self.old).train()
        torch.manual_seed(734)
        expected = old(self.f, self.a, self.valid)
        torch.manual_seed(734)
        _, details = m(self.f, self.a, self.valid, return_details=True)
        torch.testing.assert_close(details["full"], expected, atol=3e-6, rtol=3e-6)
        self.assertFalse(m.base.classifier.bias.requires_grad)

    def test_center_stream_matches_separately_evaluated_single_clips(self):
        m = self.model("context_delta_continue")
        _, d = m(self.f, self.a, self.valid, return_details=True)
        for b in range(3):
            for t in range(int(self.valid[b].sum())):
                single = self.old(self.f[b:b+1, t:t+1], self.a[b:b+1, t:t+1], torch.ones(1, 1, dtype=torch.bool))
                torch.testing.assert_close(d["center"][b, t], single[0, 0], atol=3e-6, rtol=3e-6)

    def test_increment_equation_bias_and_exact_singleton_A(self):
        for v in VARIANTS[3:]:
            m = self.model(v)
            out, d = m(self.f, self.a, self.valid, return_details=True)
            torch.testing.assert_close(out[self.valid], (self.a + d["full"] - d["center"])[self.valid], atol=4e-6, rtol=4e-6)
            self.assertTrue(torch.equal(out[1], self.a[1]))
            self.assertTrue(torch.equal(m(self.f, self.a, self.valid, mode="zero"), self.a))
            before = out.clone()
            with torch.no_grad():
                m.base.classifier.bias.add_(100)
            self.assertTrue(torch.equal(before, m(self.f, self.a, self.valid)))

    def test_shared_dropout_prevents_fake_context_in_training_singletons(self):
        m = self.model("context_delta_gate_aug").train()
        f, a = self.f[:, :1], self.a[:, :1]
        valid = torch.ones(3, 1, dtype=torch.bool)
        for _ in range(5):
            self.assertTrue(torch.equal(m(f, a, valid), a))

    def test_gate_initialization_is_unit_and_gradients_are_real(self):
        m = self.model("context_delta_gate_aug").train()
        optimizer = torch.optim.AdamW(m.parameters(), lr=.001)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            out, d = m(self.f, self.a, self.valid, return_details=True)
            if step == 0:
                self.assertTrue(torch.equal(d["coefficient"], torch.ones_like(d["coefficient"])))
            loss = F.cross_entropy(out[self.valid], torch.zeros(int(self.valid.sum()), dtype=torch.long))
            loss.backward()
            for p in m.parameters():
                if p.grad is not None:
                    self.assertTrue(bool(torch.isfinite(p.grad).all()))
            self.assertGreater(float(m.gate_weight.grad.norm()), 0)
            self.assertGreater(float(m.base.classifier.weight.grad.norm()), 0)
            before = m.gate_weight.detach().clone()
            optimizer.step()
            self.assertGreater(float((m.gate_weight - before).abs().max()), 0)

    def test_neighbor_replacement_affects_increment_not_current_A(self):
        m = self.model("context_delta_continue")
        changed = self.f.clone()
        changed[0, 1] += 3
        before = m(self.f, self.a, self.valid)
        after = m(changed, self.a, self.valid)
        self.assertGreater(float((before[0, 0] - after[0, 0]).abs().max()), 1e-6)
        self.assertTrue(torch.equal(before[1], after[1]))

    def test_padding_and_batching_do_not_create_neighbors(self):
        m = self.model("context_delta_continue")
        batch = m(self.f, self.a, self.valid)
        for b in range(3):
            n = int(self.valid[b].sum())
            single = m(self.f[b:b+1, :n], self.a[b:b+1, :n], self.valid[b:b+1, :n])
            torch.testing.assert_close(batch[b, :n], single[0], atol=4e-6, rtol=4e-6)

    def test_clip_only_is_truly_independent_with_near_matched_parameters(self):
        c = TCNConfig(768, 768, 64, 3, .1)
        m = ContextEvidenceTCN(c, "clip_only_matched").eval()
        self.assertEqual(sum(p.numel() for p in m.parameters()), 148229)
        self.assertEqual(m.receptive_field_clips, 1)
        f = torch.randn(2, 3, 1543)
        a = f[:, :, -7:].clone()
        valid = torch.ones(2, 3, dtype=torch.bool)
        with torch.no_grad():
            m.base.classifier.weight.normal_()
        before = m(f, a, valid)
        f[0, 1] += 100
        after = m(f, a, valid)
        self.assertTrue(torch.equal(before[0, 0], after[0, 0]))
        with self.assertRaises(ValueError):
            m.load_old(FrozenEvidenceTCN(c).state_dict())

    def test_strict_old_load_and_bad_inputs_fail(self):
        m = self.model("context_delta_continue")
        state = dict(self.old.state_dict())
        state.pop("classifier.weight")
        with self.assertRaises((ValueError, RuntimeError)):
            m.load_old(state)
        with self.assertRaises(ValueError):
            m(self.f, self.a, self.valid.float())
        with self.assertRaises(ValueError):
            m(self.f, self.a, torch.zeros_like(self.valid))
        with self.assertRaises(ValueError):
            m(self.f, self.a, self.valid, mode="invalid")
        f = self.f.clone(); f[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            m(f, self.a, self.valid)

    def test_only_unchanged_anchor_is_supervised_in_replaced_view(self):
        segments, metadata = [], []
        for r in range(2):
            ids = [f"record{r}-clip{j}" for j in range(2)]
            f = np.random.default_rng(r).normal(size=(2, 16)).astype(np.float32)
            segments.append(Segment(ids, f, f[:, -7:].copy(), np.array([r, r]), ["clinical"]*2))
            metadata.extend({"clip_id": c, "split": "train", "source_kind": "clinical",
                "recording_key": str(r), "source_collection": "same"} for c in ids)
        plan = replacement_plan(segments, metadata, "train")
        batch, positions = worker.anchor_batch(segments, plan, [0], True)
        m = self.model("context_delta_continue")
        before = worker.batch_loss(m, batch, torch.device("cpu"), positions)
        f, a, y, valid, s = batch
        y = y.clone()
        y[0, 1-positions[0]] = 6
        after = worker.batch_loss(m, (f, a, y, valid, s), torch.device("cpu"), positions)
        self.assertTrue(torch.equal(before, after))
        with self.assertRaises((ValueError, RuntimeError)):
            broken = dict(plan); broken["replacements"] = [dict(plan["replacements"][0], anchor_position=1-plan["replacements"][0]["anchor_position"])]
            worker.anchor_batch(segments, broken, [0], True)

    def test_default_plan_records_all_six_serial_stages_without_GPU_or_output(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "new"
            fake = {"python": "python", "output": str(output)}
            with patch.object(run.old_run, "plan", return_value=fake), \
                 patch.object(run, "wait_released_gpu", side_effect=AssertionError("plan probes GPU")):
                p = run.plan(argparse.Namespace())
            self.assertFalse(output.exists())
            self.assertEqual(p["variants"], list(VARIANTS))
            self.assertEqual(list(p["commands"]), ["unit_tests", "audit", "smoke"] + ["train_"+v for v in VARIANTS] + ["evaluate"])
            self.assertFalse(p["training"]["early_stopping"])

    def test_model_creation_does_not_perturb_sampling_rng(self):
        # Private sampling is generated independently before any model exists.
        torch.manual_seed(432)
        initial = torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.model("context_delta_gate_aug")
        self.assertTrue(torch.equal(initial, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
