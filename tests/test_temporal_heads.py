import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from experiments.temporal_heads import run, worker
from experiments.temporal_heads.model import CONFIG, VARIANTS, TemporalHead, objective
from experiments.temporal_heads.vendor.asformer import AttLayer, MyTransformer
from experiments.temporal_heads.vendor.mstcn2 import MS_TCN2

VENDOR = Path(__file__).parents[1] / "experiments/temporal_heads/vendor"


def upstream(name):
    text = (VENDOR / ("upstream_" + name + "_model.py.txt")).read_text().split("\nclass Trainer:")[0]
    text = text.replace("from loguru import logger\n", "").replace("from eval import segment_bars_with_confidence\n", "")
    text = text.replace("MS_TCB    def __init__", "    def __init__")
    text = text.replace('device = torch.device("cuda" if torch.cuda.is_available() else "cpu")', 'device = torch.device("cpu")')
    text = text.replace("window_mask[:, :, i:i+self.bl] = 1", "window_mask[:, i, i:i+self.bl] = 1")
    module = types.ModuleType(name)
    exec(compile(text, "pinned_upstream_" + name, "exec"), module.__dict__)
    return module


class TemporalHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(42)
        self.f = torch.randn(3, 5, 1543)
        self.a = self.f[..., -7:].clone()
        self.y = torch.randint(7, (3, 5))
        self.valid = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)

    def test_mstcnpp_is_exact_pinned_upstream_architecture(self):
        original = upstream("mstcn2").MS_TCN2(3, 3, 3, 8, 12, 7).eval()
        current = MS_TCN2(3, 3, 3, 8, 12, 7).eval()
        current.load_state_dict(original.state_dict(), strict=True)
        x = torch.randn(2, 12, 7)
        torch.testing.assert_close(current(x), original(x), atol=0, rtol=0)
        self.assertEqual([c.dilation[0] for c in current.PG.conv_dilated_1], [4, 2, 1])
        self.assertEqual([c.dilation[0] for c in current.PG.conv_dilated_2], [1, 2, 4])

    def test_corrected_asformer_matches_upstream_nonsingleton(self):
        original = upstream("asformer").MyTransformer(3, 3, 2, 2, 8, 12, 7, .3).eval()
        current = MyTransformer(3, 3, 2, 2, 8, 12, 7, .3).eval()
        result = current.load_state_dict(original.state_dict(), strict=False)
        self.assertTrue(all(k.endswith("window_mask") for k in result.missing_keys))
        self.assertFalse(result.unexpected_keys)
        x, mask = torch.randn(1, 12, 7), torch.ones(1, 1, 7)
        torch.testing.assert_close(current(x, mask), original(x, mask), atol=3e-6, rtol=3e-6)

    def test_large_window_fast_path_matches_native_padding_softmax(self):
        for stage in ("encoder", "decoder"):
            for length in (1, 3, 7):
                layer = AttLayer(8, 8, 8, 2, 2, 2, 32, stage, "sliding_att").eval()
                slow = copy.deepcopy(layer); slow.fast_short_windows = False
                x, f, mask = torch.randn(1, 8, length), torch.randn(1, 8, length), torch.ones(1, 1, length)
                torch.testing.assert_close(layer(x, f, mask), slow(x, f, mask), atol=3e-6, rtol=3e-6)

    def test_asformer_batch_windows_do_not_mix_independent_sequences(self):
        for length in (3, 7, 25):
            model = MyTransformer(3, 4, 2, 2, 8, 12, 7, .3).eval()
            x = torch.randn(3, 12, length); mask = torch.ones(3, 1, length)
            together = model(x, mask)
            for b in range(3):
                torch.testing.assert_close(together[:, b:b+1], model(x[b:b+1], mask[b:b+1]), atol=5e-6, rtol=5e-6)

    def test_all_heads_singletons_finite_and_padding_independent(self):
        for v in VARIANTS:
            m = TemporalHead(v, 1543).eval()
            out = m(self.f, self.a, self.valid)
            self.assertEqual(out.shape, self.a.shape)
            for b in range(3):
                n = int(self.valid[b].sum())
                single = m(self.f[b:b+1, :n], self.a[b:b+1, :n], self.valid[b:b+1, :n])
                torch.testing.assert_close(out[b, :n], single[0], atol=4e-6, rtol=4e-6)
            if v == "plain_tcn_fresh":
                self.assertTrue(torch.equal(out, self.a))
            else:
                self.assertFalse(torch.equal(out[1, 0], self.a[1, 0]))
            changed = self.f.clone(); changed[1, 1:] += 1000
            torch.testing.assert_close(out, m(changed, self.a, self.valid), atol=0, rtol=0)

    def test_head_gradients_and_three_actual_updates_A_not_trainable(self):
        for v in VARIANTS:
            m = TemporalHead(v, 1543).train()
            optimizer = torch.optim.AdamW(m.parameters(), lr=.001)
            for _ in range(3):
                before = {n: w.detach().clone() for n, w in m.named_parameters()}
                optimizer.zero_grad(set_to_none=True)
                stages = m(self.f, self.a, self.valid, all_stages=True)
                loss, ce, smooth = objective(stages, self.y, self.valid)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                norms = worker.gradient_norms(m)
                self.assertGreater(max(norms.values()), 0)
                optimizer.step()
                self.assertTrue(any(not torch.equal(w, before[n]) for n, w in m.named_parameters()))
            self.assertIsNone(self.f.grad); self.assertIsNone(self.a.grad)

    def test_multistage_loss_has_all_stage_supervision(self):
        pred = torch.randn(4, 3, 5, 7, requires_grad=True)
        loss, ce, _ = objective(pred, self.y, self.valid)
        expected = sum(F.cross_entropy(s[self.valid], self.y[self.valid]) for s in pred)
        torch.testing.assert_close(ce, expected)
        loss.backward()
        for gradient in pred.grad:
            self.assertGreater(float(gradient.abs().sum()), 0)

    def test_singleton_smoothness_is_finite_zero(self):
        pred = torch.randn(4, 3, 1, 7, requires_grad=True)
        valid = torch.ones(3, 1, dtype=torch.bool)
        loss, ce, smooth = objective(pred, self.y[:, :1], valid)
        self.assertTrue(torch.equal(smooth, torch.tensor(0.)))
        self.assertTrue(torch.equal(loss, ce)); loss.backward()
        self.assertTrue(bool(torch.isfinite(pred.grad).all()))

    def test_loss_does_not_supervise_padding_or_fake_cross_segment_pairs(self):
        pred = torch.randn(4, 3, 5, 7)
        initial = objective(pred, self.y, self.valid)[0]
        altered = pred.clone(); altered[:, ~self.valid] += 900
        changed_labels = self.y.clone(); changed_labels[~self.valid] = -100
        torch.testing.assert_close(initial, objective(altered, changed_labels, self.valid)[0], atol=0, rtol=0)

    def test_invalid_shapes_masks_and_nan_fail(self):
        m = TemporalHead("mstcnpp", 1543)
        for mask in (self.valid.float(), torch.zeros_like(self.valid), torch.tensor([[1, 0, 1, 0, 0]]*3, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                m(self.f, self.a, mask)
        broken = self.f.clone(); broken[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            m(broken, self.a, self.valid)
        with self.assertRaises(ValueError):
            TemporalHead("mstcnpp", 768)

    def test_checkpoint_strict_loading_fails_on_missing_shared_weight(self):
        m = TemporalHead("asformer", 1543)
        state = dict(m.state_dict()); state.pop(next(iter(state)))
        with self.assertRaises(RuntimeError):
            m.load_state_dict(state, strict=True)

    def test_initialization_does_not_change_sampling_random_state(self):
        before = torch.get_rng_state().clone()
        for v in VARIANTS:
            worker.make_model(v, torch.device("cpu"))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_default_plan_does_not_probe_gpu_or_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prior = root / "old"; prior.mkdir()
            for name, value in {"suite_result.json": {"status": "complete", "training_completed": True, "exit_codes": {"evaluate": 0}},
                                "run_protocol.json": {"evidence": {}, "base_sha256": {}},
                                "evaluation/public_safe_summary.json": {"training_completed": True, "test_read": False}}.items():
                f = prior / name; f.parent.mkdir(exist_ok=True); f.write_text(json.dumps(value))
            fake = {"python": sys.executable, "output": str(root / "new"), "evidence": {}, "base_sha256": {}}
            with patch.object(run.old_run, "plan", return_value=fake), patch.object(run, "wait_released_gpu", side_effect=AssertionError("GPU touched")):
                p = run.plan(argparse.Namespace(prior_context_run=str(prior)))
            self.assertFalse((root / "new").exists())
            self.assertEqual(p["variants"], list(VARIANTS))
            self.assertEqual(list(p["commands"]), ["unit_tests", "audit", "smoke"] + ["train_"+v for v in VARIANTS] + ["evaluate"])

    def test_failure_stops_later_queue_preserving_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fresh"
            p = {"output": str(output), "index": tmp, "source_protocol_dir": tmp, "source_commit": "0"*40,
                 "gpu_physical_index": 0, "gpu_uuid": "GPU-example",
                 "commands": {"unit_tests": [sys.executable, "-c", "raise SystemExit(2)"], "train_mstcnpp": [sys.executable, "-c", "raise AssertionError('must not run')"]}}
            with patch.object(run, "unchanged"), patch.object(run, "wait_released_gpu"), patch.object(run, "fresh_output"), patch.dict("os.environ", {}, clear=True):
                with self.assertRaises(RuntimeError):
                    run.execute(p)
            self.assertTrue((output / ".launch_once").exists())
            state = json.loads((output / "status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["exit_codes"], {"unit_tests": 2})
            self.assertFalse((output / "launcher_logs/train_mstcnpp.log").exists())


if __name__ == "__main__":
    unittest.main()
