"""Synthetic CPU checks, not evidence of development performance."""
import copy
import math
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.relation.tcn_baseline import Segment
from experiments.direction_gate.model import AcceptanceGate, VARIANTS, confidence, train_statistics
from experiments.direction_gate.worker import (replacement_plan, replacement_segment, native_outputs,
                                               retention, smoke, train_gate, simple_metrics)
from experiments.direction_gate import run
from experiments.videomae_tcn.io import write_json
from experiments.relation.data import sha256_file


class DirectionGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        gen = torch.Generator().manual_seed(42)
        self.a = torch.randn(48, 7, generator=gen)
        self.t = self.a + torch.randn(48, 7, generator=gen)
        self.y = torch.arange(48) % 7
        self.stats = train_statistics(self.a, self.t)

    def test_parameter_counts(self):
        for v, n in zip(VARIANTS, (1, 7, 56, 62)):
            self.assertEqual(sum(p.numel() for p in AcceptanceGate(v, self.stats).parameters()), n)

    def test_all_variants_identical_initialization(self):
        expected = self.a + .95 * (self.t - self.a)
        for v in VARIANTS:
            out, alpha = AcceptanceGate(v, self.stats)(self.a, self.t)
            torch.testing.assert_close(out, expected, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(alpha, torch.full((48,), .95))

    def test_unit_zero_exact(self):
        for v in VARIANTS:
            m = AcceptanceGate(v, self.stats)
            self.assertTrue(torch.equal(m(self.a, self.t, 'unit')[0], self.t))
            self.assertTrue(torch.equal(m(self.a, self.t, 'zero')[0], self.a))

    def test_singleton_not_forced_to_A(self):
        m = AcceptanceGate('direction', self.stats)
        out, _ = m(self.a[:1], self.t[:1])
        self.assertFalse(torch.equal(out, self.a[:1]))

    def test_raw_evidence_frozen(self):
        a, t = self.a.clone().requires_grad_(), self.t.clone().requires_grad_()
        m = AcceptanceGate('direction_confidence', self.stats)
        torch.nn.functional.cross_entropy(m(a, t)[0], self.y).backward()
        self.assertIsNone(a.grad)
        self.assertIsNone(t.grad)
        self.assertGreater(float(m.confidence_weight.grad.norm()), 0)
        self.assertGreater(float(m.pair_offset.grad.norm()), 0)

    def test_direction_asymmetry(self):
        a = torch.tensor([[5., 0, 0, 0, 0, 0, 0], [0., 5, 0, 0, 0, 0, 0]])
        t = a.flip(0)
        m = AcceptanceGate('direction', train_statistics(a, t))
        with torch.no_grad():
            m.pair_offset[0, 1] = 10
            m.pair_offset[1, 0] = -10
        self.assertGreater(float(m(a, t)[1][0]), float(m(a, t)[1][1]))

    def test_unseen_pair_falls_back_to_class(self):
        stats = copy.deepcopy(self.stats)
        stats['pair_counts'] = [[0] * 7 for _ in range(7)]
        m = AcceptanceGate('direction', stats)
        with torch.no_grad():
            m.pair_offset.fill_(50)
        torch.testing.assert_close(m(self.a, self.t)[1], torch.full((48,), .95))

    def test_support_shrink_monotonic(self):
        stats = copy.deepcopy(self.stats)
        stats['pair_counts'][0][1], stats['pair_counts'][1][0] = 1, 100
        m = AcceptanceGate('direction', stats)
        self.assertLess(float(m.shrink[0, 1]), float(m.shrink[1, 0]))

    def test_confidence_shift_invariant_and_finite_extremes(self):
        torch.testing.assert_close(confidence(self.a, self.t), confidence(self.a + 50, self.t - 30), atol=4e-6, rtol=4e-6)
        a = self.a * 10000
        self.assertTrue(bool(torch.isfinite(confidence(a, -a)).all()))

    def test_confidence_initialization_does_not_use_rng(self):
        state = torch.get_rng_state().clone()
        AcceptanceGate('direction_confidence', self.stats)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_invalid_inputs_and_stats(self):
        m = AcceptanceGate('scalar', self.stats)
        for a, t in ((self.a[:, :6], self.t), (self.a * float('nan'), self.t)):
            with self.assertRaises(ValueError):
                m(a, t)
        with self.assertRaises(ValueError):
            m(self.a, self.t, 'invalid')
        stats = copy.deepcopy(self.stats); stats['q_scale'][0] = 0
        with self.assertRaises(ValueError):
            AcceptanceGate('direction', stats)

    def fixture_segments(self):
        segments, metadata = [], []
        for split in ('train', 'val'):
            for r in range(2):
                ids = [f'{split}-{r}-{j}' for j in range(2)]
                f = np.arange(20, dtype=np.float32).reshape(2, 10) + r * 100
                a = np.zeros((2, 7), np.float32) + r
                segments.append(Segment(ids, f, a, np.array([0, 1]), ['clinical'] * 2))
                metadata.extend({'clip_id': clip, 'split': split, 'source_kind': 'clinical',
                                 'recording_key': f'{split}-{r}', 'source_collection': 'lishui'} for clip in ids)
        return segments, metadata

    def test_replacement_donors_same_split_different_record(self):
        segments, metadata = self.fixture_segments()
        by_id = {r['clip_id']: r for r in metadata}
        # Normally the caller supplies only this split's segments.
        plan = replacement_plan(segments[:2], metadata, 'train')
        for row in plan['replacements']:
            a, d = by_id[row['anchor_clip_id']], by_id[row['donor_clip_id']]
            self.assertEqual(d['split'], 'train')
            self.assertNotEqual(a['recording_key'], d['recording_key'])
        self.assertEqual(len(plan['replacements']), 4)

    def test_replacement_sampling_no_labels_and_reproducible(self):
        segments, metadata = self.fixture_segments()
        first = replacement_plan(segments[:2], metadata, 'train')
        for seg in segments:
            seg.labels[:] = 6
        self.assertEqual(first, replacement_plan(segments[:2], metadata, 'train'))

    def test_replacement_paired_features_scores_and_anchor_unchanged(self):
        segments, metadata = self.fixture_segments()
        row = replacement_plan(segments[:2], metadata, 'train')['replacements'][0]
        old = segments[row['segment']]
        out = replacement_segment(segments, row)
        donor = segments[row['donor_segment']]
        j, dp, anchor = row['neighbor_position'], row['donor_position'], row['anchor_position']
        np.testing.assert_array_equal(out.features[j], donor.features[dp])
        np.testing.assert_array_equal(out.a_logits[j], donor.a_logits[dp])
        np.testing.assert_array_equal(out.features[anchor], old.features[anchor])
        np.testing.assert_array_equal(out.a_logits[anchor], old.a_logits[anchor])
        self.assertEqual(out.labels[anchor], old.labels[anchor])

    def test_no_donor_records_excluded(self):
        segments, metadata = self.fixture_segments()
        plan = replacement_plan(segments[:1], metadata, 'train')
        self.assertFalse(plan['replacements'])
        self.assertEqual(plan['excluded']['no_different_recording_donor'], 2)

    def test_smoke_real_gradient_three_steps(self):
        data = {'train': {'a': self.a, 't': self.t, 'labels': self.y}, 'statistics': self.stats}
        result = smoke(data, torch.device('cpu'))
        self.assertTrue(result['all_passed'])
        for steps in result['variants'].values():
            self.assertEqual(len(steps), 3)
            self.assertTrue(all(math.isfinite(s['loss']) for s in steps))

    def test_fixed_budget_training_and_strict_saved_state(self):
        data = {'train': {'a': self.a, 't': self.t, 'labels': self.y},
                'val': {'a': self.a, 't': self.t, 'labels': self.y}, 'statistics': self.stats,
                'orders': torch.stack([torch.arange(48), torch.arange(47, -1, -1)])}
        p = {'evidence': {'counts': {'train': 48}}, 'source_commit': 'a' * 40, 'base_sha256': {}}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            with patch('builtins.print'):
                train_gate(p, out, data, 'direction_confidence', torch.device('cpu'))
            ck = torch.load(out / 'direction_confidence/best.pt', weights_only=True)
            m = AcceptanceGate('direction_confidence', self.stats)
            m.load_state_dict(ck['state_dict'], strict=True)
            self.assertAlmostEqual(simple_metrics(data['val'], m(self.a, self.t)[0])['macro_f1'], ck['macro_f1'])
            with self.assertRaises(FileExistsError):
                train_gate(p, out, data, 'direction_confidence', torch.device('cpu'))

    def test_retention_counts(self):
        y = torch.tensor([0, 1, 2, 3])
        a = torch.eye(7)[torch.tensor([1, 1, 2, 3])]
        old = torch.eye(7)[torch.tensor([0, 2, 2, 3])]
        new = torch.eye(7)[torch.tensor([0, 1, 4, 3])]
        x = retention(y, a, old, new)
        self.assertEqual(x['retained_fixes'], 1)
        self.assertEqual(x['prevented_old_harm'], 1)
        self.assertEqual(x['new_harm_where_old_TCN_was_correct'], 1)

    def test_default_readonly_and_reject_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'existing'; out.mkdir()
            with self.assertRaises(FileExistsError):
                run.fresh_output(out, Path(d) / 'i/index', Path(d) / 'source')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); base = root / 'base'; base.mkdir()
            (base / 'tcn').mkdir(); (base / 'evaluation').mkdir()
            (base / 'tcn/best.pt').write_bytes(b'plan-only-fixture')
            (base / 'evaluation/private_val_predictions.jsonl').write_text('')
            info = {'counts': {'train': 4, 'val': 3}}
            write_json(base / 'run_protocol.json', {'evidence': info})
            write_json(base / 'tcn/result.json', {'training_completed': True, 'test_read': False,
                       'best_checkpoint_sha256': sha256_file(base / 'tcn/best.pt')})
            write_json(base / 'suite_result.json', {'status': 'complete', 'training_completed': True,
                       'exit_codes': {s: 0 for s in ('unit_tests', 'smoke', 'train', 'evaluate')}})
            write_json(base / 'evaluation/result.json', {'status': 'complete', 'val_clip_count': 3})
            args = argparse.Namespace(index=str(root / 'features/index.jsonl'), source_protocol_dir=str(root / 'source'),
                   output=str(root / 'new'), base_run=str(base), expected_commit='a' * 40,
                   gpu=0, gpu_uuid='GPU-12345678-abcd', python=__import__('sys').executable)
            with patch.object(run, 'prepare', return_value=(None, None, info)), \
                 patch.object(run, 'repository_state', return_value={'commit': 'a' * 40, 'clean': True}), \
                 patch.object(run, 'check_gpu', side_effect=AssertionError('plan probes GPU')):
                result = run.plan(args)
            self.assertTrue(result['plan_only_does_not_probe_gpu'])
            self.assertFalse((root / 'new').exists())


if __name__ == '__main__':
    unittest.main()
