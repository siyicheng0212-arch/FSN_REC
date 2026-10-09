import copy
import unittest

import numpy as np
import torch

from experiments.relation.tcn_baseline import (
    FrozenEvidenceTCN, TCNConfig, _forward_loss, build_segments, collate_segments,
)


class MemoryIndex:
    def __init__(self, metadata):
        self.clips = {row["clip_id"]: copy.deepcopy(row) for row in metadata}

    def read(self, clip_id):
        number = {"train_a": 1, "train_b": 2, "train_c": 3, "val_d": 4}[clip_id]
        return {"global_tokens": np.full((2, 3), number, dtype=np.float32),
                "local_tokens": np.full((3, 2), number * .1, dtype=np.float32),
                "global_positions": np.array([0., 1.], dtype=np.float32),
                "local_positions": np.array([0., .5, 1.], dtype=np.float32),
                "logits": np.asarray([number, 0., 0., 0., 0., 0., 0.], dtype=np.float32)}


def fixture():
    rows = []
    for clip, split, group, label in (
        ("train_a", "train", "group_train", 1),
        ("train_b", "train", "group_train", 2),
        ("train_c", "train", "group_train", 3),
        ("val_d", "val", "group_val", 4),
    ):
        rows.append({"clip_id": clip, "split": split, "group_id": group,
                     "label_id": label, "source_kind": "clinical",
                     "source_video_key": group, "timebase_key": "one_clock",
                     "recording_key": group})
    chains = [{"chain_id": "train_chain", "split": "train",
               "ordered_clip_ids": ["train_a", "train_b", "train_c"],
               "eligible": [True, False]}]
    edges = [{"edge_id": "edge_ab", "split": "train", "status": "C",
              "label_origin": "source_rule_v1", "left_clip_id": "train_a",
              "right_clip_id": "train_b"}]
    return MemoryIndex(rows), rows, chains, edges


class TCNBaselineTest(unittest.TestCase):
    def test_initially_identical_to_a_and_padding_is_masked(self):
        torch.manual_seed(42)
        model = FrozenEvidenceTCN(TCNConfig(3, 2, hidden_dim=8, blocks=2, dropout=0)).eval()
        one = torch.randn(1, 2, 12)
        a = torch.randn(1, 2, 7)
        mask = torch.ones(1, 2, dtype=torch.bool)
        self.assertTrue(torch.equal(model(one, a, mask), a))
        with torch.no_grad():
            model.classifier.weight.fill_(.02)
        alone = model(one, a, mask)
        padded_x = torch.cat((one, torch.randn(1, 3, 12)), dim=1)
        padded_a = torch.cat((a, torch.randn(1, 3, 7)), dim=1)
        padded_mask = torch.tensor([[True, True, False, False, False]])
        together = model(padded_x, padded_a, padded_mask)
        torch.testing.assert_close(alone, together[:, :2], atol=1e-6, rtol=1e-6)

    def test_source_cut_partitions_train_and_keeps_val_separate(self):
        index, rows, chains, edges = fixture()
        train = build_segments(index, rows, chains, edges, "train")
        self.assertEqual([segment.clip_ids for segment in train],
                         [["train_a", "train_b"], ["train_c"]])
        batch = collate_segments(train)
        self.assertEqual(batch[0].shape, (2, 2, 12))
        self.assertEqual(batch[3].tolist(), [[True, True], [True, False]])
        self.assertEqual(batch[2].tolist(), [[1, 2], [3, -100]])
        val_chain = [{"chain_id": "val_chain", "split": "val",
                      "ordered_clip_ids": ["val_d"], "eligible": []}]
        val = build_segments(index, rows, val_chain, edges, "val")
        self.assertEqual([s.clip_ids for s in val], [["val_d"]])

    def test_real_train_step_updates_a_correction(self):
        index, rows, chains, edges = fixture()
        batch = collate_segments(build_segments(index, rows, chains, edges, "train"))
        model = FrozenEvidenceTCN(TCNConfig(3, 2, hidden_dim=8, blocks=2, dropout=0))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.05)
        original = model(batch[0], batch[1], batch[3]).detach().clone()
        loss, _, valid = _forward_loss(model, batch, torch.device("cpu"))
        self.assertEqual(int(valid.sum()), 3)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertGreater(float(model.classifier.weight.grad.norm()), 0.)
        optimizer.step()
        updated = model(batch[0], batch[1], batch[3]).detach()
        self.assertGreater(float((updated - original).abs().max()), 0.)

    def test_sealed_adjacency_and_coverage_are_required(self):
        index, rows, chains, edges = fixture()
        bad = copy.deepcopy(chains)
        bad[0]["eligible"] = [True, True]
        with self.assertRaises(ValueError):
            build_segments(index, rows, bad, edges, "train")
        with self.assertRaises(ValueError):
            build_segments(index, rows, chains, [], "train")
        bad = copy.deepcopy(chains)
        bad[0]["ordered_clip_ids"] = ["train_a", "train_b", "val_d"]
        with self.assertRaises(ValueError):
            build_segments(index, rows, bad, edges, "train")


if __name__ == "__main__":
    unittest.main()
