import unittest
from dataclasses import asdict

import torch

from experiments.relation.tcn_baseline import (
    FrozenEvidenceTCN,
    TCNConfig,
    _parse_dilations,
    parser,
)


class LocalTCNTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_default_schedule_and_receptive_field_stay_legacy_compatible(self):
        config = TCNConfig(global_dim=3, local_dim=2, hidden_dim=8, blocks=3, dropout=0.0)
        self.assertIsNone(config.dilations)  # keeps old sealed config payload compatible
        self.assertEqual(config.dilation_schedule, (1, 2, 4))
        self.assertEqual(config.receptive_field_clips, 15)
        self.assertIsNone(asdict(config)["dilations"])

    def test_local_schedule_has_seven_clip_receptive_field(self):
        config = TCNConfig(
            global_dim=3, local_dim=2, hidden_dim=8, blocks=3,
            dropout=0.0, dilations=(1, 1, 1),
        )
        model = FrozenEvidenceTCN(config).eval()
        self.assertEqual(config.receptive_field_clips, 7)
        self.assertEqual([block.conv.dilation for block in model.blocks],
                         [(1, 1), (1, 1), (1, 1)])

    def test_middle_schedule_has_nine_clip_receptive_field(self):
        config = TCNConfig(3, 2, hidden_dim=8, blocks=3, dropout=0.0,
                           dilations=(1, 1, 2))
        self.assertEqual(config.receptive_field_clips, 9)

    def test_explicit_legacy_schedule_matches_default_model(self):
        default = TCNConfig(3, 2, hidden_dim=8, blocks=3, dropout=0.0)
        explicit = TCNConfig(3, 2, hidden_dim=8, blocks=3, dropout=0.0,
                             dilations=(1, 2, 4))
        torch.manual_seed(42)
        first = FrozenEvidenceTCN(default).eval()
        torch.manual_seed(42)
        second = FrozenEvidenceTCN(explicit).eval()
        x = torch.randn(2, 5, 12)
        a = torch.randn(2, 5, 7)
        valid = torch.tensor([[True] * 5, [True, True, True, False, False]])
        torch.testing.assert_close(first(x, a, valid), second(x, a, valid),
                                   atol=0.0, rtol=0.0)

    def test_local_model_preserves_initial_logits_and_masks_padding(self):
        torch.manual_seed(7)
        model = FrozenEvidenceTCN(
            TCNConfig(3, 2, hidden_dim=8, blocks=3, dropout=0.0,
                      dilations=(1, 1, 1))
        ).eval()
        x = torch.randn(1, 3, 12)
        a = torch.randn(1, 3, 7)
        mask = torch.tensor([[True, True, True]])
        self.assertTrue(torch.equal(model(x, a, mask), a))
        with torch.no_grad():
            model.classifier.weight.fill_(0.02)
        expected = model(x[:, :2], a[:, :2], torch.tensor([[True, True]]))
        padded = model(
            torch.cat([x[:, :2], torch.randn(1, 4, 12)], dim=1),
            torch.cat([a[:, :2], torch.randn(1, 4, 7)], dim=1),
            torch.tensor([[True, True, False, False, False, False]]),
        )
        torch.testing.assert_close(expected, padded[:, :2], atol=1e-6, rtol=1e-6)

    def test_dilation_cli_parsing_and_validation(self):
        self.assertEqual(_parse_dilations("1, 1,1"), (1, 1, 1))
        with self.assertRaises(Exception):
            _parse_dilations("1,0,1")
        args = parser().parse_args([
            "--index", "/tmp/index.jsonl",
            "--source-protocol-dir", "/tmp/source",
            "--output", "/tmp/out",
            "--dilations", "1,1,1",
        ])
        self.assertEqual(args.dilations, (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
