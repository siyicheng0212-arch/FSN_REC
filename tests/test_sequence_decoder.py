import unittest

import torch

from experiments.pilot_data import ClipRecord, EXPECTED_LABELS
from experiments.sequence_decoder import (
    decode_sequences,
    fit_transition_prior,
    sequence_marginal_probabilities,
)


def record(clip_id, record_id, start, label):
    return ClipRecord(
        clip_id=clip_id,
        split="train",
        label_id=label,
        normalized_label=EXPECTED_LABELS[label],
        source_collection="unit",
        group_id=record_id,
        video_path="/tmp/video.mp4",
        clip_start_sec=float(start),
        clip_end_sec=float(start + 1),
        clip_duration_sec=1.0,
        record_id=record_id,
    )


class SequenceDecoderTest(unittest.TestCase):
    def test_fit_is_sorted_by_time_and_row_normalised(self):
        rows = [
            record("clip-b", "video-a", 1, 1),
            record("clip-a", "video-a", 0, 0),
            record("clip-c", "video-a", 2, 2),
        ]
        prior = fit_transition_prior(rows, smoothing=1.0)
        self.assertAlmostEqual(float(prior.initial_probability.sum()), 1.0)
        self.assertTrue(
            torch.allclose(
                prior.transition_probability.sum(1),
                torch.ones(7, dtype=prior.transition_probability.dtype),
            )
        )
        self.assertGreater(
            float(prior.transition_probability[0, 1]),
            float(prior.transition_probability[0, 6]),
        )

    def test_viterbi_uses_transition_but_singletons_remain_visual(self):
        train = []
        for index in range(20):
            train.extend(
                [
                    record(f"clip-{index}-a", f"train-{index}", 0, 0),
                    record(f"clip-{index}-b", f"train-{index}", 1, 1),
                    record(f"clip-{index}-c", f"train-{index}", 2, 2),
                ]
            )
        prior = fit_transition_prior(train)
        validation = [
            record("clip-v-a", "video-v", 0, 0),
            record("clip-v-b", "video-v", 1, 1),
            record("clip-single", "single", 0, 6),
        ]
        logits = torch.full((3, 7), -4.0)
        logits[0, 0] = 4.0
        logits[1, 3] = 0.2
        logits[1, 1] = 0.1
        logits[2, 6] = 4.0
        decoded = decode_sequences(logits, validation, prior, weight=2.0)
        self.assertEqual(decoded.tolist(), [0, 1, 6])

    def test_validation(self):
        rows = [record("clip-a", "video", 0, 0)]
        prior = fit_transition_prior(rows)
        with self.assertRaisesRegex(ValueError, "shape"):
            decode_sequences(torch.zeros(2, 7), rows, prior)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            decode_sequences(torch.zeros(1, 7), rows, prior, weight=-1)

    def test_forward_backward_matches_brute_force_two_clip_marginals(self):
        rows = [
            record("clip-a", "video", 0, 0),
            record("clip-b", "video", 1, 1),
        ]
        prior = fit_transition_prior(self.training_rows_for_probabilities())
        logits = torch.tensor([
            [0.3, 0.1, 0.0, -0.2, -0.5, -1.0, -1.2],
            [0.1, 0.4, -0.1, -0.3, -0.4, -0.9, -1.1],
        ])
        actual = sequence_marginal_probabilities(logits, rows, prior)
        emission = torch.softmax(logits.double(), dim=1)
        joint = (
            prior.initial_probability[:, None]
            * emission[0, :, None]
            * prior.transition_probability
            * emission[1, None, :]
        )
        joint /= joint.sum()
        expected = torch.stack((joint.sum(dim=1), joint.sum(dim=0)))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-10))

    def training_rows_for_probabilities(self):
        return [
            record("clip-training-a", "training", 0, 0),
            record("clip-training-b", "training", 1, 1),
            record("clip-training-c", "training", 2, 1),
        ]

    def test_singleton_marginals_equal_visual_softmax(self):
        rows = [record("clip-only", "one", 0, 6)]
        logits = torch.tensor([[1.0, 0, 0, 0, 0, 0, 2.0]])
        prior = fit_transition_prior(self.training_rows_for_probabilities())
        actual = sequence_marginal_probabilities(logits, rows, prior)
        self.assertTrue(torch.allclose(actual, torch.softmax(logits.double(), dim=1)))


if __name__ == "__main__":
    unittest.main()
