import unittest

import torch

from experiments.pilot_data import ClipRecord, EXPECTED_LABELS
from experiments.structured_decoders import (
    decode_second_order_sequences,
    decode_segmental_sequences,
    fit_second_order_prior,
    fit_segmental_prior,
)


def record(clip, video, start, label):
    return ClipRecord(
        clip_id=clip,
        split="train",
        label_id=label,
        normalized_label=EXPECTED_LABELS[label],
        source_collection="unit",
        group_id=video,
        video_path="/tmp/video.mp4",
        clip_start_sec=float(start),
        clip_end_sec=float(start + 1),
        clip_duration_sec=1.0,
        record_id=video,
    )


class StructuredDecoderTest(unittest.TestCase):
    def training_rows(self):
        rows = []
        for index in range(20):
            labels = [0, 1, 2, 3, 3, 4, 4, 5]
            rows.extend(
                record(f"clip-{index}-{step}", f"video-{index}", step, label)
                for step, label in enumerate(labels)
            )
        return rows

    def validation_rows(self):
        labels = [0, 1, 2, 3, 3, 4, 4, 5]
        return [
            record(f"clip-v-{step}", "video-v", step, label)
            for step, label in enumerate(labels)
        ]

    def test_priors_are_normalised(self):
        second = fit_second_order_prior(self.training_rows())
        segmental = fit_segmental_prior(self.training_rows())
        self.assertTrue(torch.allclose(second.initial_probability.sum(), torch.tensor(1.0, dtype=torch.float64)))
        self.assertTrue(torch.allclose(second.first_transition_probability.sum(1), torch.ones(7, dtype=torch.float64)))
        self.assertTrue(torch.allclose(second.second_transition_probability.sum(2), torch.ones((7, 7), dtype=torch.float64)))
        self.assertTrue(torch.allclose(segmental.duration_probability.sum(1), torch.ones(7, dtype=torch.float64)))

    def test_decoders_recover_procedural_sequence(self):
        train = self.training_rows()
        validation = self.validation_rows()
        logits = torch.full((len(validation), 7), -2.0)
        target = torch.tensor([row.label_id for row in validation])
        logits[torch.arange(len(target)), target] = 0.1
        # Make the two middle action-4 clips visually prefer action 3.
        logits[5:7, 3] = 0.2
        second = decode_second_order_sequences(
            logits, validation, fit_second_order_prior(train), weight=2.0
        )
        segmental = decode_segmental_sequences(
            logits, validation, fit_segmental_prior(train), weight=2.0
        )
        self.assertEqual(second.tolist(), target.tolist())
        self.assertEqual(segmental.tolist(), target.tolist())

    def test_singletons_remain_visual(self):
        train = self.training_rows()
        singleton = [record("clip-one", "single", 0, 6)]
        logits = torch.tensor([[4.0, 0, 0, 0, 0, 0, 5.0]])
        self.assertEqual(
            decode_second_order_sequences(
                logits, singleton, fit_second_order_prior(train)
            ).item(),
            6,
        )
        self.assertEqual(
            decode_segmental_sequences(
                logits, singleton, fit_segmental_prior(train)
            ).item(),
            6,
        )


if __name__ == "__main__":
    unittest.main()
