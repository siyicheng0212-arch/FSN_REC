"""Independent scoring checks for selective sequence decoding and train-only fit."""

from itertools import product
import unittest

import numpy as np

from experiments.relation.decoder import decode, fit_transition


def exhaustive_best_score(logits, transition, effective, strength=1.0, bound=None):
    def score(path):
        total = sum(logits[t, label] for t, label in enumerate(path))
        for t, r in enumerate(effective):
            value = strength * np.log((1 - r) + logits.shape[1] * r * transition[path[t], path[t + 1]])
            if bound is not None:
                value = np.clip(value, -bound, bound)
            total += value
        return total
    paths = list(product(range(logits.shape[1]), repeat=logits.shape[0]))
    return max(map(score, paths)), score


class SelectiveDecoderTest(unittest.TestCase):
    def setUp(self):
        self.logits = np.array([[1.3, .5, -.1], [.3, .9, -.4], [.7, -.2, .5]])
        self.transition = np.array([[.05, .8, .15], [.1, .1, .8], [.7, .2, .1]])

    def test_hard_and_soft_equal_exhaustive_score_with_and_without_cuts(self):
        for mode in ("hard", "soft"):
            for reliability in ([1., 1.], [.9, 0.], [0., .95], [.4, .95], [0., 0.]):
                for eligible in ([True, True], [False, True], [True, False]):
                    for bound in (None, .3):
                        with self.subTest(mode=mode, q=reliability, eligible=eligible, bound=bound):
                            result = decode(self.logits, self.transition, reliability, eligible=eligible,
                                            mode=mode, strength=.7, potential_bound=bound)
                            expected, score = exhaustive_best_score(self.logits, self.transition,
                                                                   result["effective_reliability"], .7, bound)
                            self.assertAlmostEqual(score(result["predictions"]), expected, places=12)

    def test_random_small_sequences_match_independent_enumeration(self):
        rng = np.random.default_rng(42)
        for _ in range(30):
            logits = rng.normal(size=(3, 3))
            transition = rng.uniform(.01, 1., size=(3, 3))
            transition /= transition.sum(axis=1, keepdims=True)
            q = rng.uniform(size=2)
            for mode in ("hard", "soft"):
                result = decode(logits, transition, q, eligible=[True, True], mode=mode, threshold=.5)
                expected, score = exhaustive_best_score(logits, transition, result["effective_reliability"])
                self.assertAlmostEqual(score(result["predictions"]), expected, places=12)

    def test_zero_reliability_is_exact_original_even_with_ties_and_zero_threshold(self):
        logits = np.array([[0., 0., 0.], [1., 1., -2.], [-1., 3., 3.]])
        for mode in ("hard", "soft"):
            result = decode(logits, self.transition, [0., 0.], eligible=[True, True], mode=mode, threshold=0.)
            self.assertEqual(result["predictions"], [0, 0, 1])
            self.assertEqual(result["changed_indices"], [])
            self.assertEqual(result["effective_reliability"], [0., 0.])

    def test_no_candidate_mask_safely_falls_back(self):
        result = decode(self.logits, self.transition, [1., 1.])
        self.assertEqual(result["predictions"], self.logits.argmax(axis=1).tolist())
        self.assertEqual(result["effective_reliability"], [0., 0.])

    def test_all_off_does_not_modify_even_extreme_finite_visual_logits(self):
        huge = np.finfo(np.float64).max
        logits = np.array([[huge, -huge, 0.], [0., huge, -huge]])
        result = decode(logits, self.transition, [0.], eligible=[True])
        self.assertEqual(result["predictions"], [0, 1])

    def test_all_one_equals_transition_only_uniform_start(self):
        result = decode(self.logits, self.transition, [1., 1.], eligible=[True, True])
        paths = list(product(range(3), repeat=3))
        def old_transition_only_score(path):
            return sum(self.logits[t, label] for t, label in enumerate(path)) + sum(
                np.log(self.transition[path[t], path[t+1]]) for t in range(2))
        expected = max(paths, key=old_transition_only_score)
        self.assertEqual(result["predictions"], list(expected))

    def test_cuts_make_subchains_independent(self):
        logits = np.vstack([self.logits, np.array([[.1, -.8, .7]])])
        whole = decode(logits, self.transition, [1., 0., 1.], eligible=[True]*3)
        left = decode(logits[:2], self.transition, [1.], eligible=[True])
        right = decode(logits[2:], self.transition, [1.], eligible=[True])
        self.assertEqual(whole["predictions"], left["predictions"] + right["predictions"])

    def test_singleton_strength_zero_and_bound_zero(self):
        singleton = decode(self.logits[:1], self.transition, [])
        self.assertEqual(singleton["predictions"], [0])
        for overrides in ({"strength": 0.}, {"potential_bound": 0.}):
            result = decode(self.logits, self.transition, [1., 1.], eligible=[True, True], **overrides)
            self.assertEqual(result["predictions"], result["visual_predictions"])

    def test_soft_mode_does_not_use_hard_threshold(self):
        a = decode(self.logits, self.transition, [.4, .8], eligible=[True, True], mode="soft", threshold=0.)
        b = decode(self.logits, self.transition, [.4, .8], eligible=[True, True], mode="soft", threshold=1.)
        self.assertEqual(a, b)
        self.assertEqual(a["effective_reliability"], [.4, .8])

    def test_invalid_values_and_shapes_fail_before_decoding(self):
        invalid = [
            {"logits": [[np.nan, 0., 1.]]},
            {"logits": [1., 2., 3.]},
            {"logits": np.empty((0, 3))},
            {"transition": np.eye(3)},
            {"transition": np.ones((3, 3))},
            {"transition": [[np.inf]*3]*3},
            {"transition": np.ones((2, 2))/2},
            {"reliability": [np.nan, 0.]},
            {"reliability": [1.01, 0.]},
            {"reliability": [-.01, 0.]},
            {"reliability": [1.]},
            {"eligible": [1, 0]},
            {"eligible": [True]},
            {"threshold": -1.}, {"threshold": 1.1}, {"threshold": np.nan},
            {"mode": "automatic"}, {"strength": -1.}, {"strength": np.inf},
            {"potential_bound": -1.}, {"potential_bound": np.nan},
        ]
        for override in invalid:
            with self.subTest(override=str(override)):
                arguments = dict(logits=self.logits, transition=self.transition,
                                 reliability=[1., 1.], eligible=[True, True])
                arguments.update(override)
                with self.assertRaises(ValueError):
                    decode(**arguments)


class TrustedTransitionFitTest(unittest.TestCase):
    def test_only_c_edges_count_and_smoothing_matches_hand_calculation(self):
        edges = [
            {"split": "train", "status": "C", "left_clip_id": "a", "right_clip_id": "b"},
            {"split": "train", "status": "C", "left_clip_id": "b", "right_clip_id": "c"},
            {"split": "train", "status": "D", "left_clip_id": "c", "right_clip_id": "a"},
            {"split": "train", "status": "U", "left_clip_id": "c", "right_clip_id": "b"},
        ]
        result = fit_transition({"a": 0, "b": 1, "c": 2}, edges, num_classes=3)
        expected_counts = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        np.testing.assert_array_equal(result["counts"], expected_counts)
        np.testing.assert_allclose(result["transition"], (expected_counts+1)/np.array([[4], [4], [3]]))
        self.assertEqual(result["audit"]["accepted_C"], 2)
        self.assertEqual(result["audit"]["ignored_D"], 1)
        self.assertEqual(result["audit"]["ignored_U"], 1)

    def test_empty_trusted_set_is_uniform_not_inferred_from_source(self):
        result = fit_transition({"a": 0, "b": 1}, [], num_classes=3)
        np.testing.assert_allclose(result["transition"], np.ones((3,3))/3)

    def test_validation_rows_rejected_for_every_status(self):
        for status in ("C", "D", "U"):
            for split in ("val", "test", None):
                edge = {"split": split, "status": status, "left_clip_id": "a", "right_clip_id": "b"}
                with self.subTest(status=status, split=split), self.assertRaises(ValueError):
                    fit_transition({"a": 0, "b": 1}, [edge], num_classes=3)

    def test_bad_labels_and_malformed_c_rejected(self):
        valid = {"split": "train", "status": "C", "left_clip_id": "a", "right_clip_id": "b"}
        for labels in ({"a": 0}, {"a": 0, "b": 3}, {"a": 0, "b": True}, {"a": 0, "b": .5}):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                fit_transition(labels, [valid], num_classes=3)
        for change in ({"left_clip_id": ""}, {"right_clip_id": "a"}, {"right_clip_id": "unknown"}, {"status": "X"}, {"status": []}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                fit_transition({"a": 0, "b": 1}, [{**valid, **change}], num_classes=3)
        with self.assertRaises(ValueError):
            fit_transition({"a": 0}, [], smoothing=0.)


if __name__ == "__main__":
    unittest.main()
