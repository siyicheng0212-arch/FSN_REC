"""Meaningful preflight failures and immutable snapshot behavior; no videos needed."""

import contextlib
import hashlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cvm.protocol import ProtocolError, audit_protocol, load_protocol, main
from experiments.pilot_data import EXPECTED_LABELS


class FrozenProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rows = {}
        self.paths = {}
        for split in ("train", "val", "test"):
            self.rows[split] = [
                {
                    "schema_version": "fsn-pilot-manifest-1.0",
                    "clip_id": f"clip-private-{split}-{label}",
                    "split": split,
                    "label_id": label,
                    "normalized_label": name,
                    "source_collection": "private-source",
                    "group_id": f"private-group-{split}-{label}",
                    "video_path": f"/private/videos/{split}-{label}.mp4",
                    "clip_start_sec": 0.0,
                    "clip_end_sec": 1.0,
                    "clip_duration_sec": 1.0,
                }
                for label, name in EXPECTED_LABELS.items()
            ]
            self.paths[split] = self.root / f"{split}.jsonl"
        self.write_inputs()

    def write_inputs(self):
        for split, rows in self.rows.items():
            self.write_rows(self.paths[split], rows)

    @staticmethod
    def write_rows(path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def audit(self, *, include_test=False, prior=(), output=None, video_root=None):
        self.write_inputs()
        return audit_protocol(
            self.paths["train"], self.paths["val"], output or self.root / "frozen",
            test=self.paths["test"] if include_test else None,
            prior_eval_manifests=prior, video_root=video_root,
        )

    def assert_rejected(self, message, *, include_test=False, prior=()):
        with self.assertRaisesRegex(ProtocolError, message):
            self.audit(include_test=include_test, prior=prior)
        self.assertFalse((self.root / "frozen").exists())

    def test_success_public_summary_is_aggregate_and_private_snapshot_is_frozen(self):
        path = self.audit()
        public = path.read_text(encoding="utf-8")
        for private_value in ("private-group", "private-source", "/private/videos", "clip-private"):
            self.assertNotIn(private_value, public)
        protocol = load_protocol(path)
        self.assertEqual(protocol.summary["audit"]["split_counts"], {"train": 7, "val": 7})
        self.assertFalse(protocol.summary["audit"]["recording_group_is_patient_identity"])
        self.assertFalse(protocol.summary["audit"]["identity_independence"]["patient"]["independent_claim_ready"])
        with self.assertRaises(TypeError):
            protocol.summary["audit"]["split_counts"]["train"] = 999
        # Mutation of an original input cannot silently replace the frozen snapshot.
        self.paths["train"].write_text("invalid input after freeze", encoding="utf-8")
        self.assertEqual(len(load_protocol(path).records["train"]), 7)

    def test_duplicate_clip_rejected_even_with_different_label(self):
        self.rows["val"][0]["clip_id"] = self.rows["train"][1]["clip_id"]
        self.assert_rejected("duplicate clip_id")

    def test_recording_group_overlap_rejected(self):
        self.rows["val"][0]["group_id"] = self.rows["train"][0]["group_id"]
        self.assert_rejected("recording group overlap")

    def test_source_path_overlap_rejected_with_distinct_clip_and_group(self):
        self.rows["val"][0]["video_path"] = "/private/videos/unused/../train-0.mp4"
        self.assert_rejected("source video or recording overlap")

    def test_source_hash_alias_overlap_rejected(self):
        digest = hashlib.sha256(b"same source bytes").hexdigest()
        self.rows["train"][0]["video_sha256"] = digest
        self.rows["val"][0]["video_metadata"] = {"sha256": digest}
        self.assert_rejected("source video or recording overlap")

    def test_source_id_overlap_rejected_despite_different_local_paths(self):
        self.rows["train"][0]["source_video_id"] = "private-online-source"
        self.rows["val"][0]["source_video_id"] = "private-online-source"
        self.assert_rejected("source video or recording overlap")

    def test_verified_patient_overlap_rejected(self):
        for split in ("train", "val"):
            self.rows[split][0]["identity_metadata"] = {"patient_id": "private-patient", "status": "verified"}
        self.assert_rejected("verified patient identity overlap")

    def test_verified_practitioner_overlap_rejected(self):
        for split in ("train", "val"):
            self.rows[split][0].update(practitioner_id="private-surgeon", practitioner_id_verified=True)
        self.assert_rejected("verified practitioner identity overlap")

    def test_inferred_filename_group_does_not_certify_patient_independence(self):
        for members in self.rows.values():
            for row in members:
                row.update(grouping_basis="filename_patient_code", grouping_confidence="high")
        protocol = load_protocol(self.audit())
        self.assertFalse(protocol.summary["audit"]["identity_independence"]["patient"]["complete_verified_coverage"])

    def test_complete_verified_identity_coverage_is_recorded_without_public_ids(self):
        for split, members in self.rows.items():
            for index, row in enumerate(members):
                row["identity_metadata"] = {"patient_id": f"private-person-{split}-{index}", "status": "verified"}
        path = self.audit()
        self.assertTrue(load_protocol(path).summary["audit"]["identity_independence"]["patient"]["independent_claim_ready"])
        self.assertNotIn("private-person", path.read_text(encoding="utf-8"))

    def test_same_split_multiple_clips_per_recording_allowed_if_intervals_do_not_overlap(self):
        self.rows["train"][1].update(group_id=self.rows["train"][0]["group_id"],
                                      video_path=self.rows["train"][0]["video_path"],
                                      clip_start_sec=1.0, clip_end_sec=2.0)
        protocol = load_protocol(self.audit())
        self.assertEqual(protocol.summary["audit"]["group_counts"]["train"], 6)

    def test_overlapping_conflicting_labels_rejected(self):
        self.rows["train"][1]["video_path"] = self.rows["train"][0]["video_path"]
        self.assert_rejected("overlapping source intervals have conflicting labels")

    def test_same_interval_cannot_be_duplicated_by_new_clip_id(self):
        duplicate = dict(self.rows["train"][0], clip_id="clip-other-copy")
        self.rows["train"].append(duplicate)
        self.assert_rejected("duplicate source interval")

    def test_seven_classes_required_in_every_supplied_split(self):
        self.rows["test"].pop()
        self.assert_rejected("all seven classes are required in test", include_test=True)

    def test_invalid_labels_and_durations_rejected(self):
        cases = [
            ({"normalized_label": "wrong"}, "conflicting seven-class label"),
            ({"label_id": True}, "invalid seven-class"),
            ({"clip_duration_sec": 2.0}, "does not match"),
            ({"clip_end_sec": float("nan")}, "finite number"),
            ({"clip_start_sec": -1.0}, "invalid clip duration"),
            ({"source_video_duration_sec": 0.5}, "exceeds declared"),
        ]
        original = dict(self.rows["train"][0])
        for updates, message in cases:
            with self.subTest(message=message):
                self.rows["train"][0] = {**original, **updates}
                self.assert_rejected(message)

    def test_missing_group_cannot_fall_back_to_clip_id(self):
        self.rows["train"][0].pop("group_id")
        self.assert_rejected("missing group_id")

    def test_cli_does_not_reassign_manifest_split(self):
        self.rows["val"][0]["split"] = "train"
        self.assert_rejected("manifest split does not match")

    def test_relative_paths_require_explicit_video_root(self):
        self.rows["train"][0]["video_path"] = "videos/train-0.mp4"
        self.assert_rejected("relative video_path requires")
        protocol = load_protocol(self.audit(video_root=self.root))
        self.assertEqual(protocol.records["train"][0].video_path, str(self.root / "videos/train-0.mp4"))

    def test_explicit_cache_split_routes_cache_without_changing_evaluation_role(self):
        self.rows["val"][0]["cache_split"] = "train"
        protocol = load_protocol(self.audit())
        record = protocol.records["val"][0]
        self.assertEqual(record.split, "val")
        self.assertEqual(protocol.cache_record(record).split, "train")

    def test_test_without_history_is_not_verified_clean(self):
        protocol = load_protocol(self.audit(include_test=True))
        self.assertEqual(protocol.summary["test_history_status"], "history_not_verified")

    def test_test_prior_clip_group_source_or_identity_exposure_rejected(self):
        row = self.rows["test"][0]
        self.rows["test"][0].update(patient_id="private-test-person", patient_id_verified=True)
        cases = [
            {"clip_id": row["clip_id"]},
            {"group_id": row["group_id"]},
            {"video_path": row["video_path"]},
            {"patient_id": "private-test-person"},
        ]
        prior = self.root / "old-evaluation.jsonl"
        for exposure in cases:
            with self.subTest(exposure=list(exposure)):
                self.write_rows(prior, [exposure])
                self.assert_rejected("test has prior evaluation exposure", include_test=True, prior=[prior])

    def test_repartition_of_old_validation_to_test_rejected_without_history_file(self):
        self.rows["test"][0]["original_split"] = "val"
        self.assert_rejected("test has prior evaluation exposure", include_test=True)

    def test_prior_development_exposure_recorded_and_clean_test_has_scoped_history(self):
        prior = self.root / "old-evaluation.jsonl"
        self.write_rows(prior, [self.rows["val"][0]])
        path = self.audit(include_test=True, prior=[prior])
        protocol = load_protocol(path)
        self.assertEqual(protocol.summary["test_history_status"], "verified_clean")
        self.assertEqual(protocol.summary["audit"]["prior_exposure_record_counts"]["val"], 1)
        self.assertIn("supplied", protocol.summary["test_history_scope"])

    def test_partial_prior_identifier_list_cannot_certify_clean_test(self):
        prior = self.root / "old-evaluation.jsonl"
        self.write_rows(prior, [{"clip_id": "clip-unrelated"}])
        protocol = load_protocol(self.audit(include_test=True, prior=[prior]))
        self.assertEqual(protocol.summary["test_history_status"], "history_not_verified")

    def test_tampered_snapshot_and_tampered_audit_are_rejected(self):
        path = self.audit()
        manifest = path.parent / "private" / "train.jsonl"
        original = manifest.read_bytes()
        manifest.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ProtocolError, "SHA-256 changed"):
            load_protocol(path)
        manifest.write_bytes(original)
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["audit"]["split_counts"]["train"] = 999
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ProtocolError, "summary does not match"):
            load_protocol(path)

    def test_tampered_implementation_or_exposure_snapshot_is_rejected(self):
        prior = self.root / "old-evaluation.jsonl"
        self.write_rows(prior, [{"clip_id": "clip-unrelated"}])
        path = self.audit(prior=[prior])
        summary = json.loads(path.read_text(encoding="utf-8"))
        original_implementation_sha = summary["implementation_sha256"]
        summary["implementation_sha256"] = "0" * 64
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ProtocolError, "implementation changed"):
            load_protocol(path)
        summary["implementation_sha256"] = original_implementation_sha
        path.write_text(json.dumps(summary), encoding="utf-8")
        exposure = path.parent / "private" / "prior-eval-000.jsonl"
        exposure.write_text('{"clip_id":"clip-another"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ProtocolError, "exposure SHA-256 changed"):
            load_protocol(path)

    def test_no_overwrite_and_safe_cli_output(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = main(["audit", "--train", str(self.paths["train"]), "--val", str(self.paths["val"]),
                           "--output", str(self.root / "frozen")])
        self.assertEqual(status, 0)
        self.assertNotIn("private-group", stdout.getvalue())
        self.assertNotIn("/private/videos", stdout.getvalue())
        path = self.root / "frozen" / "protocol.json"
        original = path.read_bytes()
        with self.assertRaisesRegex(ProtocolError, "overwrites are forbidden"):
            self.audit()
        self.assertEqual(path.read_bytes(), original)
        self.rows["train"][0]["video_path"] = self.rows["val"][0]["video_path"]
        self.write_inputs()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = main(["audit", "--train", str(self.paths["train"]), "--val", str(self.paths["val"]),
                           "--output", str(self.root / "other")])
        self.assertEqual(status, 2)
        self.assertNotIn("/private/videos", stderr.getvalue())
        self.assertNotIn("clip-private", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
