"""Source domains are withheld without fabricating a new unseen test set."""

import contextlib
import hashlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cvm.protocol import ProtocolError, audit_protocol, load_protocol
from cvm.source_holdout import (
    PREDECLARED_SELECTION,
    SourceHoldoutDataInsufficient,
    derive_source_holdout,
    load_source_holdout_protocol,
    main,
    source_alias,
)
from experiments.pilot_data import EXPECTED_LABELS


class SourceHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.domains = ("private-acquisition-domain-A", "private-acquisition-domain-B")
        self.rows = {}
        self.paths = {}
        for split in ("train", "val", "test"):
            self.rows[split] = [
                {"clip_id": f"clip-private-{split}-{domain}-{label}", "split": split,
                 "label_id": label, "normalized_label": name,
                 "source_collection": source, "group_id": f"private-record-{split}-{domain}-{label}",
                 "video_path": f"/private/videos/{split}-{domain}-{label}.mp4",
                 "clip_start_sec": 0., "clip_end_sec": 1., "clip_duration_sec": 1.}
                for domain, source in enumerate(self.domains) for label, name in EXPECTED_LABELS.items()
            ]
            self.paths[split] = self.root / f"{split}.jsonl"
        self.prior = self.root / "prior-eval.jsonl"
        self.output = self.root / "source-derived"

    @staticmethod
    def write_rows(path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def parent(self, *, with_test=True, with_history=True):
        for split, rows in self.rows.items():
            self.write_rows(self.paths[split], rows)
        self.write_rows(self.prior, [self.rows["val"][-1]])
        return audit_protocol(self.paths["train"], self.paths["val"], self.root / "initial",
                              test=self.paths["test"] if with_test else None,
                              prior_eval_manifests=[self.prior] if with_history else [])

    def derive(self, parent, *, sources=None, exposure=PREDECLARED_SELECTION):
        return derive_source_holdout(parent, sources or [self.domains[0]], self.output,
                                      source_selection_exposure=exposure)

    def test_exact_source_selection_keeps_only_initial_test_members(self):
        initial = self.parent()
        path = self.derive(initial)
        protocol = load_source_holdout_protocol(path)
        for split in ("train", "val"):
            self.assertEqual({row.source_collection for row in protocol.records[split]}, {self.domains[1]})
        self.assertEqual({row.source_collection for row in protocol.records["test"]}, {self.domains[0]})
        original_test = {row["clip_id"] for row in self.rows["test"]}
        self.assertTrue({row.clip_id for row in protocol.records["test"]} <= original_test)
        self.assertEqual(protocol.summary["test_history_status"], "verified_clean")
        self.assertEqual(protocol.summary["source_holdout"]["parent_protocol_sha256"], load_protocol(initial).protocol_sha256)
        self.assertFalse(protocol.summary["audit"]["identity_independence"]["patient"]["independent_claim_ready"])
        self.assertEqual(dict(protocol.summary["audit"]["split_counts"]), {"train": 7, "val": 7, "test": 7})

    def test_raw_source_ids_paths_and_clip_ids_remain_private(self):
        path = self.derive(self.parent())
        public = path.read_text(encoding="utf-8")
        for private in (*self.domains, "clip-private", "private-record", "/private/videos", str(self.root)):
            self.assertNotIn(private, public)
        plan = json.loads((path.parent / "private" / "source_holdout_plan.json").read_text())
        self.assertEqual(plan["heldout_sources"], [self.domains[0]])
        self.assertEqual((path.parent / "private" / "source_holdout_plan.json").stat().st_mode & 0o777, 0o600)

    def test_no_initial_test_cannot_promote_train_or_val(self):
        with self.assertRaisesRegex(ProtocolError, "initial verified_clean test"):
            self.derive(self.parent(with_test=False))
        self.assertFalse(self.output.exists())

    def test_unknown_history_cannot_become_verified_clean_by_derivation(self):
        with self.assertRaisesRegex(ProtocolError, "initial verified_clean test"):
            self.derive(self.parent(with_history=False))
        self.assertFalse(self.output.exists())

    def test_source_selection_requires_explicit_pre_score_attestation(self):
        initial = self.parent()
        with self.assertRaisesRegex(ProtocolError, "predeclared before source scores"):
            self.derive(initial, exposure="chosen_after_scores")
        self.assertFalse(self.output.exists())

    def test_missing_requested_test_source_is_data_insufficient_not_auto_changed(self):
        initial = self.parent()
        with self.assertRaises(SourceHoldoutDataInsufficient) as caught:
            self.derive(initial, sources=["private-source-not-in-test"])
        self.assertEqual(caught.exception.aggregate["status"], "data_insufficient")
        self.assertEqual(caught.exception.aggregate["missing_initial_test_source_aliases"], [source_alias("private-source-not-in-test")])
        self.assertFalse(self.output.exists())

    def test_missing_class_after_exclusion_is_data_insufficient(self):
        self.rows["train"] = [row for row in self.rows["train"]
                              if not (row["source_collection"] == self.domains[1] and row["label_id"] == 0)]
        initial = self.parent()
        with self.assertRaises(SourceHoldoutDataInsufficient) as caught:
            self.derive(initial)
        self.assertEqual(caught.exception.aggregate["missing_label_ids"]["train"], [0])
        self.assertFalse(self.output.exists())

    def test_selecting_all_sources_cannot_leave_empty_train_val(self):
        initial = self.parent()
        with self.assertRaises(SourceHoldoutDataInsufficient):
            self.derive(initial, sources=self.domains)
        self.assertFalse(self.output.exists())

    def test_parent_history_is_carried_byte_for_byte(self):
        initial = self.parent()
        derived = self.derive(initial)
        before = (initial.parent / "private" / "prior-eval-000.jsonl").read_bytes()
        after = (derived.parent / "private" / "prior-eval-000.jsonl").read_bytes()
        self.assertEqual(before, after)

    def test_changed_private_selection_plan_rejected(self):
        path = self.derive(self.parent())
        plan = path.parent / "private" / "source_holdout_plan.json"
        plan.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ProtocolError, "private plan SHA changed"):
            load_source_holdout_protocol(path)

    def test_changed_parent_protocol_rejected(self):
        initial = self.parent()
        path = self.derive(initial)
        initial.write_text(initial.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ProtocolError, "parent protocol SHA changed"):
            load_source_holdout_protocol(path)

    def test_old_training_clip_cannot_replace_initial_test_even_if_generic_audit_passes(self):
        path = self.derive(self.parent())
        manifest = path.parent / "private" / "test.jsonl"
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        # Same seven labels/counts/durations and no remaining train overlap:
        # the extra provenance check must still reject this promotion.
        promoted = dict(self.rows["train"][0], split="test", cache_split="train")
        rows[0] = promoted
        self.write_rows(manifest, rows)
        summary = json.loads(path.read_text())
        summary["manifests"]["test"]["sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ProtocolError, "exact declared parent subsets"):
            load_source_holdout_protocol(path)

    def test_explicit_cache_route_is_preserved(self):
        self.rows["test"][0]["cache_split"] = "val"
        protocol = load_source_holdout_protocol(self.derive(self.parent()))
        record = protocol.records["test"][0]
        self.assertEqual(record.split, "test")
        self.assertEqual(protocol.cache_record(record).split, "val")

    def test_derived_parent_and_duplicate_selection_are_rejected(self):
        initial = self.parent()
        with self.assertRaisesRegex(ProtocolError, "must not be repeated"):
            self.derive(initial, sources=[self.domains[0], self.domains[0]])
        derived = self.derive(initial)
        with self.assertRaisesRegex(ProtocolError, "not another derivative"):
            derive_source_holdout(derived, [self.domains[0]], self.root / "nested",
                                 source_selection_exposure=PREDECLARED_SELECTION)

    def test_no_overwrite_and_cli_output_has_only_aliases(self):
        initial = self.parent()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = main(["derive", "--protocol", str(initial), "--heldout-source", self.domains[0],
                           "--source-selection-exposure", PREDECLARED_SELECTION, "--output", str(self.output)])
        self.assertEqual(result, 0)
        self.assertNotIn(self.domains[0], stdout.getvalue())
        self.assertNotIn(str(self.root), stdout.getvalue())
        original = (self.output / "protocol.json").read_bytes()
        with self.assertRaisesRegex(ProtocolError, "overwrites are forbidden"):
            self.derive(initial)
        self.assertEqual((self.output / "protocol.json").read_bytes(), original)

    def test_insufficient_cli_is_machine_readable_and_creates_no_output(self):
        initial = self.parent()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = main(["derive", "--protocol", str(initial), "--heldout-source", "private-missing-source",
                           "--source-selection-exposure", PREDECLARED_SELECTION, "--output", str(self.output)])
        self.assertEqual(result, 3)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "data_insufficient")
        self.assertNotIn("private-missing-source", stdout.getvalue())
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
