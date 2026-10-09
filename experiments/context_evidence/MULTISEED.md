# Paired temporal continuation stability

Reuse completed seed42; add only seeds10 and1217 for legacy_continue and
context_delta_continue. Freeze the same A42 features and initialize every job
strictly from the same completed ordinary TCN42 checkpoint. This is NOT whole
system three-seed training. val823 is a repeatedly used development set.

Architecture, 30 epochs, 16-segment batches, AdamW .001/.01, clipping5, normal
and unchanged-anchor CE weights .5/.5 and earliest best full7-class F1 remain
unchanged. No special neighbor replacement is applied in training. Fixed42
anchor eligibility/challenge is retained; each training seed changes segment
orders, anchor orders and per-step dropout, identically within its paired controls.
No A export, visual retraining, test data, resume, tuning or extra seeds.

Entry: `python -m experiments.context_evidence.multiseed` with --index,
--source-protocol-dir, --base-run, --seed42-run, --output, --expected-commit,
--python, --gpus 0,1,2,3 and --gpu-uuids UUID0,UUID1,UUID2,UUID3.
Default is a read-only sealed plan (no GPU probe or directory creation).
Add --execute only after the complete plan and current physical inventory pass.

The master output is new; four sibling private outputs carry seed/variant suffixes.
All four jobs independently run CPU tests, historical823 replay, three real CUDA
updates, 30 epochs and strict best-checkpoint evaluation. GPU UUID is mapped to
internal cuda:0, never DDP. No overwrites or retries; on failure other children
finish naturally. Aggregates contain full7-class metrics, confusion matrices,
source/duration/recording aggregate distributions, correction/harm and resource
records inherited from the same evaluator. Final report gives per-seed results,
sample SD and paired F1 differences; historical42 is read-only and SHA sealed.

Tests: `python -m unittest tests.test_context_multiseed tests.test_context_evidence`
plus inherited relation/VideoMAE/direction tests. Output artifacts, paths, IDs,
features, weights, sampling plans and raw logs must remain private.
