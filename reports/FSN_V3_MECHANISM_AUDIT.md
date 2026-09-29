# Why FSN-v3 improved: frozen-logit mechanism audit

This is an exploratory analysis of the existing 823-clip **validation** set,
not an independent test. The same frozen visual logits from seeds 42, 123,
and 2026 were reused for every row below. Only the seven-class sequence
prior was changed. All controls use training labels only, transition weight
1.0, and the same video grouping. The complete sanitized aggregate and
per-seed results are in
[`public_results/fsn/v3/first_order_ablation.json`](../public_results/fsn/v3/first_order_ablation.json).
The frozen manifest SHA-256 hashes are in that JSON. No clip IDs, videos,
individual predictions, or local paths are published.

| Decoder/control | Mean validation macro-F1 | Delta vs visual |
|---|---:|---:|
| Independent visual logits | 0.77595 | — |
| Uniform start and transitions | 0.77595 | 0.00000 |
| Empirical first-label prior only | 0.77981 | +0.00386 |
| Empirical directed transitions only | 0.80893 | +0.03298 |
| FSN-v3 full directed bigram | **0.80983** | **+0.03388** |
| Training order shuffled within each file, labels preserved | 0.75420 | −0.02175 |
| Validation inference order shuffled within each file | 0.72505 | −0.05090 |
| Global same-class persistence, no class-pair identity | 0.77754 | +0.00159 |
| Unordered neighboring class pairs, no direction | 0.78190 | +0.00595 |
| Independent next-class frequency at each step | 0.75980 | −0.01615 |

**Inference:** Almost all of the measured gain requires *directed order of
neighboring annotated segments*. It is not explained by the first-action
frequency, a global preference to repeat the same class, or an unordered
co-occurrence table. The result holds descriptively for all three seeds,
but three seeds and one repeatedly examined validation set do not establish
generalization or statistical significance.

The matrix is soft, not a forced script: in the fitted training prior, a
`扫散` segment is followed by `扫散` with probability 0.573, `再灌注` with 0.187,
and `拔针` with 0.109. `再灌注` can be followed by `扫散` with probability 0.286.
Viterbi combines this prior with clip visual probabilities; it changes only
137 of 2,469 clip predictions across three seeds, correcting 96 originally
wrong predictions and harming 24 originally correct predictions. It reduces
`扫散→再灌注` errors from 114 to 79 but increases the reverse direction from
86 to 93. Thus the headline F1 gain does **not** mean both ambiguous actions
were solved.

Crucially, a media file is not guaranteed to be a continuous treatment.
Online videos can be edited montages, and annotation boundaries can skip
unseen events. The ablation establishes predictiveness of **dataset segment
order**, not that the learned order is a true clinical workflow or that the
decoder understands the needle, practitioner hand, or patient movement.
Treat FSN-v3 as a strong structured post-processing baseline, not the main
clinical innovation. Before claiming workflow transfer, annotate
continuous-vs-edited status and evaluate those strata separately; for
montage/unknown-order records, test a visual-only fallback or an independently
verified continuity rule.

Reproduce with `python -m experiments.ablate_sequence_prior` using the frozen
`train.jsonl`, `val.jsonl`, and the three private
`validation_predictions.jsonl`/`result.json` pairs from FSN-v3. The command's
full arguments and output schema are described in
[`experiments/README.md`](../experiments/README.md). The private per-clip
prediction files must not be committed.
