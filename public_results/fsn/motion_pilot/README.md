# FSN equal-budget motion sampling pilot

Both arms use original Uni-AdaFocus with 36 candidate frames, matched pretrained weights,
training settings, frozen source-record splits, and seeds 42, 123, and 2026.
This compares sampling only; it does not add a role-aware motion module.

| Seed | Uniform macro-F1 | Three-window macro-F1 | Dense − uniform |
|---:|---:|---:|---:|
| 42 | 0.7638 | 0.7582 | -0.0056 |
| 123 | 0.7853 | 0.7530 | -0.0322 |
| 2026 | 0.7899 | 0.7774 | -0.0125 |

Mean macro-F1: uniform **0.7796**, three-window **0.7629**, paired difference **-0.0168**.
Mean accuracy: uniform 0.8218, three-window 0.8165.
Mean weighted-F1: uniform 0.8191, three-window 0.8124.
With only three seeds, the interval in summary.json is descriptive.

| Arm | Sweep → reperfusion errors (3-seed mean) | Reperfusion → sweep errors (3-seed mean) |
|---|---:|---:|
| uniform | 25.0 | 35.0 |
| three_windows | 22.0 | 35.0 |

Three-window sampling had lower seven-class macro-F1 in all three paired seeds.
This is a negative result for replacing the uniform sampler with this fixed design.

| Class (validation support) | Uniform F1 | Three-window F1 | Difference |
|---|---:|---:|---:|
| 消毒 (40) | 0.7264 | 0.7647 | +0.0383 |
| 进针 (119) | 0.8987 | 0.8899 | -0.0089 |
| 运针 (151) | 0.8470 | 0.8245 | -0.0225 |
| 扫散 (387) | 0.8532 | 0.8529 | -0.0003 |
| 再灌注 (72) | 0.5188 | 0.4969 | -0.0219 |
| 拔针 (48) | 0.7851 | 0.7799 | -0.0053 |
| 固定 (6) | 0.8283 | 0.7314 | -0.0969 |

Sensitivity excluding the six-sample fixation class: mean F1 over the other six classes is 0.7715 for uniform and 0.7681 for three-window.  Sweeping F1 is essentially unchanged, while reperfusion F1 falls under three-window sampling.

| Duration | Support | Uniform present-class macro-F1 | Three-window present-class macro-F1 |
|---|---:|---:|---:|
| ≤10 s | 651 | 0.7744 | 0.7678 |
| >10 s | 172 | 0.7028 | 0.7076 |

| Source | Clips | Uniform accuracy | Three-window accuracy |
|---|---:|---:|---:|
| FSN | 166 | 0.7590 | 0.7831 |
| bilibili | 37 | 0.6937 | 0.6847 |
| dy | 263 | 0.8530 | 0.8669 |
| ks | 34 | 0.8725 | 0.8824 |
| lishui | 209 | 0.8469 | 0.7799 |
| menzhen | 89 | 0.8127 | 0.8240 |
| youtube | 25 | 0.8533 | 0.8933 |

Lishui accuracy decreases for three-window sampling in all three seeds; the aggregate source effect is not uniform. The fixed windows may miss useful parts of a long clip, but frame-selection and visibility audits are needed to establish the cause.

Report class-wise F1, both error directions, and source/duration slices from
[the aggregate JSON](three_seed_comparison.json) alongside the
[training and validation curves](learning_curves.svg). Source slices may omit classes;
their fixed-seven-class macro-F1 must not be compared as if all classes were present.
Fixation has only six evaluation clips, so its F1 is especially unstable.

The source corpus includes edited online videos. This experiment makes no claim
about continuous clinical workflow, direct blood reperfusion measurement,
or detection of clinician and patient roles.

## Controlled within-clip fusion diagnostic

Using each arm's independently trained best checkpoint for the same seed, we
also averaged their seven-class softmax probabilities at equal weight. This
uses **only the current clip** and does not fit fusion weights. It is a
diagnostic ensemble of two complete models, not a trained new architecture;
its inference cost is correspondingly higher.

| Seed | Uniform | Dense | Uniform+dense | Uniform+uniform control |
|---:|---:|---:|---:|---:|
| 42 | 0.76378 | 0.75817 | 0.76796 | 0.78433 |
| 123 | 0.78528 | 0.75304 | 0.79490 | 0.79124 |
| 2026 | 0.78988 | 0.77739 | 0.79384 | 0.78985 |
| Mean | **0.77965** | 0.76287 | 0.78557 | **0.78847** |

The fixed fusion beats uniform alone by 0.00417, 0.00962, and 0.00396
macro-F1 for the three paired seeds (mean +0.00592). Across three repetitions
of the 823 validation clips, the dense model uniquely classifies 142 samples
correctly when uniform is wrong, while uniform uniquely classifies 155 when
dense is wrong. The fixed fusion corrects 70 uniform errors and harms 33
uniform correct predictions. **This does not establish that higher temporal
sampling caused the complementarity.** A same-cost control averaging two
different-seed uniform models reaches 0.78847, higher than uniform+dense
at 0.78557. Cyclic control pairs are 42+123, 123+2026, and 2026+42;
they do not perfectly match same-seed training stochasticity, so they are
a diagnostic rather than a definitive causal estimate.

For **393 of 823 clips (duration ≤3 s)**, the three-window sampling code
uses exactly the same uniform timestamps. Any model disagreement there must
arise from training differences, not additional dense-frame evidence.
On the 430 clips longer than 3 s, uniform+dense mean seven-class macro-F1
is 0.74067, below the two-uniform control's 0.75122. The same conclusion
holds for the whole validation set; there is currently **no evidence that
the fixed dense-window sampling contributes beyond ordinary ensembling**.
The duration-slice macro-F1 values use the same seven classes in each method,
but rare classes have small support in these slices.

The clinically important error trade-off remains: sweeping incorrectly
predicted as reperfusion falls from 75 to 59 across the three seeds, while
reperfusion incorrectly predicted as sweeping rises from 105 to 112.
Reperfusion F1 changes only from about 0.5188 to 0.5200. This diagnostic
does **not** establish improved bidirectional distinction or clinician/patient
activity recognition. It does not justify a new dense-motion architecture
by itself. The [controlled aggregate](fixed_dual_rate_fusion_controlled.json)
contains overall and duration-slice comparisons; the earlier
[uncontrolled diagnostic](fixed_dual_rate_fusion.json) is retained for audit,
not as evidence of a dense-motion benefit. No private clip predictions or
videos are published.
