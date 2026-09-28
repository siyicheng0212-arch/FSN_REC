# FSN-v3 three-seed metric summary

All values are means over seeds 42, 123, and 2026 on the frozen 823-clip
validation split.  Values after `±` are sample standard deviations across the
three seeds.  There is no independent test result in this protocol.

## Overall metrics

| Metric | Original visual model | FSN-v3 | Absolute change |
|---|---:|---:|---:|
| Accuracy / micro-F1 | 0.8226 ± 0.0201 | **0.8518 ± 0.0061** | **+0.0292** |
| Macro-F1 | 0.7760 ± 0.0193 | **0.8098 ± 0.0127** | **+0.0339** |
| Weighted-F1 | 0.8249 ± 0.0183 | **0.8510 ± 0.0054** | **+0.0261** |
| Macro precision | 0.7944 ± 0.0246 | **0.8391 ± 0.0085** | **+0.0447** |
| Macro recall / balanced accuracy | 0.7662 ± 0.0102 | **0.7874 ± 0.0173** | **+0.0212** |
| Macro specificity | 0.9631 ± 0.0036 | **0.9671 ± 0.0012** | **+0.0039** |
| Cohen's kappa | 0.7492 ± 0.0271 | **0.7872 ± 0.0081** | **+0.0380** |
| Multiclass MCC | 0.7500 ± 0.0268 | **0.7890 ± 0.0084** | **+0.0390** |

FSN-v3 improves every overall metric and substantially reduces cross-seed
variance for macro-F1, accuracy, precision, kappa, and MCC.

## Per-class metrics

| Class | Support | Original F1 | FSN-v3 F1 | Change |
|---|---:|---:|---:|---:|
| Disinfection | 40 | 0.7574 ± 0.0334 | **0.8125 ± 0.0321** | **+0.0551** |
| Needle insertion | 119 | 0.9218 ± 0.0172 | **0.9489 ± 0.0073** | **+0.0271** |
| Needle manipulation | 151 | 0.8461 ± 0.0217 | **0.8761 ± 0.0127** | **+0.0300** |
| Sweeping | 387 | 0.8531 ± 0.0174 | **0.8710 ± 0.0076** | **+0.0179** |
| Reperfusion | 72 | 0.5413 ± 0.0234 | **0.5801 ± 0.0103** | **+0.0388** |
| Needle removal | 48 | 0.7847 ± 0.0405 | **0.8176 ± 0.0403** | **+0.0329** |
| Fixation | 6 | 0.7273 ± 0.0000 | **0.7626 ± 0.0612** | **+0.0354** |

All seven mean class F1 values improve.  Reperfusion remains the weakest class.
Fixation has only six validation samples, so its apparent change is not a
reliable basis for method claims.

## Directional confusion

Across three seeds (2,469 predictions total):

| Error direction | Original | FSN-v3 | Change |
|---|---:|---:|---:|
| Sweeping → reperfusion | 114 | **79** | **-35 (-30.7%)** |
| Reperfusion → sweeping | **86** | 93 | +7 (+8.1%) |
| Combined two-way errors | 200 | **172** | **-28 (-14.0%)** |

The procedural prior fixes many false reperfusion predictions, but slightly
increases the reverse error.  Both directions must therefore be reported.

## Duration slices

Fixed seven-class macro-F1 is misleading for the long-duration slice because
two classes have zero support.  The support-aware macro-F1 is the primary
diagnostic below.

| Slice | Metric | Original | FSN-v3 | Change |
|---|---|---:|---:|---:|
| ≤10 seconds (651) | Accuracy | 0.8285 | **0.8571** | **+0.0287** |
| ≤10 seconds | Present-class macro-F1 | 0.7736 | **0.8008** | **+0.0272** |
| >10 seconds (172) | Accuracy | 0.8004 | **0.8314** | **+0.0310** |
| >10 seconds | Present-class macro-F1 | 0.7445 | **0.7763** | **+0.0318** |

The gain is not confined to either short or long clips.

## Source slices

| Source | Clips | Original accuracy | FSN-v3 accuracy | Change |
|---|---:|---:|---:|---:|
| FSN | 166 | 0.7369 | **0.7731** | **+0.0361** |
| bilibili | 37 | 0.7568 | **0.8018** | **+0.0450** |
| Douyin | 263 | 0.8619 | **0.8821** | **+0.0203** |
| Kuaishou | 34 | **0.9020** | 0.8922 | -0.0098 |
| Lishui | 209 | 0.8405 | **0.8900** | **+0.0494** |
| Outpatient clinic | 89 | 0.8202 | **0.8277** | **+0.0075** |
| YouTube | 25 | 0.8267 | **0.8400** | **+0.0133** |

Six of seven source accuracies improve.  Kuaishou decreases by roughly one
percentage point on only 34 clips and should be followed up rather than hidden.

## Uncertainty and interpretation

- Mean paired macro-F1 gain: `+0.03388`, sample SD `0.02879`.
- Mean paired accuracy gain: `+0.02916`, sample SD `0.01752`.
- Every seed improves, but the descriptive n=3 Student-t interval for the
  macro-F1 gain crosses zero.  The experiment is promising but not a formal
  significance claim.
- The decoder assumes chronologically ordered clips from the same record.
  Singleton clips retain the original visual prediction.
- Deployment on unsegmented video still requires a boundary proposal or
  temporal localization stage.
