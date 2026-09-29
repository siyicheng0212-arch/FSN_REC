# FSN-IEA train-internal paired pilot: negative mechanism result

Status: **completed normally; no claim of improvement**. Original and the
opt-in within-clip evidence candidate ran on separate RTX 3090 GPUs with the
same official SSv2 initialization, uniform 36-frame input, seed 42, and
5 head-warm-up plus 10 fine-tuning epochs. Both processes exited with code 0.
The 6,586/786 train/inner-validation clips were split by video group from
the original training set alone. The previously designated 823-clip
validation set and any independent test were **not** used to choose this
architecture. The code commit and frozen manifest hashes are in the
[aggregate-only JSON](../public_results/fsn/iea/inner_pilot_summary.json).

| Best inner-validation metric | Original | FSN-IEA |
|---|---:|---:|
| Best fine-tune epoch | 8 | 5 |
| Seven-class macro-F1 | **0.80355** | 0.79930 |
| Accuracy | **0.82570** | 0.80025 |
| Sweep F1 | **0.8470** | 0.8167 |
| Reperfusion F1 | 0.5900 | 0.6000 |
| Sweep → reperfusion errors | **28** | 41 |
| Reperfusion → sweep errors | 49 | **43** |
| Trainable parameters | 27,208,867 | 27,224,418 |

The small reperfusion F1 gain trades off against a larger sweeping decline
and 13 additional sweep-to-reperfusion errors. This does not meet the
predeclared bidirectional criterion. Both train losses kept falling during
fine-tuning; IEA peaked earlier and did not recover its best validation F1
by epoch 10. With one seed, small rare-class supports, and different
best-selected epochs, these differences are descriptive rather than a
statistical superiority test.

Crucially, the proposed mechanism **barely activated**. Server-side
re-evaluation of IEA's best checkpoint on the 786 inner-validation clips
found a mean absolute residual of 0.00767 logits versus 4.30951 for the
baseline path. The residual changed only **one** final prediction, harming
one originally correct clip and correcting none. Sweeping and reperfusion
predictions were unchanged by the residual on that checkpoint. Therefore,
the class-metric differences between independently trained Original and IEA
mostly reflect training trajectories, not learned within-clip motion
evidence. The module passed gradient, initial-equivalence, short smoke, and
two-clip overfit checks, but those checks did not establish population-level
utility.

Decision: **do not run a formal three-seed IEA experiment or claim a paper
contribution from this version**. Further work must first establish a
reliably observable cue and a clinically defensible primary-label rule for
co-occurring practitioner and patient actions. Do not repair this negative
result by repeatedly tuning on the 823-clip validation set. No video,
annotation, clip identity, individual prediction, raw log, cache, or model
weight is included in this report.
