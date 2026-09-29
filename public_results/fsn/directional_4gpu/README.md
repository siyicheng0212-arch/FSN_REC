# Four-GPU within-clip directional-evidence comparison

Status: **completed, negative method gate**. Four jobs finished with exit code
0 and readable `result.json`, `best.pt`, and `last.pt`. All four RTX 3090s
were idle at the completion check. This experiment trained on 7,372 clips and
selected checkpoints on the same 823-clip **development-validation** split
used in prior experiments. It has **no independent test result**. The
training/evaluation manifest hashes and full aggregate metrics are in
[three_seed_comparison.json](three_seed_comparison.json); [epoch curves](learning_curves.svg)
are also provided. No raw video, annotation, clip prediction, log, cache, or
checkpoint is published.

The four new jobs used commit `5e01c6b`; reused Original seeds 42 and
2026 came from the earlier `0ad5aa3` run protocol. The data, sampling,
official initialization, and optimization settings were matched, but the
code provenance is not identical.

The candidate adds opt-in within-clip latent directional/co-occurrence
evidence to the unchanged Original path, with the same uniform 36-frame
input. Its two streams have **no clinician/patient activity supervision**;
they must not be interpreted as validated clinical-role detectors.

| Seed | Original best Macro-F1 | Candidate best Macro-F1 | Paired difference | Best finetune epochs, Original / candidate |
|---|---:|---:|---:|---:|
| 42 | 0.7638 | 0.7689 | +0.0051 | 16 / 27 |
| 2026 | 0.7899* | 0.7568 | -0.0330 | 39* / 18 |
| 1217 | 0.7524 | 0.7871 | +0.0346 | 15 / 45 |
| Three-seed mean | **0.7687** | **0.7709** | **+0.0022** | — |

The paired-difference sample standard deviation is **0.0339**. Candidate
mean accuracy is **0.8129**, below Original's **0.8190**; weighted F1 is
0.8128 versus 0.8157. The candidate adds 15,139 trainable parameters
(27,224,006 versus 27,208,867). Loss continued falling while validation
F1 oscillated and often retreated after the best epoch. Thus the small
mean Macro-F1 increase is not stable evidence of improvement.

\* Original seed 2026 was interrupted before normal early stopping. Its
finetune-39 `best.pt` was loaded and **re-inferred** on the same 823
development-validation clips, reproducing Macro-F1 0.78987855. It has no
normal-exit `result.json` or complete curve and must not be presented as
a completed early-stopped run.

## The task-critical errors move in opposite directions

Across three seeds, true sweeping predicted as reperfusion rose from
**73/1,161** (6.29%) with Original to **94/1,161** (8.10%) with the
candidate. True reperfusion predicted as sweeping fell from **113/216**
(52.31%) to **93/216** (43.06%). Mean reperfusion F1 rose from 0.4987 to
0.5315, while sweeping F1 fell from 0.8525 to 0.8437. This is a
**directional trade-off**, not a solution of both errors.

| Class (validation support) | Original mean F1 | Candidate mean F1 |
|---|---:|---:|
| 消毒 (40) | 0.7535 | 0.7516 |
| 进针 (119) | 0.8866 | 0.8916 |
| 运针 (151) | 0.8403 | 0.8311 |
| 扫散 (387) | 0.8525 | 0.8437 |
| 再灌注 (72) | 0.4987 | 0.5315 |
| 拔针 (48) | 0.8018 | 0.7914 |
| 固定 (6) | 0.7475 | 0.7555 |

The fixed class has only six validation clips, so its per-seed F1 varies
greatly. Full per-seed precision/recall/F1, seven-class confusion matrices,
and source/duration slices are in the safe aggregate JSON. For clips at
most 10 seconds (n=651), mean Macro-F1 was 0.7589→0.7696; for clips
longer than 10 seconds (n=172), 0.5271→0.5331, with a marked decline on
seed 42. Source slices are descriptive only: some lack classes, so their
seven-class Macro-F1 is not comparable to the overall score.

## Does the added branch actually change the prediction?

At each candidate's own best checkpoint, identical validation inputs
were evaluated with the new correction disabled versus enabled. Checkpoint
SHA-256 values from all three audits match the **final** best checkpoints,
including the seed-1217 audit that was originally run before training ended.

| Seed | Branch off → on Macro-F1 | Changed predictions | Corrected / harmed |
|---|---:|---:|---:|
| 42 | 0.76956 → 0.76886 | 1/823 | 0 / 1 |
| 2026 | 0.75675 → 0.75683 | 3/823 | 1 / 2 |
| 1217 | 0.78707 → 0.78707 | 0/823 | 0 / 0 |

Only **4/2,469** predictions changed: one was corrected and three were
harmed. The mean absolute pair-logit corrections were about
0.0032/0.0046/0.0014, versus main-path logit magnitudes
7.96/5.34/10.84. The candidate/Original differences therefore mostly
reflect different *joint training trajectories*, not the intended
additional inference mechanism. The current architecture has not earned
a clinical-mechanism or performance claim.

Historical IEA best scores for the overlapping seeds were 0.7775 (42)
and 0.7579 (2026); the present candidate reached 0.7689 and 0.7568,
respectively. This is only descriptive, not a new paired architecture
ablation. FSN-v3 uses neighboring clips and a different information
budget; it must be reported separately, never as a single-clip baseline.

The next justified step is a train-only clinical visibility/overlap
annotation and an independent held-out evaluation, not another
validation-driven architecture sweep. A separate low-cost train-internal
[pair-probe diagnostic](https://github.com/siyicheng0212-arch/FSN_REC/pull/8)
likewise failed to surpass the pretrained Original reference.
