# Explicit cache-only recovery

This entry point does not train models or restart the failed visual launcher.
It consumes the already sealed, corrected train7307/Menzhen-dev888 protocol.
Menzhen was authorized for model selection and is not an independent test.
Original outputs and original decoding code remain untouched.

The original long-lived CPU decoder was SIGKILLed. The exact initiator was not
available; cgroup memory-high events alone do not prove an OOM kill. Recovery
therefore bounds decoder lifetime to at most 16 clips from one video per fresh
subprocess, monitors worker RSS (12 GiB), enforces a Linux address-space ceiling
of 32 GiB and applies a 900-second job timeout.
Only its own worker can be stopped. No other job or GPU process is interrupted.
Failure preserves the new output and never retries automatically.

Missing train clips and validation clips are decoded first. Existing train
arrays have no complete original index; recovery re-reads real decoded PTS,
recomputes the unchanged 16-frame view and requires exact pixel equality before
importing an old array. A hash/shape check alone is not called pixel replay.
Old bytes are hardlinked when possible, otherwise copied without replacement.
Every committed clip immediately has a durable private metadata sidecar.
Only after full coverage do compatible `private_index.json` / `cache_seal.json`
files appear. Unsealed output is not ready for training.

Use the pinned Python environment with Decord CPU, numpy, torch and opencv.
CPU decode is intentional; this does not claim an accelerated GPU decoder.
No frames, timestamps, spatial transforms, labels or splits are changed.

```bash
python -m pytest tests/test_hospital_cache_recovery.py tests/test_hospital_holdout.py -q

# Substitute actual absolute private paths and this checkout's full commit.
python -m experiments.hospital_holdout.cache_recovery \
  --failed-run "$FSN_FAILED_RUN" --output "$FSN_RECOVERY_OUTPUT" \
  --expected-commit "$FSN_RECOVERY_COMMIT"

# Separate new smoke output: one previously cached clip, the first missing
# training clip and one hospital-development clip; no complete cache seal.
python -m experiments.hospital_holdout.cache_recovery \
  --failed-run "$FSN_FAILED_RUN" --output "$FSN_RECOVERY_SMOKE_OUTPUT" \
  --expected-commit "$FSN_RECOVERY_COMMIT" --smoke --execute

# After successful tests, smoke and identical read-only full plan, persist the
# SAME command with --execute using nohup. Choose a new log outside checkout;
# open it exclusively. Do not overwrite a previous log/output.
python -m experiments.hospital_holdout.cache_recovery \
  --failed-run "$FSN_FAILED_RUN" --output "$FSN_RECOVERY_OUTPUT" \
  --expected-commit "$FSN_RECOVERY_COMMIT" --execute
```

All run protocols, jobs, inventories, sidecars, logs and caches are private and
must not enter Git. Publish only code, synthetic tests and deidentified docs.
Formal training requires separate confirmation of a complete compatible cache,
fixed training code and a fresh output; this cache-only launcher cannot train A
or queue temporal experiments.
