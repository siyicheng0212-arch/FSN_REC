#!/usr/bin/env bash
# Four independent single-GPU comparisons; this is not four-GPU DDP.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${FSN_MANIFEST_DIR:?Set FSN_MANIFEST_DIR to an existing train/val manifest directory}"
: "${FSN_CACHE_DIR:?Set FSN_CACHE_DIR to a valid uniform 36f224 cache}"
: "${FSN_CHECKPOINT:?Set FSN_CHECKPOINT to the official SSv2 AdaFocus pretrained checkpoint}"
FSN_PYTHON="${FSN_PYTHON:-python}"
FSN_OUTPUT_DIR="${FSN_OUTPUT_DIR:-local_motion_results/four_gpu}"
FSN_GPU_IDS="${FSN_GPU_IDS:-0,1,2,3}"
FSN_SEED="${FSN_SEED:-42}"
FSN_EPOCHS="${FSN_EPOCHS:-100}"
FSN_PATIENCE="${FSN_PATIENCE:-10}"
FSN_BATCH_SIZE="${FSN_BATCH_SIZE:-4}"
FSN_ACCUMULATION_STEPS="${FSN_ACCUMULATION_STEPS:-16}"
FSN_WORKERS="${FSN_WORKERS:-4}"
FSN_LR="${FSN_LR:-0.002}"
FSN_DRY_RUN="${FSN_DRY_RUN:-0}"
IFS=',' read -r -a gpu_ids <<< "$FSN_GPU_IDS"
variants=(original local_appearance local_motion local_motion_context)
if [[ ${#gpu_ids[@]} -ne 4 ]]; then
  echo 'FSN_GPU_IDS must contain exactly four distinct device IDs/UUIDs.' >&2
  exit 2
fi
declare -A seen_gpus=()
for gpu in "${gpu_ids[@]}"; do
  if [[ -z "$gpu" || -n "${seen_gpus[$gpu]:-}" ]]; then
    echo 'GPU IDs must be nonempty and unique.' >&2
    exit 2
  fi
  seen_gpus[$gpu]=1
done
for input_file in "$FSN_MANIFEST_DIR/train.jsonl" "$FSN_MANIFEST_DIR/val.jsonl" "$FSN_CHECKPOINT"; do
  [[ -f "$input_file" ]] || { echo "Missing input: $input_file" >&2; exit 2; }
done
[[ -d "$FSN_CACHE_DIR" ]] || { echo 'Missing cache directory.' >&2; exit 2; }
for variant in "${variants[@]}"; do
  run_dir="$FSN_OUTPUT_DIR/$variant/seed_$FSN_SEED"
  if [[ -e "$run_dir/best.pt" || -e "$run_dir/result.json" ]]; then
    echo "Existing run at $run_dir; use a new FSN_OUTPUT_DIR to avoid overwriting." >&2
    exit 2
  fi
done

if [[ "$FSN_DRY_RUN" != 1 ]]; then
  for gpu in "${gpu_ids[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" "$FSN_PYTHON" -c 'import torch; assert torch.cuda.device_count() == 1, "Expected one visible GPU"; assert torch.cuda.is_bf16_supported(), "Trainer requires bf16 GPU support"'
  done
  mkdir -p "$FSN_OUTPUT_DIR/launcher_logs"
fi
common=(--manifest-dir "$FSN_MANIFEST_DIR" --cache-dir "$FSN_CACHE_DIR"
  --checkpoint "$FSN_CHECKPOINT" --output-dir "$FSN_OUTPUT_DIR"
  --seed "$FSN_SEED" --epochs "$FSN_EPOCHS" --patience "$FSN_PATIENCE"
  --batch-size "$FSN_BATCH_SIZE" --accumulation-steps "$FSN_ACCUMULATION_STEPS"
  --workers "$FSN_WORKERS" --lr "$FSN_LR" --weight-decay 0.0005
  --global-lr-ratio 0.5 --stn-lr-ratio 0.2 --temporal-lr-ratio 0.2
  --head-warmup-epochs 5 --head-warmup-lr 0.001 --module-warmup-epochs 0
  --class-weight-mode sqrt_inverse --clip-grad 20
  --local-motion-dim 64 --local-motion-window 3 --local-motion-temperature 0.07
  --local-motion-context-grid 2 --local-motion-lr-ratio 1)
pids=()
cancel_children() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap 'cancel_children; exit 130' INT
trap 'cancel_children; exit 143' TERM
for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  command=("$FSN_PYTHON" -u -m experiments.train_adafocus --variant "$variant" "${common[@]}")
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu_ids[$index]}"
  printf '%q ' "${command[@]}"
  printf '\n'
  if [[ "$FSN_DRY_RUN" == 1 ]]; then continue; fi
  CUDA_VISIBLE_DEVICES="${gpu_ids[$index]}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
    "${command[@]}" > "$FSN_OUTPUT_DIR/launcher_logs/${variant}_seed_${FSN_SEED}.log" 2>&1 &
  pids+=("$!")
  printf '%s\t%s\t%s\n' "$variant" "${gpu_ids[$index]}" "$!" >> "$FSN_OUTPUT_DIR/launcher_logs/pids.tsv"
done
status=0
for index in "${!pids[@]}"; do
  exit_code=0
  wait "${pids[$index]}" || exit_code=$?
  printf '%s\t%s\n' "${variants[$index]}" "$exit_code" >> "$FSN_OUTPUT_DIR/launcher_logs/exit_codes.tsv"
  if [[ "$exit_code" != 0 ]]; then status=1; fi
done
exit "$status"
