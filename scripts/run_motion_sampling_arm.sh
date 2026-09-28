#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 uniform|three_windows GPU_INDEX" >&2
  exit 2
fi

sampling=$1
gpu_index=$2
if [[ $sampling != uniform && $sampling != three_windows ]]; then
  echo "invalid sampling mode: $sampling" >&2
  exit 2
fi
if [[ $gpu_index != 0 && $gpu_index != 1 ]]; then
  echo "GPU_INDEX must be 0 or 1" >&2
  exit 2
fi

repo=/root/autodl-tmp/FSN_REC_motion_pilot
manifests=/root/autodl-tmp/FSN_REC/processed_server/manifests_trainval
checkpoint=/root/autodl-tmp/checkpoints/sthv2_p128_8and12.pth.tar
results=/root/autodl-tmp/formal_motion_pilot/results/$sampling
python=/root/miniconda3/bin/python
if [[ $sampling == uniform ]]; then
  cache=/root/autodl-tmp/full_cache_36f224_trainval
else
  cache=/root/autodl-tmp/full_cache_three_windows_36f224
fi

cd "$repo"
echo "start $(date -Is) sampling=$sampling gpu=$gpu_index commit=$(git rev-parse HEAD)"
"$python" -c 'import hashlib,json,sys;from pathlib import Path
config=json.loads(Path("configs/motion_sampling_protocol.json").read_text())
paths={"train_manifest_sha256":Path(sys.argv[1]),"validation_manifest_sha256":Path(sys.argv[2]),"official_checkpoint_sha256":Path(sys.argv[3])}
for key,path in paths.items():
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 if digest!=config[key]: raise SystemExit(f"{key} mismatch: {digest}")
print("frozen manifest and checkpoint hashes verified")' \
  "$manifests/train.jsonl" "$manifests/val.jsonl" "$checkpoint"
"$python" -c 'from pathlib import Path;import sys;from experiments.full_data import FullClipDataset
root=Path(sys.argv[1]);cache=Path(sys.argv[2]);sampling=sys.argv[3]
counts={split:len(FullClipDataset(root/f"{split}.jsonl",cache,36,224,sampling)) for split in ("train","val")}
assert counts=={"train":7372,"val":823},counts
print("cache verified",counts,sampling)' "$manifests" "$cache" "$sampling"

for seed in 42 123 2026; do
  result=$results/original/seed_$seed/result.json
  if [[ -f $result ]]; then
    "$python" -c 'import json,sys;d=json.load(open(sys.argv[1]));assert d["seed"]==int(sys.argv[2]) and d["sampling"]==sys.argv[3] and d["test_metrics"] is None' \
      "$result" "$seed" "$sampling"
    echo "completed result already present: sampling=$sampling seed=$seed"
    continue
  fi
  echo "training start $(date -Is) sampling=$sampling seed=$seed gpu=$gpu_index"
  CUDA_VISIBLE_DEVICES=$gpu_index "$python" -u -m experiments.train_adafocus \
    --variant original \
    --sampling "$sampling" \
    --manifest-dir "$manifests" \
    --cache-dir "$cache" \
    --output-dir "$results" \
    --checkpoint "$checkpoint" \
    --head-warmup-epochs 5 \
    --head-warmup-lr 0.001 \
    --epochs 100 \
    --patience 10 \
    --batch-size 4 \
    --accumulation-steps 16 \
    --workers 4 \
    --lr 0.002 \
    --weight-decay 0.0005 \
    --global-lr-ratio 0.5 \
    --stn-lr-ratio 0.2 \
    --temporal-lr-ratio 0.2 \
    --fsn-module-lr-ratio 1.0 \
    --class-weight-mode sqrt_inverse \
    --clip-grad 20 \
    --seed "$seed"
  echo "training complete $(date -Is) sampling=$sampling seed=$seed"
done
echo "arm complete $(date -Is) sampling=$sampling"
