#!/bin/bash
set -euo pipefail

# Run PrismNet training (+optional eval) sequentially for all *_Hela.h5 under data/clip_data.
#
# This script avoids output overwrite by creating a unique out_dir per run (timestamp)
# and per dataset (protein). It calls tools/main.py directly.
#
# Usage:
#   bash tools/run_all_hela_h5.sh [--eval] [--structure_source shape|pretrain] [--pretrain_* ...] [--batch_size N] ...
#
# Examples:
#   bash tools/run_all_hela_h5.sh --eval
#   bash tools/run_all_hela_h5.sh --eval --structure_source pretrain --pretrain_framework data2vec --pretrain_timestamp 20260324T045257 --pretrain_checkpoint final --pretrain_amp --batch_size 32

DATA_DIR="data/clip_data"
ARCH="PrismNet"
MODE="pu"

RUN_ID=$(date +%Y%m%dT%H%M%S)
OUT_ROOT="out/batch_${RUN_ID}"

mkdir -p "$OUT_ROOT"

echo "[Info] DATA_DIR=$DATA_DIR"
echo "[Info] OUT_ROOT=$OUT_ROOT"

# Prefer enumerating actual .h5 files if present; otherwise fall back to all.list names.
shopt -s nullglob
H5_FILES=("${DATA_DIR}"/*_Hela.h5)
shopt -u nullglob

if [ ${#H5_FILES[@]} -gt 0 ]; then
  for h5 in "${H5_FILES[@]}"; do
    p=$(basename "$h5" .h5)
    out_dir="${OUT_ROOT}/${p}"
    echo "[Run] ${p} -> ${out_dir}"
    python -u tools/main.py \
      --train \
      --p_name "${p}.h5" \
      --data_dir "$DATA_DIR" \
      --mode "$MODE" \
      --arch "$ARCH" \
      --out_dir "$out_dir" \
      "$@"
  done
else
  if [ ! -f "${DATA_DIR}/all.list" ]; then
    echo "[Error] No *_Hela.h5 found and ${DATA_DIR}/all.list is missing." >&2
    exit 1
  fi

  while IFS= read -r p; do
    [ -z "$p" ] && continue
    out_dir="${OUT_ROOT}/${p}"
    echo "[Run] ${p} -> ${out_dir}"
    python -u tools/main.py \
      --train \
      --p_name "$p" \
      --data_dir "$DATA_DIR" \
      --mode "$MODE" \
      --arch "$ARCH" \
      --out_dir "$out_dir" \
      "$@"
  done < "${DATA_DIR}/all.list"
fi

echo "[Done] Results under: ${OUT_ROOT}"
