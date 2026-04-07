#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"

if [[ -z "${MODE}" ]]; then
  echo "Usage: $0 {train|compare8|compare32}"
  exit 1
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate januspro
export PYTHONPATH=/root/miniconda3/lib/python3.10/site-packages:${PYTHONPATH:-}

REPO_DIR="/root/autodl-tmp/repos/Janus"
TRAIN_MANIFEST="/root/autodl-tmp/data/emoart_5k/gen_full_official_texture_meta/train.jsonl"
VAL_MANIFEST="/root/autodl-tmp/data/emoart_5k/gen_full_official_texture_meta/val.jsonl"
TRAIN_OUT="/root/autodl-tmp/emoart_gen_runs/out_gen_expA_conservative_crop_headonly_v1"
COMPARE8_OUT="/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_conservative_crop_headonly_8"
COMPARE32_OUT="/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_conservative_crop_headonly_32"
ADAPTER_PATH="${TRAIN_OUT}/final_adapter"

cd "${REPO_DIR}"

case "${MODE}" in
  train)
    python "${REPO_DIR}/train_emoart_gen_lora.py" \
      --train-data "${TRAIN_MANIFEST}" \
      --val-data "${VAL_MANIFEST}" \
      --output-dir "${TRAIN_OUT}" \
      --lora-r 32 \
      --lora-alpha 64 \
      --lora-dropout 0.05 \
      --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
      --generation-module-mode head_only \
      --learning-rate 1e-4 \
      --generation-learning-rate 1e-4 \
      --scheduler-type cosine \
      --gradient-accumulation-steps 16 \
      --num-epochs 3 \
      --image-preprocess-mode crop \
      --dtype bf16 \
      --prompt-template default \
      --art-texture-mode off \
      --art-texture-fields all \
      --art-texture-prob 0.0
    ;;
  compare8)
    python "${REPO_DIR}/compare_emoart_gen.py" \
      --manifest-path "${VAL_MANIFEST}" \
      --adapter-path "${ADAPTER_PATH}" \
      --output-dir "${COMPARE8_OUT}" \
      --num-samples 8 \
      --seed 42 \
      --dtype bf16 \
      --image-preprocess-mode crop \
      --sample-strategy greedy \
      --prompt-template default \
      --art-texture-mode off \
      --art-texture-fields all
    ;;
  compare32)
    python "${REPO_DIR}/compare_emoart_gen.py" \
      --manifest-path "${VAL_MANIFEST}" \
      --adapter-path "${ADAPTER_PATH}" \
      --output-dir "${COMPARE32_OUT}" \
      --num-samples 32 \
      --seed 42 \
      --dtype bf16 \
      --image-preprocess-mode crop \
      --sample-strategy greedy \
      --prompt-template default \
      --art-texture-mode off \
      --art-texture-fields all
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Usage: $0 {train|compare8|compare32}"
    exit 1
    ;;
esac
