#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"

if [[ -z "${MODE}" ]]; then
  echo "Usage: $0 {train|compare8|compare32|packet8|packet32}"
  exit 1
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate januspro
export PYTHONPATH=/root/miniconda3/lib/python3.10/site-packages:${PYTHONPATH:-}

REPO_DIR="/root/autodl-tmp/repos/Janus"
TRAIN_MANIFEST="/root/autodl-tmp/emoart_gen_runs/data_risk_audit_train_texture_meta/train_line_structure_focus_v1.jsonl"
VAL_MANIFEST="/root/autodl-tmp/data/emoart_5k/gen_full_official_texture_meta/val.jsonl"
BASE_ADAPTER="/root/autodl-tmp/emoart_gen_runs/out_gen_expA_frameclean_v2/final_adapter"
TRAIN_OUT="/root/autodl-tmp/emoart_gen_runs/out_gen_expA_line_structure_focus_v1"
COMPARE8_OUT="/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_line_structure_focus_v1_8"
COMPARE32_OUT="/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_line_structure_focus_v1_32"
ADAPTER_PATH="${TRAIN_OUT}/final_adapter"

cd "${REPO_DIR}"

case "${MODE}" in
  train)
    python "${REPO_DIR}/train_emoart_gen_lora.py" \
      --train-data "${TRAIN_MANIFEST}" \
      --val-data "${VAL_MANIFEST}" \
      --output-dir "${TRAIN_OUT}" \
      --resume-from-checkpoint "${BASE_ADAPTER}" \
      --per-device-train-batch-size 2 \
      --per-device-eval-batch-size 2 \
      --lora-r 32 \
      --lora-alpha 64 \
      --lora-dropout 0.05 \
      --target-modules q_proj k_proj v_proj o_proj \
      --generation-module-mode frozen \
      --learning-rate 4e-5 \
      --generation-learning-rate 4e-5 \
      --scheduler-type cosine \
      --gradient-accumulation-steps 8 \
      --num-epochs 0.75 \
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
      --parallel-size 2 \
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
      --parallel-size 2 \
      --image-preprocess-mode crop \
      --sample-strategy greedy \
      --prompt-template default \
      --art-texture-mode off \
      --art-texture-fields all
    ;;
  packet8)
    if [[ ! -f "${COMPARE8_OUT}/comparison.jsonl" ]]; then
      echo "Missing compare output: ${COMPARE8_OUT}/comparison.jsonl"
      echo "Run: $0 compare8"
      exit 1
    fi
    python "${REPO_DIR}/build_emoart_compare_triptych_packet.py" \
      --compare-dir "${COMPARE8_OUT}" \
      --output-dir "${COMPARE8_OUT}/triptych_packet"
    ;;
  packet32)
    if [[ ! -f "${COMPARE32_OUT}/comparison.jsonl" ]]; then
      echo "Missing compare output: ${COMPARE32_OUT}/comparison.jsonl"
      echo "Run: $0 compare32"
      exit 1
    fi
    python "${REPO_DIR}/build_emoart_compare_triptych_packet.py" \
      --compare-dir "${COMPARE32_OUT}" \
      --output-dir "${COMPARE32_OUT}/triptych_packet"
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Usage: $0 {train|compare8|compare32|packet8|packet32}"
    exit 1
    ;;
esac
