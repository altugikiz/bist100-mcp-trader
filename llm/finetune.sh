#!/usr/bin/env bash
# LoRA fine-tuning of Qwen3-4B on harness transcripts (MLX, Apple silicon), then import into Ollama.
# 8B example: MODEL=mlx-community/Qwen3-8B-4bit ADAPTERS=llm/adapters8b FUSED=llm/fused8b OUTNAME=bist-analyst8b-ft NUM_LAYERS=8 LR=5e-5 GRAD_ACC=4 ITERS=300 bash llm/finetune.sh smoke
# Usage (from the repo root, venv active, other heavy apps closed):
#   bash llm/finetune.sh smoke    # 10 iterations: checks that training fits in memory, prints speed
#   bash llm/finetune.sh train    # full training run (ITERS, default 300)
#   bash llm/finetune.sh export   # fuse adapter, import into Ollama as bist-analyst-ft
set -euo pipefail
MODEL="${MODEL:-mlx-community/Qwen3-4B-4bit}"
ITERS="${ITERS:-200}"
ADAPTERS="${ADAPTERS:-llm/adapters}"
FUSED="${FUSED:-llm/fused}"
OUTNAME="${OUTNAME:-bist-analyst-ft}"      # must end with -ft (raw chat-template mode in llm/agent.py)
NUM_LAYERS="${NUM_LAYERS:-16}"
LR="${LR:-1e-4}"
GRAD_ACC="${GRAD_ACC:-1}"                   # >1 averages gradients over several samples (less noisy than batch 1)

common=(--model "$MODEL" --data data/llm_train --batch-size 1 --num-layers "$NUM_LAYERS"
        --max-seq-length 6144 --grad-checkpoint --mask-prompt
        --learning-rate "$LR" --adapter-path "$ADAPTERS")
if [ "$GRAD_ACC" -gt 1 ]; then common+=(--grad-accumulation-steps "$GRAD_ACC"); fi

case "${1:-}" in
  smoke)
    pip install -q mlx-lm
    python -m mlx_lm lora "${common[@]}" --train --iters 10 --steps-per-report 5 --steps-per-eval 10 --val-batches 2 --save-every 10
    ;;
  train)
    python llm/filter_dataset.py --model "$MODEL" --limit "${LIMIT:-5900}"
    python -m mlx_lm lora "${common[@]}" --train --iters "$ITERS" --steps-per-report 10 --steps-per-eval 100 --val-batches 3 --save-every 50
    ;;
  resume)
    python llm/filter_dataset.py --model "$MODEL" --limit "${LIMIT:-5900}"
    python -m mlx_lm lora "${common[@]}" --train --iters "$ITERS" --steps-per-report 10 --steps-per-eval 100 --val-batches 3 --save-every 50 \
      --resume-adapter-file "$ADAPTERS/adapters.safetensors"
    ;;
  export)
    rm -rf "$FUSED"
    python -m mlx_lm fuse --model "$MODEL" --adapter-path "$ADAPTERS" --save-path "$FUSED" --dequantize
    # Reuse the chat template / renderer of the library model, replace only the weights and our own settings
    ollama show qwen3:8b --modelfile | grep -v -E '^(#|FROM|SYSTEM|PARAMETER)' > llm/Modelfile.ft.tmp || true
    {
      echo "FROM ./$FUSED"
      cat llm/Modelfile.ft.tmp
      grep -E '^PARAMETER' llm/Modelfile
      sed -n '/^SYSTEM/,$p' llm/Modelfile
    } > llm/Modelfile.ft
    rm -f llm/Modelfile.ft.tmp
    ollama create "$OUTNAME" -f llm/Modelfile.ft -q int8
    echo "created $OUTNAME"
    ;;
  *)
    echo "usage: bash llm/finetune.sh smoke|train|resume|export"; exit 1
    ;;
esac
