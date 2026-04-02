#!/bin/bash
# Math evaluation script: compare baseline vs fine-tuned on GSM8K Distilled and Calc-GSM8K
# Usage: bash run_eval_math.sh

BASE_MODEL="Qwen/Qwen3.5-9B"
LORA_MODEL="outputs-math-sft-qwen3.5-9b"
BATCH_SIZE=4
MAX_NEW_TOKENS=1024

echo "============================================"
echo "  Math Reasoning Evaluation Pipeline"
echo "============================================"

# 1. Baseline on GSM8K Distilled
echo ""
echo "[1/4] Evaluating baseline on GSM8K Distilled..."
python eval_math_accuracy.py \
    --base_model ${BASE_MODEL} \
    --eval_dataset gsm8k_distilled \
    --batch_size ${BATCH_SIZE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature 0.0 \
    --output_file results/baseline_gsm8k_distilled.jsonl

# 2. Fine-tuned on GSM8K Distilled
echo ""
echo "[2/4] Evaluating fine-tuned model on GSM8K Distilled..."
python eval_math_accuracy.py \
    --base_model ${BASE_MODEL} \
    --lora_model ${LORA_MODEL} \
    --eval_dataset gsm8k_distilled \
    --batch_size ${BATCH_SIZE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature 0.0 \
    --output_file results/finetuned_gsm8k_distilled.jsonl

# 3. Baseline on Calc-GSM8K
echo ""
echo "[3/4] Evaluating baseline on Calc-GSM8K..."
python eval_math_accuracy.py \
    --base_model ${BASE_MODEL} \
    --eval_dataset calc_gsm8k \
    --batch_size ${BATCH_SIZE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature 0.0 \
    --output_file results/baseline_calc_gsm8k.jsonl

# 4. Fine-tuned on Calc-GSM8K
echo ""
echo "[4/4] Evaluating fine-tuned model on Calc-GSM8K..."
python eval_math_accuracy.py \
    --base_model ${BASE_MODEL} \
    --lora_model ${LORA_MODEL} \
    --eval_dataset calc_gsm8k \
    --batch_size ${BATCH_SIZE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature 0.0 \
    --output_file results/finetuned_calc_gsm8k.jsonl

# Print comparison summary
echo ""
echo "============================================"
echo "  Evaluation Complete - Summary"
echo "============================================"
echo ""
echo "Results saved in results/ directory:"
echo "  - results/baseline_gsm8k_distilled_summary.json"
echo "  - results/finetuned_gsm8k_distilled_summary.json"
echo "  - results/baseline_calc_gsm8k_summary.json"
echo "  - results/finetuned_calc_gsm8k_summary.json"
echo ""

# Display summary table if jq is available
if command -v python3 &> /dev/null; then
    python3 -c "
import json, os, glob

print(f'| {\"Model\":<30} | {\"GSM8K Distilled\":<17} | {\"Calc-GSM8K\":<12} |')
print(f'|{\"-\"*32}|{\"-\"*19}|{\"-\"*14}|')

for label, prefix in [('Qwen3.5-9B baseline', 'baseline'), ('Qwen3.5-9B + SFT', 'finetuned')]:
    gsm = calc = 'N/A'
    f1 = f'results/{prefix}_gsm8k_distilled_summary.json'
    f2 = f'results/{prefix}_calc_gsm8k_summary.json'
    if os.path.exists(f1):
        with open(f1) as f: gsm = f'{json.load(f)[\"accuracy\"]:.2f}%'
    if os.path.exists(f2):
        with open(f2) as f: calc = f'{json.load(f)[\"accuracy\"]:.2f}%'
    print(f'| {label:<30} | {gsm:<17} | {calc:<12} |')
"
fi
