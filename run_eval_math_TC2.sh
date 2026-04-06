#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --nodelist=TC2N08
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --job-name=math_eval
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err
#SBATCH --signal=B:USR1@300

nvidia-smi

module load anaconda
eval "$(conda shell.bash hook)"
conda activate env_vllm

export PYTHONUNBUFFERED=1

mkdir -p results

SCRIPT_PATH="$(scontrol show job ${SLURM_JOB_ID} | grep -oP 'Command=\K\S+')"
CURRENT_STEP_FILE="results/.current_step"

# 通过 summary JSON 文件判断某步骤是否已完成
is_step_done() {
    case $1 in
        1) [ -f "results/baseline_gsm8k_distilled_summary.json" ] ;;
        2) [ -f "results/finetuned_gsm8k_distilled_summary.json" ] ;;
        3) [ -f "results/baseline_calc_gsm8k_summary.json" ] ;;
        4) [ -f "results/finetuned_calc_gsm8k_summary.json" ] ;;
        *) return 1 ;;
    esac
}

# 获取 JSONL 文件的行数，用于确定断点续跑的起始样本索引
get_start_index() {
    local file=$1
    if [ -f "$file" ]; then
        wc -l < "$file"
    else
        echo 0
    fi
}

# 捕获 SLURM 超时前 300 秒发送的 USR1 信号，记录当前步骤并重新提交任务
resubmit() {
    local current_step
    current_step=$(cat "${CURRENT_STEP_FILE}" 2>/dev/null || echo "未知")
    local last_done=0
    for i in 1 2 3 4; do
        if is_step_done $i; then
            last_done=$i
        else
            break
        fi
    done
    echo "$(date): 收到超时信号 — 当前第 ${current_step}/4 步，已完成 ${last_done}/4 步"
    echo "$(date): 终止评估进程 (PID: ${EVAL_PID})..."
    kill ${EVAL_PID} 2>/dev/null
    wait ${EVAL_PID} 2>/dev/null
    echo "$(date): 提交续跑任务，下次将从第 $((last_done + 1)) 步继续"
    ssh CCDS-TC2 "cd /home/msai/lius0131/MathGPT && sbatch ${SCRIPT_PATH}"
    echo "$(date): 新任务已提交"
    exit 0
}
trap 'resubmit' USR1

run_eval() {
    BASE_MODEL="Qwen/Qwen3.5-9B"
    LORA_MODEL="outputs-math-sft-qwen3.5-9b"
    BATCH_SIZE=16
    MAX_NEW_TOKENS=1024

    echo "============================================"
    echo "  Math Reasoning Evaluation Pipeline"
    echo "============================================"

    # 1. Baseline on GSM8K Distilled
    if is_step_done 1; then
        echo "[1/4] 已完成，跳过: Baseline on GSM8K Distilled"
    else
        echo 1 > "${CURRENT_STEP_FILE}"
        START_IDX=$(get_start_index "results/baseline_gsm8k_distilled.jsonl")
        echo ""
        echo "[1/4] Evaluating baseline on GSM8K Distilled... (起始推理轮次: ${START_IDX})"
        python eval_math_accuracy.py \
            --base_model ${BASE_MODEL} \
            --eval_dataset gsm8k_distilled \
            --batch_size ${BATCH_SIZE} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --temperature 0.0 \
            --start_index ${START_IDX} \
            --use_vllm \
            --output_file results/baseline_gsm8k_distilled.jsonl
    fi

    # 2. Fine-tuned on GSM8K Distilled
    if is_step_done 2; then
        echo "[2/4] 已完成，跳过: Fine-tuned on GSM8K Distilled"
    else
        echo 2 > "${CURRENT_STEP_FILE}"
        START_IDX=$(get_start_index "results/finetuned_gsm8k_distilled.jsonl")
        echo ""
        echo "[2/4] Evaluating fine-tuned model on GSM8K Distilled... (起始推理轮次: ${START_IDX})"
        python eval_math_accuracy.py \
            --base_model ${BASE_MODEL} \
            --lora_model ${LORA_MODEL} \
            --eval_dataset gsm8k_distilled \
            --batch_size ${BATCH_SIZE} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --temperature 0.0 \
            --start_index ${START_IDX} \
            --use_vllm \
            --output_file results/finetuned_gsm8k_distilled.jsonl
    fi

    # 3. Baseline on Calc-GSM8K
    if is_step_done 3; then
        echo "[3/4] 已完成，跳过: Baseline on Calc-GSM8K"
    else
        echo 3 > "${CURRENT_STEP_FILE}"
        START_IDX=$(get_start_index "results/baseline_calc_gsm8k.jsonl")
        echo ""
        echo "[3/4] Evaluating baseline on Calc-GSM8K... (起始推理轮次: ${START_IDX})"
        python eval_math_accuracy.py \
            --base_model ${BASE_MODEL} \
            --eval_dataset calc_gsm8k \
            --batch_size ${BATCH_SIZE} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --temperature 0.0 \
            --start_index ${START_IDX} \
            --use_vllm \
            --output_file results/baseline_calc_gsm8k.jsonl
    fi

    # 4. Fine-tuned on Calc-GSM8K
    if is_step_done 4; then
        echo "[4/4] 已完成，跳过: Fine-tuned on Calc-GSM8K"
    else
        echo 4 > "${CURRENT_STEP_FILE}"
        START_IDX=$(get_start_index "results/finetuned_calc_gsm8k.jsonl")
        echo ""
        echo "[4/4] Evaluating fine-tuned model on Calc-GSM8K... (起始推理轮次: ${START_IDX})"
        python eval_math_accuracy.py \
            --base_model ${BASE_MODEL} \
            --lora_model ${LORA_MODEL} \
            --eval_dataset calc_gsm8k \
            --batch_size ${BATCH_SIZE} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --temperature 0.0 \
            --start_index ${START_IDX} \
            --use_vllm \
            --output_file results/finetuned_calc_gsm8k.jsonl
    fi

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

    # Display summary table if python3 is available
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
}

run_eval &
EVAL_PID=$!
wait ${EVAL_PID}
EXIT_CODE=$?

# 如果评估正常结束（exit code 0），则清理状态文件，不再重新提交
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "$(date): 全部评估已正常完成"
    rm -f "${CURRENT_STEP_FILE}"
else
    echo "$(date): 评估异常退出 (exit code: ${EXIT_CODE})"
fi
