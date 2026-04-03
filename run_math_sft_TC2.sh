#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --nodelist=TC2N08
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --job-name=math_sft
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err
#SBATCH --signal=B:USR1@300

nvidia-smi

module load anaconda
eval "$(conda shell.bash hook)"
conda activate env

export PYTHONUNBUFFERED=1
export CUDA_HOME=/apps/cuda_12.8.0
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# 减少显存碎片，缓解 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -z "${HF_TOKEN}" ]; then
    echo "ERROR: HF_TOKEN is not set. Use:"
    echo "  sbatch --export=ALL,HF_TOKEN=\$HF_TOKEN run_math_sft_TC2.sh"
    exit 1
fi
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1

SCRIPT_PATH="$(scontrol show job ${SLURM_JOB_ID} | grep -oP 'Command=\K\S+')"

# 捕获 SLURM 超时前发送的 USR1 信号，自动重新提交任务
resubmit() {
    echo "$(date): 收到超时信号，准备重新提交任务..."
    echo "当前 step 结束后将从 checkpoint 自动恢复"
    ssh CCDS-TC2 "cd /home/msai/lius0131/MathGPT && sbatch --export=ALL,HF_TOKEN=${HF_TOKEN} ${SCRIPT_PATH}"
    echo "$(date): 新任务已提交"
}
trap 'resubmit' USR1

# 自动探测可恢复的 checkpoint，避免把布尔值 True 当作路径传入
RESUME_ARGS=()
if [ -d "outputs-math-sft-qwen3.5-9b" ]; then
    LATEST_CKPT=$(ls -d outputs-math-sft-qwen3.5-9b/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
    if [ -n "${LATEST_CKPT}" ]; then
        echo "发现 checkpoint，恢复训练: ${LATEST_CKPT}"
        RESUME_ARGS=(--resume_from_checkpoint "${LATEST_CKPT}")
    else
        echo "未发现 checkpoint，从头开始训练"
    fi
else
    echo "输出目录不存在，从头开始训练"
fi

# 启动训练（后台运行以便 trap 能捕获信号）
python supervised_finetuning.py \
    --model_name_or_path Qwen/Qwen3.5-9B \
    --train_file_dir ./data/math_sft_v2 \
    --validation_split_percentage 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --dataloader_num_workers 4 \
    --do_train \
    --do_eval \
    --template_name qwen \
    --use_peft True \
    --max_train_samples -1 \
    --max_eval_samples 500 \
    --model_max_length 1024 \
    --num_train_epochs 2 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 20 \
    --eval_steps 400 \
    --eval_strategy steps \
    --save_steps 200 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --output_dir outputs-math-sft-qwen3.5-9b \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --flash_attn True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --cache_dir ./cache \
    "${RESUME_ARGS[@]}" &
TRAIN_PID=$!
wait ${TRAIN_PID}
EXIT_CODE=$?

# 如果训练正常结束（exit code 0），则不再重新提交
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "$(date): 训练已正常完成，无需重新提交"
else
    echo "$(date): 训练异常退出 (exit code: ${EXIT_CODE})"
fi
