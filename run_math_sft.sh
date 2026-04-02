#!/bin/bash
# Math SFT training script for Qwen3.5-9B on OpenWebMath + NuminaMath-CoT
# Hardware: Single L40S 48GB VRAM
# Usage: bash run_math_sft.sh

CUDA_VISIBLE_DEVICES=0 python supervised_finetuning.py \
    --model_name_or_path Qwen/Qwen3.5-9B \
    --train_file_dir ./data/math_sft \
    --validation_file_dir ./data/math_sft \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --template_name qwen \
    --use_peft True \
    --max_train_samples -1 \
    --max_eval_samples 50 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 20 \
    --eval_steps 500 \
    --eval_strategy steps \
    --save_steps 1000 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --output_dir outputs-math-sft-qwen3.5-9b \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --flash_attn True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --cache_dir ./cache
