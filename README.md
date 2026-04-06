# MathSFT: Enhancing Math Reasoning via Supervised Fine-Tuning

> AI6130 Large Language Model Course Project
>
> 基于 [MedicalGPT](https://github.com/shibing624/MedicalGPT) 框架，对大语言模型进行数学推理能力的有监督微调（SFT）与评估。

---

## 1. 项目概述

本项目以 MedicalGPT 开源训练框架为基础，将其从医疗领域适配到**数学推理**领域，完成以下目标：

1. 使用 **OpenWebMath** + **NuminaMath-CoT** 混合数据集对大模型进行 SFT 微调
2. 在 **GSM8K Distilled** 和 **Calc-GSM8K** 两个数学基准上评估推理准确率
3. 对比原始模型（baseline）与微调后模型的性能差异

### 1.1 研究动机

大语言模型在通用任务上表现出色，但在数学推理方面仍有提升空间。通过在高质量数学语料上进行有监督微调，可以显著增强模型的数学问题求解与逐步推理能力。本项目通过系统化的实验设计，量化分析 SFT 对数学推理性能的影响。

---

## 2. 模型与数据集

### 2.1 基础模型

| 优先级 | 模型 | HuggingFace ID | 参数量 | 对话模板 |
|--------|------|----------------|--------|----------|
| 首选 | Qwen3.5 (9B) | `Qwen/Qwen3.5-9B` | ~10B | `qwen` |
| 备选1 | Qwen2.5 (7B) | `Qwen/Qwen2.5-7B-Instruct` | 7.6B | `qwen` |
| 备选2 | LLaMA 3 (8B) | `meta-llama/Meta-Llama-3-8B-Instruct` | 8B | `llama3` |
| 备选3 | DeepSeek (7B) | `deepseek-ai/deepseek-llm-7b-chat` | 7B | `deepseek` |

**主力模型选择依据**：Qwen3.5-9B 采用混合架构（Gated Delta Networks + MoE），在数学竞赛基准上表现优异（HMMT Feb 2025: 83.2, AIME 2026: 91.3），原生支持 262K 上下文长度。

### 2.2 训练数据集

| 数据集 | 来源 | 数量 | 占比 | 说明 |
|--------|------|------|------|------|
| [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | 结构化数学 QA | 50,000 条 | **76.9%** | 带完整 Chain-of-Thought 推理步骤的数学题，直接对应 GSM8K 解题范式 |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math) | 网页数学文本 | 15,000 条（从 50K 随机采样） | 23.1% | 高质量数学文本，提供领域背景知识 |
| **合计** | | **65,000 条** | 100% | 存放于 `data/math_sft_v2/` |

**混合策略**：以 NuminaMath-CoT（CoT 结构化解题）为主、OpenWebMath（数学背景知识）为辅，比例约 3.3:1，重点强化解题推理能力。训练集与验证集通过 `--validation_split_percentage 2` 自动切分（2% ≈ 1,274 条独立验证样本）。

> **数据长度分布**（基于实际 tokenizer 估算）：P50 ≈ 711 tokens，P90 ≈ 2,143 tokens，无样本超过 4,096 tokens。`model_max_length=2048` 可覆盖约 89% 样本。

### 2.3 评估数据集

| 数据集 | 来源 | 说明 |
|--------|------|------|
| [GSM8K Distilled](https://huggingface.co/datasets/camel-ai/gsm8k_distilled) | 小学数学推理 | 带完整推理步骤和 `\boxed{}` 格式答案的蒸馏版 GSM8K |
| [Calc-GSM8K](https://huggingface.co/datasets/MU-NLPC/Calc-gsm8k) | 增强版 GSM8K | 扩展的 GSM8K 数据集，17K+ 样本 |

---

## 3. 项目结构

```
MathGPT/
├── supervised_finetuning.py          # SFT 训练主程序（框架原有）
├── template.py                       # 对话模板（支持 qwen/llama3/deepseek 等）
├── inference.py                      # 推理脚本
│
├── scripts/
│   └── convert_openwebmath.py        # [新增] OpenWebMath 数据转换脚本
├── eval_math_accuracy.py             # [新增] 数学推理准确率评估脚本
├── run_math_sft.sh                   # [新增] 数学 SFT 训练启动脚本（本地）
├── run_math_sft_TC2.sh               # [新增] TC2 集群 SLURM 训练脚本（含自动重提交）
├── run_eval_math.sh                  # [新增] 评估流水线启动脚本（本地）
├── run_eval_math_TC2.sh              # [新增] TC2 集群 SLURM 评估脚本
│
├── data/
│   ├── math_sft/                     # [旧版] 初始数据目录（50K OpenWebMath）
│   │   ├── openwebmath_sft.jsonl
│   │   └── numina_cot_sft.jsonl
│   └── math_sft_v2/                  # [新增] 当前使用的训练数据目录（65K，调整比例）
│       ├── numina_cot_sft.jsonl      #   NuminaMath-CoT 50K（76.9%）
│       └── openwebmath_sft.jsonl     #   OpenWebMath 随机采样 15K（23.1%）
├── results/                          # [新增] 评估结果目录
│
├── docs/
│   └── numina_cot_sharegpt.py        # NuminaMath-CoT 格式转换（框架原有）
├── validate_jsonl.py                 # JSONL 数据验证工具
├── merge_peft_adapter.py             # LoRA 权重合并工具
├── requirements.txt                  # 项目依赖
└── README_MedicalGPT_original.md     # 原始 MedicalGPT README（已保留）
```

---

## 4. 环境配置

### 4.1 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | 1x 24GB (RTX 3090/4090) | 1x 48GB (L40S/A6000) |
| 内存 | 32GB | 64GB |
| 磁盘 | 100GB 可用空间 | 200GB+ SSD |
| Python | **3.10+**（必须） | 3.10 / 3.11 |
| CUDA | 12.x | 12.8（TC2 集群） |

> 本项目默认配置针对 **单卡 L40S 48GB** 优化。若使用 24GB 显卡，需启用 QLoRA 4bit 量化（在 `run_math_sft.sh` 中添加 `--load_in_4bit True`）。  
> **注意**：`transformers>=5.1.0`、`trl>=0.27.0`、`peft>=0.14.0` 均要求 Python 3.10+，不兼容 Python 3.9。

### 4.2 安装依赖

本项目使用两个独立的 conda 环境，以避免训练与推理依赖冲突：

| 环境 | 用途 | 关键包 |
|------|------|--------|
| `env` | SFT 训练 + 标准推理 | transformers 5.x, peft, trl |
| `env_vllm` | 高速评估推理（可选） | vLLM 0.19.0, transformers 4.x |

**训练环境（`env`）**：

```bash
git clone <repository-url>
cd MathGPT
conda create -n env python=3.10 -y
conda activate env
pip install -r requirements.txt --upgrade
```

核心依赖：
- `transformers >= 5.1.0`
- `peft >= 0.14.0`
- `datasets >= 2.14.6`
- `trl >= 0.27.0`
- `math-verify == 0.5.2`（数学答案验证）
- `latex2sympy2_extended`（LaTeX 解析）

可选依赖：
- `flash-attn`（FlashAttention-2，训练和推理均可加速 **2–3×**，见下）

**flash-attn 安装说明（可选）**：`flash-attn` 不包含在 `requirements.txt` 中，需手动安装。未安装时脚本自动回退到标准 attention，功能不受影响。

```bash
# TC2 集群（CUDA 12.8）
CUDA_HOME=/apps/cuda_12.8.0 \
PATH=/apps/cuda_12.8.0/bin:$PATH \
pip install flash-attn --no-build-isolation --no-cache-dir
```

> `--no-cache-dir` 用于避免 pip 在跨文件系统移动 wheel 时报 `Invalid cross-device link` 错误。安装前可用 `nvcc --version` 确认 CUDA 版本。安装后，`eval_math_accuracy.py` 会在加载模型时自动检测并启用，无需额外参数。

**推理加速环境（`env_vllm`，可选）**：

vLLM 使用 PagedAttention + 连续批处理，推理速度比标准 HF Transformers 快 **5–10×**（9B 模型单卡约 400–800 tokens/s）。由于 vLLM 要求 transformers 4.x，与训练环境不兼容，需单独创建：

```bash
conda create -n env_vllm python=3.10 -y
conda activate env_vllm
pip install vllm==0.19.0 --no-cache-dir
pip install peft loguru datasets math_verify latex2sympy2_extended --no-cache-dir
```

> 安装包较多（约 3–4 GB），建议在磁盘充裕时执行。`--no-cache-dir` 可有效避免配额超限。

---

## 5. 使用指南

### 5.1 Phase 1: 数据准备

目标：构建 `data/math_sft_v2/` 目录，包含 NuminaMath-CoT（50K）和 OpenWebMath（15K）共 65,000 条训练样本。

**Step 1**：转换 NuminaMath-CoT 数据集（取全量 50K）

```bash
python docs/numina_cot_sharegpt.py \
    --train_end 50000 \
    --output_file numina_cot_sft.jsonl \
    --local_dir data/math_sft_v2
```

**Step 2**：转换 OpenWebMath 数据集（随机采样 15K，比例约为 NuminaMath-CoT 的 30%）

```bash
python scripts/convert_openwebmath.py \
    --num_samples 15000 \
    --min_length 500 \
    --max_length 8000 \
    --output_dir data/math_sft_v2
```

> **数据比例说明**：NuminaMath-CoT 含完整 Chain-of-Thought 推理步骤，与 GSM8K 解题范式直接对应，作为主要训练信号（76.9%）。OpenWebMath 提供数学领域背景知识，比例降至 23.1%，避免稀释解题能力。

**Step 3**：验证数据格式

```bash
python validate_jsonl.py --file_path data/math_sft_v2/numina_cot_sft.jsonl
python validate_jsonl.py --file_path data/math_sft_v2/openwebmath_sft.jsonl
```

转换后的数据格式（ShareGPT conversations）：
```json
{
  "conversations": [
    {"from": "human", "value": "Please explain step by step:\n\n...math problem..."},
    {"from": "gpt", "value": "...detailed solution with reasoning..."}
  ]
}
```

### 5.2 Phase 2: SFT 训练

```bash
bash run_math_sft.sh
```

若使用 TC2 的 Slurm 脚本 `run_math_sft_TC2.sh`，请使用 **环境变量注入 token**（不在脚本中明文保存）：

```bash
conda activate env
huggingface-cli login
export HF_TOKEN=<your_hf_token>
sbatch --export=ALL,HF_TOKEN=$HF_TOKEN run_math_sft_TC2.sh
```

说明：
- `run_math_sft_TC2.sh` 会在启动时检查 `HF_TOKEN`，未提供则直接退出
- 脚本内部会自动设置 `HUGGINGFACE_HUB_TOKEN`，并在超时自动重提交流程中继续传递 `HF_TOKEN`
- 建议使用 `huggingface-cli whoami` 验证登录状态

**关键训练参数**（以 `run_math_sft_TC2.sh` 为准）：

| 参数 | 值 | 说明 |
|------|-----|------|
| 基础模型 | `Qwen/Qwen3.5-9B` | 主力模型 |
| 训练数据 | `data/math_sft_v2/`（65K） | NuminaMath-CoT 50K + OpenWebMath 15K |
| LoRA rank | 8 | 轻量适配，节省显存 |
| LoRA alpha | 16 | 2x rank |
| LoRA dropout | 0.05 | 防过拟合 |
| target_modules | all | 自动应用到所有线性层 |
| 学习率 | 1e-4 | LoRA 推荐值 |
| warmup ratio | 0.05 | 前 5% step 线性预热 |
| weight decay | 0.01 | L2 正则 |
| 训练轮数 | 2 | 65K 样本 × 2 epochs |
| 有效 batch size | 64 | 1（per device）× 64（grad accum） |
| 最大序列长度 | 1024 | 覆盖 ~70% 样本；受 44GB GPU + fp32 logits 显存限制 |
| 精度 | bfloat16 | 混合精度训练 |
| FlashAttention-2 | 启用 | 加速注意力计算，降低显存 |
| 梯度检查点 | 启用 | 节省激活值显存 |
| 验证集 | 自动切分 2%（≈1,274 条） | `--validation_split_percentage 2` |

**监控训练进度**：

```bash
tensorboard --logdir outputs-math-sft-qwen3.5-9b
```

### 5.3 Phase 3: 评估对比

**TC2 集群（推荐）**：提交 SLURM 任务，支持断点续跑，默认使用 vLLM 加速：

```bash
sbatch run_eval_math_TC2.sh
```

**本地运行**：

```bash
bash run_eval_math.sh
```

**单独运行某个评估**（标准模式）：

```bash
# Baseline 评估
python eval_math_accuracy.py \
    --base_model Qwen/Qwen3.5-9B \
    --eval_dataset gsm8k_distilled \
    --batch_size 16 \
    --max_new_tokens 1024 \
    --temperature 0.0 \
    --output_file results/baseline_gsm8k_distilled.jsonl

# 微调模型评估
python eval_math_accuracy.py \
    --base_model Qwen/Qwen3.5-9B \
    --lora_model outputs-math-sft-qwen3.5-9b \
    --eval_dataset gsm8k_distilled \
    --batch_size 16 \
    --max_new_tokens 1024 \
    --temperature 0.0 \
    --output_file results/finetuned_gsm8k_distilled.jsonl
```

**启用 vLLM 加速（可选，需 `env_vllm` 环境）**：

在命令末尾添加 `--use_vllm` 即可切换到 vLLM 推理路径，其余参数不变（`--batch_size` 在 vLLM 模式下不生效，由引擎自动管理）：

```bash
conda activate env_vllm

# Baseline 评估（vLLM）
python eval_math_accuracy.py \
    --base_model Qwen/Qwen3.5-9B \
    --eval_dataset gsm8k_distilled \
    --max_new_tokens 1024 \
    --temperature 0.0 \
    --use_vllm \
    --output_file results/baseline_gsm8k_distilled.jsonl

# 微调模型评估（vLLM + LoRA）
python eval_math_accuracy.py \
    --base_model Qwen/Qwen3.5-9B \
    --lora_model outputs-math-sft-qwen3.5-9b \
    --eval_dataset gsm8k_distilled \
    --max_new_tokens 1024 \
    --temperature 0.0 \
    --use_vllm \
    --output_file results/finetuned_gsm8k_distilled.jsonl
```

**断点续跑**：评估中断后，再次提交 SLURM 任务会自动从上次中断的位置继续，无需额外操作。`--start_index N` 可手动指定从第 N 条样本开始（对应已输出的 `.jsonl` 行数）。

| 模式 | 预计耗时（GSM8K 7455 条，单卡） | 适用场景 |
|------|-------------------------------|---------|
| 标准（HF Transformers） | ~19 小时 | 无 vLLM 环境 |
| 标准 + Flash Attention 2 | ~7–10 小时 | 已安装 flash-attn |
| vLLM | ~1–2 小时 | 已配置 `env_vllm` |

### 5.4 Phase 4: 合并权重（可选）

将 LoRA 权重合并到基础模型，便于部署：

```bash
python merge_peft_adapter.py \
    --base_model Qwen/Qwen3.5-9B \
    --lora_model outputs-math-sft-qwen3.5-9b \
    --output_dir merged-math-qwen3.5-9b
```

---

## 6. 答案提取与验证方法

答案验证采用两级策略：`math_verify` 符号比较（优先）→ 正则数值比较（回退）。

### 6.1 `extract_number_regex(text)` — 正则数字提取

从模型输出中按优先级依次匹配（均取**最后一次**匹配，处理千位符 `,`）：

| 优先级 | 策略 | 匹配模式 |
|--------|------|---------|
| 预处理 | 去除推理块 | `<think>...</think>` |
| 1 | LaTeX boxed | `\boxed{N}` |
| 2 | GSM8K 格式 | `#### N` |
| 3 | 自然语言 | `The (final) answer is N` |
| 4 | 兜底 | 文本中最后一个数字 |

### 6.2 `verify_answer_math_verify()` — 符号化验证（优先）

使用 `math_verify` 库进行符号等价判断，支持分数、根号、科学计数法等。

- 去除 `<think>...</think>` 后，用 `LatexExtractionConfig(boxed="all", boxed_match_priority=0)` 提取预测答案
- 调用 `verify(answer_parsed, gold_parsed)` 返回 `1.0`（正确）/ `0.0`（错误）
- 解析失败返回 `None`，触发回退

### 6.3 `verify_answer_regex()` — 数值容差比较（回退）

`math_verify` 失败时使用：调用 `extract_number_regex()` 提取预测值，判断 `|pred - gt| < 1e-3`。

---

## 7. 技术细节

### 7.1 训练方法：LoRA

本项目使用 **LoRA（Low-Rank Adaptation）** 进行参数高效微调：

- 仅训练插入到模型线性层中的低秩矩阵（~1-2% 参数量）
- 大幅降低显存需求和训练时间
- `target_modules=all`：自动检测并应用到所有线性层

### 7.2 数据转换策略

**OpenWebMath**（原始网页文本 -> 对话格式）：
- 流式加载，随机采样
- 过滤条件：文本长度 500-8000 字符 + 包含数学符号/公式
- 在自然断点（段落/句子边界）处拆分为提示-回答对

**NuminaMath-CoT**（已有结构化 QA）：
- 直接使用项目自带的 `docs/numina_cot_sharegpt.py` 转换
- 保留完整的 Chain-of-Thought 推理步骤

### 7.3 框架复用

本项目复用了 MedicalGPT 框架的以下核心组件：

| 组件 | 文件 | 用途 |
|------|------|------|
| SFT 训练器 | `supervised_finetuning.py` | 完整的 LoRA/QLoRA SFT 训练流程 |
| 对话模板 | `template.py` | Qwen/LLaMA/DeepSeek 模板（无需修改） |
| 推理引擎 | `inference.py` | 批量生成和流式推理 |
| 答案验证 | `grpo_training.py` | math_verify 验证逻辑参考 |
| 数据验证 | `validate_jsonl.py` | JSONL 格式检查 |
| 权重合并 | `merge_peft_adapter.py` | LoRA -> 完整模型 |

---

## 8. 快速开始（TL;DR）

```bash
# 0. 安装训练环境
conda create -n env python=3.10 -y && conda activate env
pip install -r requirements.txt --upgrade
# flash-attn 可选，安装后自动启用（2-3x 加速）
CUDA_HOME=/apps/cuda_12.8.0 PATH=/apps/cuda_12.8.0/bin:$PATH \
    pip install flash-attn --no-build-isolation --no-cache-dir

# 0b. 安装推理加速环境（可选，vLLM，评估快 5-10x）
conda create -n env_vllm python=3.10 -y && conda activate env_vllm
pip install vllm==0.19.0 --no-cache-dir
pip install peft loguru datasets math_verify latex2sympy2_extended --no-cache-dir

# 1. 准备数据（~10 min）
conda activate env
# NuminaMath-CoT: 取全部 50K 条
python docs/numina_cot_sharegpt.py --train_end 50000 \
    --output_file numina_cot_sft.jsonl --local_dir data/math_sft_v2
# OpenWebMath: 随机采样 15K 条（约为 NuminaMath-CoT 的 30%）
python scripts/convert_openwebmath.py --num_samples 15000 \
    --output_dir data/math_sft_v2

# 2. 训练（TC2 集群，需提前设置 HF_TOKEN）
export HF_TOKEN=<your_hf_token>
sbatch --export=ALL,HF_TOKEN=$HF_TOKEN run_math_sft_TC2.sh

# 3. 评估（TC2 集群，run_eval_math_TC2.sh 默认使用 env_vllm + vLLM）
sbatch run_eval_math_TC2.sh
```

---

## 9. 参考资料

- [MedicalGPT](https://github.com/shibing624/MedicalGPT) - 基础训练框架
- [OpenWebMath](https://huggingface.co/datasets/open-web-math) - 数学预训练语料
- [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) - 数学 CoT 数据集
- [GSM8K Distilled](https://huggingface.co/datasets/camel-ai/gsm8k_distilled) - 评估基准
- [Calc-GSM8K](https://huggingface.co/datasets/MU-NLPC/Calc-gsm8k) - 评估基准
- [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35) - 主力模型
- [LoRA](https://arxiv.org/abs/2106.09685) - 参数高效微调方法

---

## License

本项目基于 [MedicalGPT](https://github.com/shibing624/MedicalGPT) 开发，遵循 [Apache License 2.0](LICENSE)。
