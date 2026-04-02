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

| 数据集 | 来源 | 数量 | 说明 |
|--------|------|------|------|
| [OpenWebMath](https://huggingface.co/datasets/open-web-math) | 网页数学文本 | 50,000 条（采样） | 14.7B tokens 高质量数学文本，经过长度和数学内容过滤 |
| [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | 结构化数学 QA | 50,000 条（采样） | 带完整 Chain-of-Thought 推理步骤的数学题 |

**混合策略**：OpenWebMath 提供丰富的数学领域知识，NuminaMath-CoT 提供结构化的问题求解范式，两者互补。

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
├── run_math_sft.sh                   # [新增] 数学 SFT 训练启动脚本
├── run_eval_math.sh                  # [新增] 评估流水线启动脚本
│
├── data/
│   └── math_sft/                     # [新增] 数学 SFT 训练数据目录
│       ├── openwebmath_sft.jsonl     #   OpenWebMath 转换后数据
│       └── numina_cot_sft.jsonl      #   NuminaMath-CoT 转换后数据
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

> 本项目默认配置针对 **单卡 L40S 48GB** 优化。若使用 24GB 显卡，需启用 QLoRA 4bit 量化（在 `run_math_sft.sh` 中添加 `--load_in_4bit True`）。

### 4.2 安装依赖

```bash
git clone <repository-url>
cd MathGPT
pip install -r requirements.txt --upgrade
```

核心依赖：
- `transformers >= 5.1.0`
- `peft >= 0.14.0`
- `datasets >= 2.14.6`
- `trl >= 0.27.0`
- `math-verify == 0.5.2`（数学答案验证）
- `latex2sympy2_extended`（LaTeX 解析）

---

## 5. 使用指南

### 5.1 Phase 1: 数据准备

**Step 1**：转换 NuminaMath-CoT 数据集（项目已有脚本）

```bash
python docs/numina_cot_sharegpt.py \
    --train_end 50000 \
    --output_file numina_cot_sft.jsonl \
    --local_dir data/math_sft
```

**Step 2**：转换 OpenWebMath 数据集

```bash
python scripts/convert_openwebmath.py \
    --num_samples 50000 \
    --min_length 500 \
    --max_length 8000 \
    --output_dir data/math_sft
```

**Step 3**：验证数据格式

```bash
python validate_jsonl.py --file_path data/math_sft/openwebmath_sft.jsonl
python validate_jsonl.py --file_path data/math_sft/numina_cot_sft.jsonl
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

**关键训练参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| 基础模型 | `Qwen/Qwen3.5-9B` | 主力模型 |
| LoRA rank | 16 | 适配 9B 模型 |
| LoRA alpha | 32 | 2x rank |
| 学习率 | 1e-4 | LoRA 推荐 |
| 训练轮数 | 3 | 100K 样本 x 3 epochs |
| 有效 batch size | 32 | 4 x 8 gradient accumulation |
| 最大序列长度 | 2048 | 数学文本适用 |
| 精度 | bfloat16 | 混合精度训练 |
| FlashAttention-2 | 启用 | 加速注意力计算 |
| 梯度检查点 | 启用 | 节省显存 |

**监控训练进度**：

```bash
tensorboard --logdir outputs-math-sft-qwen3.5-9b
```

### 5.3 Phase 3: 评估对比

运行完整评估流水线（baseline + fine-tuned，两个数据集）：

```bash
bash run_eval_math.sh
```

或单独运行某个评估：

```bash
# Baseline 评估
python eval_math_accuracy.py \
    --base_model Qwen/Qwen3.5-9B \
    --eval_dataset gsm8k_distilled \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --temperature 0.0 \
    --output_file results/baseline_gsm8k_distilled.jsonl

# 微调模型评估
python eval_math_accuracy.py \
    --base_model Qwen/Qwen3.5-9B \
    --lora_model outputs-math-sft-qwen3.5-9b \
    --eval_dataset gsm8k_distilled \
    --batch_size 4 \
    --max_new_tokens 1024 \
    --temperature 0.0 \
    --output_file results/finetuned_gsm8k_distilled.jsonl
```

### 5.4 Phase 4: 合并权重（可选）

将 LoRA 权重合并到基础模型，便于部署：

```bash
python merge_peft_adapter.py \
    --base_model Qwen/Qwen3.5-9B \
    --lora_model outputs-math-sft-qwen3.5-9b \
    --output_dir merged-math-qwen3.5-9b
```

---

## 6. 评估方法：`eval_math_accuracy.py` 详解

### 6.0 整体调用流程

```
main()
 ├── load_model_and_tokenizer(args)          # 加载模型 + tokenizer（可选 LoRA）
 ├── load_eval_dataset(dataset_name, ...)    # 加载并规范化评估数据集
 └── evaluate(model, tokenizer, samples, args)
      ├── batch_generate(...)                # 批量推理，生成答案文本
      └── 逐样本验证答案
           ├── verify_answer_math_verify()   # 优先：符号化验证（math_verify 库）
           └── verify_answer_regex()         # 回退：正则数值比较
```

---

### 6.1 `extract_number_regex(text)` — 正则数字提取

**功能**：从模型生成文本中提取数值答案，作为 `math_verify` 不可用时的基础工具。

**处理流程**（按优先级顺序）：

| 优先级 | 策略 | 匹配模式 | 示例 |
|--------|------|---------|------|
| 预处理 | 去除推理块 | `<think>...</think>` (DOTALL) | Qwen3.5 思维链标签 |
| 1 | LaTeX boxed | `\boxed{N}` | `\boxed{42}` |
| 2 | GSM8K 格式 | `#### N` | `#### 1,234` |
| 3 | 自然语言 | `The (final) answer is N` | `The answer is $3.14$` |
| 4 | 兜底 | 文本中最后一个数字 | 无结构化答案时 |

- 所有策略均取**最后一次**匹配（应对多步推理），并处理千位分隔符（`,`）
- 无法提取时返回 `None`

---

### 6.2 `verify_answer_math_verify(prediction_text, ground_truth_text)` — 符号化答案验证

**功能**：使用 `math_verify` 库进行符号级别的答案等价判断，支持分数、根号、科学计数法等复杂数学表达式。

**处理步骤**：

1. **解析 ground truth**：
   - 若含 `####`（GSM8K 格式），取 `####` 之后部分
   - 否则用 `first_match` 模式提取第一个 LaTeX 表达式

2. **解析 prediction**：
   - 先去除 `<think>...</think>` 推理块
   - 使用 `LatexExtractionConfig` 配置（关键参数）：
     - `boxed="all"`：识别所有 `\boxed{}` 变体
     - `basic_latex=True`：处理基础 LaTeX
     - `equations=True`：处理等式
     - `units=True`：处理带单位的答案
     - `boxed_match_priority=0`：boxed 答案最高优先级

3. **符号比较**：调用 `verify(answer_parsed, gold_parsed)` 返回 `float`（1.0 正确 / 0.0 错误）

4. **异常处理**：解析/比较失败时返回 `None`，触发回退机制

---

### 6.3 `verify_answer_regex(prediction_text, ground_truth_value)` — 数值容差比较

**功能**：`math_verify` 失败或不可用时的回退验证方案。

```python
|pred - gt| < 1e-3  →  正确 (1.0)
```

- 调用 `extract_number_regex()` 提取预测值
- `ground_truth_value` 为浮点数（由数据集加载时预提取）
- 提取失败或类型转换失败均返回 `0.0`

---

### 6.4 `load_model_and_tokenizer(args)` — 模型加载

**功能**：加载 base model，可选附加 LoRA adapter，支持 4-bit 量化。

**关键配置**：

| 参数 | 值 | 说明 |
|------|----|------|
| `padding_side` | `'left'` | 批量生成必须左填充，避免 EOS 后生成 |
| `device_map` | `"auto"` | 自动分配 GPU/CPU 内存 |
| `torch_dtype` | `"auto"` | 从模型配置自动推断精度 |
| `low_cpu_mem_usage` | `True` | 加速大模型加载 |
| `load_in_4bit` | 可选 | QLoRA 推理，节省约 75% 显存 |

**4-bit 量化配置**（`--load_in_4bit` 开启时）：
```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
```

**LoRA 挂载**：通过 `PeftModel.from_pretrained(model, lora_path)` 在 base model 之上叠加适配器权重，无需合并即可推理。

---

### 6.5 `load_eval_dataset(dataset_name, num_samples=-1)` — 数据集加载

**功能**：加载并规范化评估数据集，统一输出格式。

**输出格式**（每条样本）：
```python
{
    "question":          str,    # 题目文本
    "ground_truth":      str,    # 原始答案字符串（含推理过程）
    "ground_truth_value": float  # 预提取的数值答案（用于 regex 回退）
}
```

**数据集差异**：

| 字段 | `gsm8k_distilled` | `calc_gsm8k` |
|------|-------------------|--------------|
| HuggingFace ID | `camel-ai/gsm8k_distilled` | `MU-NLPC/Calc-gsm8k` |
| split | `train` | `test` |
| question 字段 | `problem` / `question` | `question` |
| ground_truth | `groud_truth_solution`（注意原始拼写错误） | `result` |
| ground_truth_value | 调用 `extract_number_regex()` 从 solution 提取 | 直接读取 `result_float` |

`num_samples=-1` 时加载全部样本；`>0` 时截取前 N 条。

---

### 6.6 `batch_generate(model, tokenizer, questions, ...)` — 批量推理

**功能**：对题目列表进行批量推理，返回生成的答案文本列表。

**关键实现细节**：

- 装饰器 `@torch.inference_mode()` 禁用梯度计算，降低显存占用
- 每道题套用统一 **system prompt**：
  ```
  You are a helpful math assistant. Solve the given math problem step by step.
  Show your reasoning clearly. Put your final numerical answer within \boxed{}.
  ```
- 使用 `tokenizer.apply_chat_template()` 构造完整对话格式（适配各模型模板）
- **答案截取**：仅解码 `outputs[i][prompt_len:]`，去除输入 prompt 部分
- `temperature=0.0` 时自动切换为 greedy decoding（`do_sample=False`）

**参数说明**：

| 参数 | 说明 |
|------|------|
| `batch_size` | 每批处理题数，影响显存占用 |
| `max_new_tokens` | 最大生成 token 数（默认 1024，CoT 推理需充足空间） |
| `temperature` | 采样温度，0.0 为贪心解码（可复现） |

---

### 6.7 `evaluate(model, tokenizer, samples, args)` — 评估主逻辑

**功能**：调度生成与验证，汇总评估结果。

**验证优先级**：

```
if HAS_MATH_VERIFY and ground_truth 非空:
    result = verify_answer_math_verify(prediction, ground_truth)
    if result is not None:  # 符号验证成功
        is_correct = result
        
if is_correct == 0.0 and ground_truth_value 非空:
    is_correct = verify_answer_regex(prediction, ground_truth_value)  # 回退
```

**返回值**：
- `results`：每条样本的详细 dict（question / ground_truth / prediction / correct）
- `accuracy`：百分比准确率（`float`）
- `correct`：正确题数（`int`）
- `total`：总题数（`int`）

---

### 6.8 命令行参数一览

```bash
python eval_math_accuracy.py \
    --base_model      Qwen/Qwen3.5-9B          # 必填：base model 路径或 HF ID
    --lora_model      outputs-math-sft-qwen3.5-9b  # 可选：LoRA adapter 路径（空 = baseline）
    --tokenizer_path  <path>                   # 可选：tokenizer 路径（默认同 base_model）
    --eval_dataset    gsm8k_distilled          # 必填：gsm8k_distilled | calc_gsm8k
    --num_samples     -1                       # 样本数（-1 = 全部）
    --batch_size      4                        # 推理 batch size
    --max_new_tokens  1024                     # 最大生成 token 数
    --temperature     0.0                      # 0.0 = greedy decoding
    --output_file     results/eval.jsonl       # 详细结果输出路径
    --load_in_4bit                             # 启用 4-bit 量化（节省显存）
```

**输出文件**：
- `<output_file>`：逐条 JSONL 详细结果
- `<output_file>.replace('.jsonl', '_summary.json')`：汇总 JSON（含 accuracy 字段，供 `run_eval_math.sh` 读取）

---

### 6.9 评估指标

- **准确率（Accuracy）**：`correct / total × 100%`
- 使用 **greedy decoding**（`temperature=0.0`）确保结果可复现
- 答案验证策略：`math_verify` 符号比较（优先）→ 数值容差比较（`|Δ| < 1e-3`，回退）

### 6.10 预期结果格式

```
| Model                          | GSM8K Distilled | Calc-GSM8K |
|--------------------------------|-----------------|------------|
| Qwen3.5-9B baseline           | xx.x%           | xx.x%      |
| Qwen3.5-9B + SFT (LoRA r=16) | xx.x%           | xx.x%      |
```

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
# 0. 安装依赖
pip install -r requirements.txt --upgrade

# 1. 准备数据（~10 min）
python docs/numina_cot_sharegpt.py --train_end 50000 --output_file numina_cot_sft.jsonl --local_dir data/math_sft
python scripts/convert_openwebmath.py --num_samples 50000 --output_dir data/math_sft

# 2. 训练（~数小时，取决于 GPU）
bash run_math_sft.sh

# 3. 评估
bash run_eval_math.sh
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
