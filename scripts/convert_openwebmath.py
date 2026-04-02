# -*- coding: utf-8 -*-
"""
@description:
Convert OpenWebMath dataset to ShareGPT conversation format for SFT training.
Streams the dataset, applies quality filters, and outputs JSONL.
"""

import argparse
import json
import os
import re

import datasets


def has_math_content(text):
    """Check if text contains meaningful math content (formulas, equations, symbols)."""
    math_patterns = [
        r'\$.*?\$',           # inline LaTeX
        r'\\\[.*?\\\]',      # display LaTeX
        r'\\frac\{',         # fractions
        r'\\sqrt',           # square roots
        r'\\sum',            # summations
        r'\\int',            # integrals
        r'\\begin\{',        # LaTeX environments
        r'[=<>]\s*\d',       # equations with numbers
        r'\d+\s*[+\-*/^]\s*\d+',  # arithmetic expressions
    ]
    for pattern in math_patterns:
        if re.search(pattern, text):
            return True
    return False


def find_split_point(text, min_ratio=0.1, max_ratio=0.4):
    """Find a natural split point in text to separate prompt from response.

    Looks for paragraph breaks, sentence endings after equations, or section headers.
    Returns the split index, or None if no good split found.
    """
    min_idx = int(len(text) * min_ratio)
    max_idx = int(len(text) * max_ratio)

    # Strategy 1: Find a paragraph break (double newline)
    for i in range(min_idx, max_idx):
        if text[i:i + 2] == '\n\n':
            return i

    # Strategy 2: Find sentence end (period + space/newline)
    for i in range(min_idx, max_idx):
        if text[i] == '.' and i + 1 < len(text) and text[i + 1] in (' ', '\n'):
            return i + 1

    # Strategy 3: Find any newline
    for i in range(min_idx, max_idx):
        if text[i] == '\n':
            return i

    return None


PROMPT_TEMPLATES = [
    "Please explain the following mathematical content step by step:\n\n{context}",
    "Read the following math text and provide a detailed explanation:\n\n{context}",
    "Explain and elaborate on the following mathematical concepts:\n\n{context}",
    "Continue the following mathematical discussion with detailed reasoning:\n\n{context}",
    "Provide a thorough explanation of the math described below:\n\n{context}",
]


def convert_example(text, idx):
    """Convert a single OpenWebMath text into ShareGPT conversation format.

    Splits the text at a natural breakpoint: the first portion becomes the
    user prompt context, and the full text becomes the assistant response.
    """
    split_idx = find_split_point(text)
    if split_idx is None:
        # No good split found, use first ~20% as context
        split_idx = max(100, int(len(text) * 0.2))

    context = text[:split_idx].strip()
    # Use the full text (including context) as the response for complete coverage
    response = text.strip()

    template = PROMPT_TEMPLATES[idx % len(PROMPT_TEMPLATES)]
    prompt = template.format(context=context)

    return {
        "id": f"openwebmath-{idx}",
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": response}
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Convert OpenWebMath to SFT format")
    parser.add_argument('--dataset_name', default='open-web-math/open-web-math',
                        help='HuggingFace dataset name')
    parser.add_argument('--num_samples', type=int, default=50000,
                        help='Number of samples to extract')
    parser.add_argument('--min_length', type=int, default=500,
                        help='Minimum text length in characters')
    parser.add_argument('--max_length', type=int, default=8000,
                        help='Maximum text length in characters')
    parser.add_argument('--output_dir', default='data/math_sft',
                        help='Output directory')
    parser.add_argument('--output_file', default='openwebmath_sft.jsonl',
                        help='Output filename')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling')
    args = parser.parse_args()

    print(f"Loading {args.dataset_name} in streaming mode...", flush=True)
    dataset = datasets.load_dataset(args.dataset_name, split="train", streaming=True, trust_remote_code=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=10000)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)

    collected = 0
    scanned = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            scanned += 1
            text = example.get('text', '')
            if not text:
                continue

            # Length filter
            if len(text) < args.min_length or len(text) > args.max_length:
                continue

            # Math content filter
            if not has_math_content(text):
                continue

            # Convert to conversation format
            conv = convert_example(text, collected)
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')
            collected += 1

            if collected % 5000 == 0:
                print(f"  Collected {collected}/{args.num_samples} (scanned {scanned})", flush=True)

            if collected >= args.num_samples:
                break

    print(f"\nDone! Collected {collected} samples from {scanned} scanned.")
    print(f"Output saved to {output_path}")


if __name__ == '__main__':
    main()
