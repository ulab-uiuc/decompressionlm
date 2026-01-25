#!/usr/bin/env python3
"""
Qwen2.5-7B-Instruct quantization ladder across multiple prompts/tasks.

Quantization ladder (same order as before):
  1) BF16 baseline
  2) GPTQ Int8
  3) AWQ 4-bit
  4) GPTQ Int4
  5) bnb 4-bit (Unsloth)

Tasks included:
  - us_law
  - git_ver_control
  - radiology_imaging
  - faa

Each task writes to its own output directory and uses its own save-name stem.
"""

import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from src.bin_entropy import estimate_prefix_mass


# ----------------------------
# Quantization variants
# ----------------------------
VARIANTS = [
    # ==========================================
    # 1. QWEN-2.5-7B-INSTRUCT
    # ==========================================
    {
        "label": "Qwen2.5-7B-Instruct__BF16_BASE",
        "repo": "Qwen/Qwen2.5-7B-Instruct",
        "dtype": torch.bfloat16,
    },
    {
        "label": "Qwen2.5-7B-Instruct__GPTQ_INT8",
        "repo": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
        "dtype": "auto",
    },
    {
        "label": "Qwen2.5-7B-Instruct__AWQ_4BIT",
        "repo": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "dtype": "auto",
    },
    {
        "label": "Qwen2.5-7B-Instruct__GPTQ_INT4",
        "repo": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        "dtype": "auto",
    },
    {
        "label": "Qwen2.5-7B-Instruct__BNB_4BIT_UNSLOTH",
        "repo": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "dtype": "auto",
    },
    # ==========================================
    # 2. LLAMA-3.1-8B-INSTRUCT
    # ==========================================
    {
        "label": "Llama-3.1-8B-Instruct__BF16_BASE",
        "repo": "meta-llama/Llama-3.1-8B-Instruct",
        "dtype": torch.bfloat16,
    },
    {
        "label": "Llama-3.1-8B-Instruct__GPTQ_INT8",
        "repo": "abdo-Mansour/Meta-Llama-3.1-8B-Instruct-GPTQ-8bit",
        "dtype": "auto",
    },
    {
        "label": "Llama-3.1-8B-Instruct__AWQ_4BIT",
        "repo": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "dtype": "auto",
    },
    {
        "label": "Llama-3.1-8B-Instruct__GPTQ_INT4",
        "repo": "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
        "dtype": "auto",
    },
    {
        "label": "Llama-3.1-8B-Instruct__BNB_4BIT_UNSLOTH",
        "repo": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "dtype": "auto",
    },
]


# ----------------------------
# Tasks / prompts
# ----------------------------
TASKS = [
    {
        "task_id": "us_law",
        "out_dir": "results/quant_ladder__us_law_16_newlines",
        "save_stem": "__us_law_bar_meta",
        "prompt": """Generate United States bar exam legal concepts as keywords.

Please output ONE concept per line.
Each concept can be multiple words if needed.
Do not include explanations or extra text.
Please begin from any random concept.
Please use English.
""",
    },
    {
        "task_id": "git_ver_control",
        "out_dir": "results/quant_ladder__git_ver_control_16_newlines",
        "save_stem": "__git_ver_control_bar_meta",
        "prompt": """Generate Git and version control concepts as keywords.

Please output ONE concept per line.
Each concept can be multiple words if needed.
Do not include explanations or extra text.
Please begin from any random concept.
Please use English.
""",
    },
    {
        "task_id": "radiology_imaging",
        "out_dir": "results/quant_ladder__radiology_imaging_16_newlines",
        "save_stem": "__radiology_imaging_bar_meta",
        "prompt": """Generate radiology and medical imaging concepts as keywords.

Please output ONE concept per line.
Each concept can be multiple words if needed.
Do not include explanations or extra text.
Please begin from any random concept.
Please use English.
""",
    },
    {
        "task_id": "faa",
        "out_dir": "results/quant_ladder__faa_16_newlines",
        "save_stem": "__faa_bar_meta",
        "prompt": """Generate FAA aviation flight rules and airspace concepts as keywords.

Please output ONE concept per line.
Each concept can be multiple words if needed.
Do not include explanations or extra text.
Please begin from any random concept.
Please use English.
""",
    },
]


def load_tokenizer(repo_id: str):
    tok = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def run_one(task: dict, variant: dict, *, prefix_len: int, max_len: int, prob_threshold: float,
            max_samples: int, batch_size: int, display_interval: int):
    label = variant["label"]
    repo = variant["repo"]
    dtype = variant["dtype"]

    os.makedirs(task["out_dir"], exist_ok=True)

    save_name = (
        label
        + task["save_stem"]
        + f"__L{prefix_len}"
        + f"__T{prob_threshold}"
        + ".delm.parquet"
    )
    save_path = os.path.join(task["out_dir"], save_name)

    print(f"\n{'='*96}")
    print(f"Task  : {task['task_id']}")
    print(f"Label : {label}")
    print(f"Repo  : {repo}")
    print(f"dtype : {dtype}")
    print(f"Out   : {save_path}")
    print(f"{'='*96}")

    model = None
    tokenizer = None

    try:
        # Cache hit → tokenizer only
        if os.path.exists(save_path):
            print(f"Found existing results at {save_path}")
            print("Skipping model load, tokenizer only...")

            tokenizer = load_tokenizer(repo)

            results = estimate_prefix_mass(
                model=None,
                tokenizer=tokenizer,
                prefix=task["prompt"],
                prefix_len=prefix_len,
                prob_threshold=prob_threshold,
                max_samples=max_samples,
                max_len=max_len,
                use_chat_template=True,
                batch_size=batch_size,
                display_interval=display_interval,
                save_path=save_path,
                model_name=label,
                enable_graph_analysis=True,
            )
        else:
            print("Loading model and tokenizer...")
            tokenizer = load_tokenizer(repo)
            
            if repo == "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4":
                config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
                qc = getattr(config, "quantization_config", None)
                qc["desc_act"] = False
                config.quantization_config = qc
                model = AutoModelForCausalLM.from_pretrained(
                    repo,
                    config=config,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    repo,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )

            results = estimate_prefix_mass(
                model=model,
                tokenizer=tokenizer,
                prefix=task["prompt"],
                prefix_len=prefix_len,
                prob_threshold=prob_threshold,
                max_samples=max_samples,
                max_len=max_len,
                use_chat_template=True,
                batch_size=batch_size,
                display_interval=display_interval,
                save_path=save_path,
                model_name=label,
                enable_graph_analysis=True,
            )

        print(f"\nResults for {task['task_id']} :: {label}")
        print(f"Unique prefixes    : {results['unique_prefixes']}")
        print(f"Samples done       : {results['samples_done']}")
        print(f"Effective set size : {results['effective_set_size']}")
        print(f"Avg tokens/seq     : {results['effective_set_stats']['avg_tokens']:.1f}")
        print(f"Total tokens       : {results['effective_set_stats']['total_tokens']}")
        print(f"Saved to           : {save_path}")

    except Exception as e:
        print(f"\n⚠️  Error processing {task['task_id']} :: {label}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print(f"\nCleaning up memory for {task['task_id']} :: {label}...")

        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        print("✓ Memory cleaned\n")


def main():
    # Experiment parameters (unchanged)
    max_len = 16
    prefix_len = 16
    prob_threshold = 1.0  # disable
    max_samples = 8192
    batch_size = 256
    display_interval = batch_size

    # Run tasks x variants
    for task in TASKS:
        for variant in VARIANTS:
            run_one(
                task,
                variant,
                prefix_len=prefix_len,
                max_len=max_len,
                prob_threshold=prob_threshold,
                max_samples=max_samples,
                batch_size=batch_size,
                display_interval=display_interval,
            )


if __name__ == "__main__":
    main()
