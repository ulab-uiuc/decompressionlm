# DecompressionLM

Zero-shot concept graph extraction from LLMs using deterministic arithmetic sampling,  
in order to discovers what language models encode [![arXiv](https://img.shields.io/badge/arXiv-2602.00377-b31b1b.svg)](https://arxiv.org/abs/2602.00377)

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.bin_entropy import estimate_prefix_mass

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

results = estimate_prefix_mass(
    model=model,
    tokenizer=tokenizer,
    prefix="Generate US federal aviation law concepts as keywords. ONE concept per line.",
    prefix_len=16,
    prob_threshold=1.0,
    max_samples=8192,
    max_len=32,
    save_path="results/faa_concepts"
)
```

## Core API

`estimate_prefix_mass()` - Main sampling function
- Samples sequences using Van der Corput codes until prefix probability mass threshold is reached
- Extracts concept graphs with deduplication, normalization, and connectivity metrics
- Caches results in `.delm.parquet` format with full metadata

`view_results.py` - Interactive result viewer
```bash
python tools/view_results.py results/faa_concepts.delm.parquet
```

## Experiments

Four experiments reproduce paper findings (see `exp/` directory):

1. `01_quantization_{16,32}.py` - Quantization ladder (BF16→AWQ-4bit→GPTQ-Int4)
2. `02_quantization_offset.py` - VdC offset robustness (8 runs per variant)
3. `03_perplexity_16.py` - Concept understanding via perplexity
4. `04_mmlu.py` - Hallucination verification (22 models ranked by MMLU-Pro Law)

Please execute them using `python -m`, e.g. `python -m exp.01_quantization_16`.  
Some of these experiment might request gated models, so you should log in to your huggingface via `hf auto login` and follow the instructions.

If you piped the output with `tee`, a lot of ANSI characters that will be rewritten over (for plotting) will be saved as well. You can use the `sed` command and regex rules to clean it up.

## Installation

```bash
pip install -r requirements.txt
pip install gptqmodel>=1.0.0
```
\* The reason is `autoawq` being deprecated; it can still work for a while, for now.  
\* This is then resulted in a clashing dependency requirements of `autoawq` which requires older
`transformers`, and `gptqmodel` which requires newer `transformers`. By installing with `pip` we can override that mismatch and it actually will work. Please read the `requirements.txt` for more information.

## Citation

```bibtex
@misc{hong2026decompressionlmdeterministicdiagnosticzeroshot,
      title={DecompressionLM: Deterministic, Diagnostic, and Zero-Shot Concept Graph Extraction from Language Models}, 
      author={Zhaochen Hong and Jiaxuan You},
      year={2026},
      eprint={2602.00377},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.00377}
}
```