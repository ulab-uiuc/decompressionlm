# DecompressionLM

Zero-shot concept graph extraction from LLMs using deterministic arithmetic sampling,  
in order to discovers what language models encode in the form of concept graphs [![arXiv](https://img.shields.io/badge/arXiv-2602.00377-b31b1b.svg)](https://arxiv.org/abs/2602.00377)

<p align="center">
  <img src="figs/decompression_lm_main_il.svg" width="50%">
</p>

<div style="width:64%; margin: 0 auto; text-align: left;">
  <b>Figure:</b> Diagram of the DecompressionLM pipeline. Given a domain-specific prompt, the language model generates tokens sampled from its output distribution. Multiple parallel sequences yield diverse outputs from which we extract legal concepts (e.g., “Constitutional law”, “Torts”). Concepts are normalized and then merged across sequences to construct a concept graph for analysis.
</div>

## Experiments

To run the experiments:
```bash
python -m exp.01_quantization
```

To view one saved sampling file:
```bash
python -m tools.view_results results/qwen2p5_7b_it_quant_ladder__us_law_16_newlines/Qwen2.5-7B-Instruct__AWQ_4BIT__us_law_bar_meta__L16__T1.0.delm.parquet
```

To reproduce the experiments to reproduce our paper findings, the scripts that you will need are in `exp/` directory:

1. `01_quantization_{16,32}.py` - Quantization ladder (BF16→AWQ-4bit→GPTQ-Int4)
2. `02_quantization_offset.py` - VdC offset robustness (8 runs per variant)
3. `03_perplexity_16.py` - Concept understanding via perplexity
4. `04_mmlu.py` - Hallucination verification (22 models ranked by MMLU-Pro Law)

Each individual experiment will be saved in `results/` directory once it is finished, so it is fine if the script is interrupted in the middle; once you rerun the script, it will skip the finished experiments.

Some of the models might be gated so please remeber to log in your huggingface cli (`hf auth`).

If you piped the output with `tee`, a lot of ANSI characters that will be rewritten over (for plotting) will be saved as well. You can use the `sed` command and regex rules to clean it up.

## On logging

The printout is very long so it is a good idea to pipe the stdout into a file. If you piped the output into a text file, it would contain ANSI sequences including plots that will be drawn over. To debloat the log file,

```bash
sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g' exp-01.log > exp-01.clean.txt
```

which can reduce the bloat but the color of the plot will be lost.

To remove ANSI (plot will be gone) and repeated sampling stat log,
```bash
sed -r '/^\x1B\[/d; /^\[SAMPLING\]/,+3d' exp-02.log > exp-02.clean.txt
```

Alternatively, you can rerun the script after full experimentation to get the final results only.

## Installation

```bash
pip install -r requirements.txt
pip install gptqmodel>=1.0.0
```
\* Because a dependency mismatch between `autoawq` (support for newer pytorch-pipelines was recently dropped) and `gptqmodel` (actively managed) namely on `transformers`, please `pip -r install requirements.txt` first and `pip install gptqmodel` later. You will see a dependency (`transformers`) mismatch error saying the `transformers` was old but `gptqmodel` is still installed and would work, at least for now.

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