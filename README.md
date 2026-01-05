# DecompressionLM: Quasi-Monte Carlo Sampling and Entropy Estimation of Learned Language Model Distributions

Most LLM evaluations sample outputs using temperature/top-k/nucleus sampling. These methods are stochastic and don't systematically explore the model's learned distribution. By systematically sampling the actual learned distribution $p(\text{text}\mid \text{prompt})$ of language models using [arithmetic coding](https://arxiv.org/abs/2210.15458) and quasi-Monte Carlo, we can enable direct measurement of what models know or memorize. We treat the language model as a compressed probability distribution and estimate its effective support size and/or entropy with respect to a given prompt.

```text
                                        QMC Entropy Convergence                                
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
49.5┤▌                                                                                             │
    │▌                                                                                             │
45.3┤█                                                                                             │
    │█▟▄   ▄▄▄▄▄▄▄▄▄▄▄▄▖▄▄▄▄▄▄▄▄▄ ▗▗ ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▖▄▄▄▖   ▄▄▖   ▗▖ ▗▄▄                ▄▄▄▄▄│
    │▛▀▀▀▀▀▀▘         ▝▀▀▘▝▀ ▝ ▝▀▀▀▀▀▀▘                  ▝▀▘▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▘ ▀▝│
41.1┤▌                                                                                             │
    │▌                                                                                             │
36.8┤▌                                                                                             │
    │▌                                                                                             │
32.6┤▌                                                                                             │
    │▌                                                                                             │
    │▌                                                                                             │
28.4┤▌                                                                                             │
    │▌                                                                                             │
24.2┤▌                                                                                             │
    └┬──────────────────────┬───────────────────────┬──────────────────────┬──────────────────────┬┘
    1.0                  4928.8                  9856.5                 14784.2             19712.0 
Entropy (bits)                                   Sample                                             

[CONVERGED] [19712/131072] H=42.937b | EOS: 0.8% | Var: 0.000494
| Speed: 106.2 samp/s, 3385.3 tok/s | Total: 3m05s
```

## Installation
```bash
pip install -r requirements.txt
```
For limited access models and tokenizers on HuggingFace, please log in to your `huggingface-cli` after requesting them.

## Test cases
You can execute the test cases of each module in `src/` by running the following commands from this directory:
```bash
python -m src.vdc
python -m src.arithmetic
python -m src.entropy
python -m src.decompress
```

## Tools
The sampled sequences are saved in Parquet files as integer token IDs to save disk space (as opposed to saving decoded literal sequences), along with the tokenizer/model name so we can decode them later. To quickly inspect the outputs, replace the path to the output Parquet file in the following command:
```bash
python -m tools.analyze_results results/gas_question.parquet
```

## Run experiments

### 0. Logging

Piping (`|`) the terminal output into a log file is a great idea, but since the plot and status lines are repeatedly cleared and redrawn, you might not see any plot or status until an experiment is completed. You can force unbuffered Python terminal I/O so you can get up-to-date plots and status messages in both your terminal and the log file.

```bash
PYTHONUNBUFFERED=1 python -m src.decompress | tee output.log
# or
python -u -m src.decompress | tee output.log
```

### 1. Main Experiments
In this section we use our method as a measurement to answer: **how much does a model know regarding a given question?** 

Other potential research questions include:
1. Does quantizing a model decrease such "support information" regarding a question?
2. Does instruction-tuning a model increase, decrease, or have no effect on such "support information" regarding a question?
3. Could this be used to explain, diagnose, or substitute certain benchmarks? (By comparing rank correlation, cost to execute, etc.)

### 2. Auxiliary Experiments
Auxiliary experiments are designed to examine and improve the methods and implementations of this project. We compare our method to naive i.i.d. Monte Carlo to verify whether our quasi-Monte Carlo approach converges faster and how close the estimates are numerically. We also apply Cranley-Patterson rotation with randomly sampled offsets 100 times and compute the average and variance to decide if it is okay to just run it once with offset zero (default).

```bash
python -m experiment_aux.00_example
python -m experiment_aux.01_qmc_vs_iid
```