# DecompressionLM: Quasi-Monte Carlo Sampling and Entropy Estimation of Learned Language Model Distributions

## Installation
```bash
pip install -r requirements.txt
```
For access limited models and tokenizers on huggingface, please log in to your `huggingface-cli` after requesting them.

## Test cases

You could execute the test cases of each module in src by executing the following command in the directory same as this README:

```bash
python -m src.vdc
python -m src.arithmetic
python -m src.entropy
python -m src.decompress
```

## Tools

The sampled sequences are saved in parquet files as integer ids to save disk space as opposed to saving the decoded literal sequence, and they are saved with the tokenizer/model name so we can read out the literal sequences. To quicky inspect the outputs, replace the path to the output parquet file in the following command executed in this directory:

```bash
python -m tools.analyze_results results/gas_question.parquet
```
## Run experiments

### 2. Auxiliary Experiments