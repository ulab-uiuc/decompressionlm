## Tools

The sampled sequences are saved in Parquet files (`.delm.parquet`) as integer token IDs to save disk space, along with comprehensive metadata including the model/tokenizer name, convergence info for each bin length, and sampling parameters. This allows any bin length to be reconstructed post-hoc from the raw sequences. To quickly inspect the results and see bin statistics with example sequences:
```bash
python -m tools.read_pq results/creative_question.delm.parquet
```

This will show:
- Convergence status for each tracked bin length
- Final entropy values
- Top most-frequent bins with decoded text
- Example sequences belonging to each bin

## Run experiments

### 0. Logging

Piping (`|`) the terminal output into a log file is a great idea, but since the plot and status lines are repeatedly cleared and redrawn, you might not see any plot or status until an experiment is completed. You can force unbuffered Python terminal I/O so you can get up-to-date plots and status messages in both your terminal and the log file.
```bash
PYTHONUNBUFFERED=1 python -m exp.00_example | tee output.log
# or
python -u -m exp.00_example | tee output.log
```

### 1. Example Experiment

Run the example experiment to estimate bin entropy for factual vs. creative questions:
```bash
python -m exp.00_example
```

This will:
- Sample sequences for two different prompts
- Track bin entropy at multiple lengths (1, 2, 4, 8 tokens)
- Display convergence plots in real-time
- Save results to `results/*.delm.parquet`
- Compare final entropy values to test the hypothesis that creative questions have higher entropy than factual ones