# Instructions

Because a dependency mismatch between `autoawq` (support for newer pytorch-pipelines was recently dropped) and `gptqmodel` (actively managed) namely on `transformers`, please `pip -r install requirements.txt` first and `pip install gptqmodel` later. You will see a dependency (`transformers`) mismatch error saying the `transformers` was old but `gptqmodel` is still installed and would work, at least for now.

To run the experiments:
```bash
python -m exp.01_quantization
```

To view one saved sampling file:
```bash
python -m tools.view_results results/qwen2p5_7b_it_quant_ladder__us_law_16_newlines/Qwen2.5-7B-Instruct__AWQ_4BIT__us_law_bar_meta__L16__T1.0.delm.parquet
```

Some of the models might be gated so please remeber to log in your huggingface cli (`hf auth`).

# On logging

The printout is very long so it is a good idea to pipe the stdout into a file. If you piped the output into a text file, it would contain ANSI sequences including plots that will be drawn over. To debloat the log file,

```bash
sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g' exp-01.log > exp-01.clean.txt
```

which can reduce the bloat but the color of the plot will be lost.

To remove ANSI (plot will be gone) and repeated sampling stat log,
```bash
ed -r '/^\x1B\[/d; /^\[SAMPLING\]/,+3d' exp-02.log > exp-02.clean.txt
```