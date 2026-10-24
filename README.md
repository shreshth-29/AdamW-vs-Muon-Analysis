# AdamW vs Muon: Diagnosing Optimization during Fine-Tuning

DAPLab recruitment task (post-training optimization). Fine-tuning a small LM on SST-2 with AdamW and with Muon, and diagnosing how the two optimizers differ — not just in accuracy, but in the trajectory they take and the solution they land in.

**Report:** [`AdamW_vs_Muon_report.pdf`](AdamW_vs_Muon_report.pdf) (one page: setup, metrics, results table, and answers to the four task questions).

## Results

SmolLM2-135M fine-tuned on SST-2 for one epoch (batch size 64), evaluated on the SST-2 validation split. The main setting compares AdamW at learning rate 2e-5 with Muon at 1e-3, each over 3 seeds. Numbers are the saved outputs of `final_analysis.ipynb`.

| optimizer | best val acc | final val acc | final val loss | weight drift ‖W−W0‖/‖W0‖ | Hessian λmax at end |
|---|---|---|---|---|---|
| AdamW 2e-5 | 0.935 ± 0.002 | 0.929 ± 0.002 | 0.204 | 0.0035 | 253 ± 20 |
| Muon 1e-3 | 0.930 ± 0.005 | 0.920 ± 0.004 | 0.205 | 0.0337 | 172 ± 33 |

The two optimizers reach similar accuracy and loss, but they end in different places:
- **Drift.** Muon moves the weights about 10 times further from the pretrained values.
- **Flatness.** Muon lands in a flatter region (lower λmax).
- **Update rank.** Muon's one-step updates have a mean stable rank of about 215, against about 4 for AdamW.

Across the ablations (`final_analysis.ipynb`, all 17 runs):
- The best single validation accuracy is 0.937 (AdamW 2e-5 with weight decay 0.01).
- Muon at 3e-3 drops to a final accuracy of 0.890.
- Muon without momentum ends at a validation loss of 0.337.

![Figure 1 of the report: loss curves, gradient norms, per-block update size, drift and rank, and Hessian lambda_max for AdamW 2e-5 and Muon 1e-3](fig1.png)


## Files

- `train_multi_config.ipynb` — config-driven training. Each entry in `configs` is one run; saves `ckpt/<run>/logs.json` and weights at various steps.
- `final_analysis.ipynb` — computes the metrics from the checkpoints, caches them in `stats.json`, and makes the report figure and tables.
- `AdamW_vs_Muon_report.pdf` — the write-up.
- `fig1.png`: Figure 1 of the report, exported from the saved output of `final_analysis.ipynb`.


## Running

```
pip install torch>=2.9 transformers datasets matplotlib tqdm
```

Run all cells of `train_multi_config.ipynb` (it skips any run whose `logs.json` already exists), then `final_analysis.ipynb`. Works on both CUDA or Apple MPS device. The results in the report were produced on an MPS Device (local training on M5 MacBook).




