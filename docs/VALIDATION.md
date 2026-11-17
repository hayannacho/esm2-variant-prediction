# Validation

Verified locally on 2026-11-17 with Python 3.11.15, macOS arm64, PyTorch 2.15.0, NumPy 2.4.6, SciPy 1.17.1, and fair-esm 2.0.0. The portable dependency ranges are in `requirements.txt`; exact versions used for this run are in `requirements-verified.txt`.

## Actual pretrained-model execution

The README command downloaded the official `esm2_t6_8M_UR50D` weights and ran real CPU inference for 80 unique, artificial 60-residue sequences. There was no replacement encoder or mocked model in this run.

Sixty labeled substitutions were split by position into 48 training and 12 validation examples. The regression head trained for five epochs; its reported training MSE changed from 0.195275 to 0.180369. Twenty remaining substitutions received predictions and a top-10 ranking.

| Held-out metric | Value |
|---|---:|
| Model MSE | 0.2323046774 |
| Training-mean baseline MSE | 0.2406856200 |
| Model Spearman correlation | -0.2937062937 |
| Constant baseline Spearman | Undefined (`null`) |

These tiny synthetic results establish successful execution, not useful ranking, biological accuracy, or superiority over a baseline. No real-data benchmark or 650M-backbone run was performed. All substitutions share one artificial sequence background, so the position-held-out split is not a test of generalization to new proteins.

## Automated checks

```bash
python -m unittest discover -s tests -v
```

Twelve tests passed. Ten check executable behavior: zero-based boundary mutations, malformed/mismatched/no-op mutations, FASTA records, valid and invalid CSV input, finite scores, duplicate variants, disjoint and repeatable position splitting, regression-head learning, and metric edge cases. Two check the saved public run outputs: recalculated metrics, held-out positions, candidate coverage, finite predictions, and top-10 ordering.

These tests require the Python dependencies but do not load ESM2 weights. To rerun model inference, use the README's separate end-to-end command with a new output directory. Floating-point results may vary with dependency versions or hardware.
