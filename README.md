# ESM2 variant prediction

Predict scores for single amino-acid substitutions using frozen ESM2 protein embeddings and a trainable neural-network regression head. Export validation metrics, candidate predictions, and a top-10 ranking.

**Runnable synthetic demo:** real ESM2 8M inference on 80 artificial variants, followed by five training epochs, position-held-out evaluation, and ranking of 20 candidates. No experimental protein measurements are included.

| Verified CPU run | Result |
|---|---|
| Backbone | `esm2_t6_8M_UR50D`, frozen; 320-dimensional embeddings |
| Artificial sequence | 60 amino acids |
| Labeled variants | 48 training / 12 validation; disjoint mutation positions |
| Candidate variants | 20 predictions and a top-10 list |
| Validation MSE | 0.232305 (training-mean baseline: 0.240686) |
| Validation Spearman correlation | −0.293706 |
| Automated checks | 12 tests passed |

The negative rank correlation does **not** demonstrate useful predictive ranking. These results verify the workflow on arbitrary synthetic labels; they are not a DMS benchmark, a reproduction of the course leaderboard, or evidence of biological fitness prediction. See [saved outputs](results/synthetic) and [validation details](docs/VALIDATION.md).

## What it does

1. Reads a single wild-type FASTA record, labeled substitutions, and candidate substitutions.
2. Validates **zero-based** coordinates, wild-type residues, target values, duplicates, and train/candidate overlap.
3. Splits labeled data by mutation position so substitutions at one position cannot occur in both training and validation.
4. Uses the final ESM2 layer and mean pooling over amino-acid tokens, excluding special tokens and padding.
5. Trains a `Linear → ReLU → Linear` regression head with Adam and mean-squared error. The ESM2 backbone remains frozen.
6. Writes validation MSE/Spearman, a constant baseline, candidate scores, and a ranked shortlist.

## Run the synthetic example

Use Python 3.11. The first run downloads Meta's pretrained ESM2 weights into PyTorch's cache; weights are not stored in this repository. The 8M model is the default for a small CPU demonstration.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python variant_prediction.py \
  --fasta examples/synthetic/sequence.fasta \
  --train examples/synthetic/train.csv \
  --candidates examples/synthetic/candidates.csv \
  --output outputs/demo

python -m unittest discover -s tests -v
```

The output directory must be new or empty. Use a different output path for another run. CPU execution uses one thread and seed 42 by default. `TORCH_HOME` can redirect the download cache. The tests do not download model weights; the separate command above exercises actual ESM2 inference and training.

Files written:

- `predictions.csv`: every candidate and its `DMS_score_predicted`.
- `validation.csv`: held-out substitutions, target scores, and predictions.
- `metrics.json`: model/configuration, split positions, losses, and validation/baseline metrics. Undefined Spearman correlations are JSON `null`.
- `top10.txt`: highest predicted scores, one substitution per line; `--top-k` changes the number of entries.

## Inputs and synthetic data

`M0A` means replace the **first** residue, methionine, with alanine. Coordinates start at **0**, following the original project convention. Convert one-based mutation identifiers before using this program. Only single substitutions and the 20 standard amino acids are accepted; sequences longer than 1,022 residues are rejected rather than silently truncated.

Labeled CSV:

```csv
mutant,DMS_score
M0A,0.25
C2W,-0.10
```

Candidate CSV needs only a `mutant` column. The illustrative identifiers above require a matching wild-type sequence. Labeled and candidate variants must be distinct, and labeled variants must cover at least two positions.

All files in `examples/synthetic/` are generated from scratch by:

```bash
python make_synthetic_example.py
```

The sequence is random with seed 2025. Target scores use an arbitrary amino-acid alphabet-index difference plus a position-dependent sine term. They are **not measured DMS scores**, anonymized course data, or a mechanistic fitness model. The `DMS_score` column name is retained for compatibility with the original workflow.

## Project context and implementation

**CX 4803 · Spring 2025 · team project.** This repository adapts the ESM2 implementation from the project's `hackathon_esm2.ipynb` into a runnable command-line workflow.

The supplied course example used one-hot amino-acid encoding and linear regression. The project implementation replaced these with frozen, mean-pooled ESM2 embeddings, an embedding cache, and a two-layer neural regression head, with validation and candidate ranking. ESM2 itself is Meta's pretrained model, not a model developed by this project.

The original notebook configured the **650M, 33-layer** ESM2 model, a 256-unit regression head, five epochs, batch size 8, and learning rate 0.0001. This version preserves the regression architecture and training defaults, while adding a small 8M CPU demo, automatic embedding dimensions, batched extraction, input validation, position-based splitting, and reproducible synthetic examples. The original split was random by row; current metrics use the new position-based split.

`--model esm2_t33_650M_UR50D` selects the original backbone configuration. That larger model has **not** been executed in this repository's validation and needs substantially more download space, memory, and runtime. No backbone fine-tuning, LoRA, active-learning query loop, or hyperparameter search is implemented.

## References

- [Meta ESM repository](https://github.com/facebookresearch/esm): pretrained models and the `fair-esm` library; [MIT license](https://github.com/facebookresearch/esm/blob/main/LICENSE).
- [Official ESM2 8M model card](https://huggingface.co/facebook/esm2_t6_8M_UR50D): model configuration and license.

The repository contains the workflow, synthetic inputs, and synthetic run outputs. Course datasets, query responses, credentials, reports, instructional notebooks, and model checkpoints are excluded.
