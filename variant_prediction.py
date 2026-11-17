"""Frozen ESM2 embeddings and a trainable regression head for single substitutions."""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MODELS = ("esm2_t6_8M_UR50D", "esm2_t33_650M_UR50D")


def read_fasta(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not lines or not lines[0].startswith(">") or sum(x.startswith(">") for x in lines) != 1:
        raise ValueError("FASTA must contain exactly one record with a header")
    sequence = "".join(lines[1:]).upper()
    if not sequence or set(sequence) - set(AMINO_ACIDS):
        raise ValueError("Use a nonempty sequence of the 20 standard amino acids")
    if len(sequence) > 1022:
        raise ValueError("This workflow supports at most 1022 residues; it does not truncate")
    return sequence


def mutate(sequence, mutation):
    """Apply a zero-based substitution (M0A changes the first residue)."""
    match = re.fullmatch(r"([ACDEFGHIKLMNPQRSTVWY])(0|[1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])", mutation)
    if not match:
        raise ValueError(f"Invalid zero-based substitution: {mutation!r}")
    wild, position, alternate = match.groups()
    position = int(position)
    if position >= len(sequence) or sequence[position] != wild:
        raise ValueError(f"Wild-type residue or position mismatch: {mutation}")
    if wild == alternate:
        raise ValueError(f"Substitution must change the residue: {mutation}")
    return sequence[:position] + alternate + sequence[position + 1:], position


def read_variants(path, sequence, labeled):
    rows, seen = [], set()
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"mutant", "DMS_score"} if labeled else {"mutant"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns: {sorted(required)}")
        for row in reader:
            mutation = (row.get("mutant") or "").strip()
            mutated, position = mutate(sequence, mutation)
            if mutation in seen:
                raise ValueError(f"Duplicate substitution: {mutation}")
            seen.add(mutation)
            item = {"mutant": mutation, "sequence": mutated, "position": position}
            if labeled:
                try:
                    score = float(row["DMS_score"])
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Invalid DMS_score for {mutation}") from error
                if not math.isfinite(score) or abs(score) > float(np.finfo(np.float32).max):
                    raise ValueError(f"DMS_score must be finite float32: {mutation}")
                item["DMS_score"] = score
            rows.append(item)
    if not rows:
        raise ValueError(f"{path}: no variants")
    return rows


def split_positions(rows, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1")
    positions = np.array(sorted({row["position"] for row in rows}))
    if len(positions) < 2:
        raise ValueError("At least two labeled positions are required")
    shuffled = np.random.default_rng(seed).permutation(positions)
    count = min(len(positions) - 1, max(1, math.ceil(len(positions) * fraction)))
    held_out = set(shuffled[:count].tolist())
    train = [i for i, row in enumerate(rows) if row["position"] not in held_out]
    valid = [i for i, row in enumerate(rows) if row["position"] in held_out]
    return train, valid


def embed_sequences(sequences, model_name, batch_size):
    import esm

    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval()
    model.requires_grad_(False)
    converter = alphabet.get_batch_converter()
    unique = list(dict.fromkeys(sequences))
    cache = {}
    with torch.inference_mode():
        for start in range(0, len(unique), batch_size):
            batch = unique[start:start + batch_size]
            _, _, tokens = converter([(str(i), seq) for i, seq in enumerate(batch)])
            representations = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
            representations = representations["representations"][model.num_layers]
            for index, sequence in enumerate(batch):
                # Exclude BOS, EOS, and padding, including for unequal sequence lengths.
                cache[sequence] = representations[index, 1:len(sequence) + 1].mean(0).clone()
            print(f"Embedded {min(start + batch_size, len(unique))}/{len(unique)} sequences", flush=True)
    return torch.stack([cache[seq] for seq in sequences])


class RegressionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        return self.fc2(self.relu(self.fc1(inputs))).squeeze(-1)


def regression_metrics(targets, predictions):
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    mse = float(np.mean((targets - predictions) ** 2))
    correlation = None
    if len(targets) > 1 and np.ptp(targets) > 0 and np.ptp(predictions) > 0:
        value = float(spearmanr(targets, predictions).statistic)
        if math.isfinite(value):
            correlation = value
    return {"mse": mse, "spearman": correlation}


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    if min(args.epochs, args.batch_size, args.hidden_dim, args.top_k, args.threads) < 1:
        raise ValueError("Epochs, batch size, hidden dimension, top-k, and threads must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    sequence = read_fasta(args.fasta)
    labeled = read_variants(args.train, sequence, labeled=True)
    candidates = read_variants(args.candidates, sequence, labeled=False)
    if {r["mutant"] for r in labeled} & {r["mutant"] for r in candidates}:
        raise ValueError("Labeled and candidate substitutions must not overlap")
    train, valid = split_positions(labeled, args.validation_fraction, args.seed)
    # Fail before downloading weights if an output directory is already populated.
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be a new or empty directory")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    embeddings = embed_sequences([r["sequence"] for r in labeled + candidates], args.model, args.batch_size)
    # Clone outside inference mode so these tensors may enter autograd in the head.
    embeddings = embeddings.clone()
    targets = torch.tensor([r["DMS_score"] for r in labeled], dtype=torch.float32)
    regressor = RegressionHead(embeddings.shape[1], args.hidden_dim)
    optimizer = torch.optim.Adam(regressor.parameters(), lr=args.learning_rate)
    losses = []
    for epoch in range(args.epochs):
        regressor.train()
        order = torch.tensor(train)[torch.randperm(len(train))]
        total = 0.0
        for indices in order.split(args.batch_size):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(regressor(embeddings[indices]), targets[indices])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; check target scale or learning rate")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        losses.append(total / len(train))
        print(f"Epoch {epoch + 1}/{args.epochs}: train MSE={losses[-1]:.6f}", flush=True)
    regressor.eval()
    with torch.no_grad():
        validation_predictions = regressor(embeddings[valid]).numpy()
        candidate_predictions = regressor(embeddings[len(labeled):]).numpy()
    if not np.isfinite(validation_predictions).all() or not np.isfinite(candidate_predictions).all():
        raise ValueError("Model produced non-finite predictions")
    metrics = {
        "model": args.model, "embedding_dimension": embeddings.shape[1],
        "backbone_frozen": True, "device": "cpu", "seed": args.seed,
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "hidden_dimension": args.hidden_dim, "batch_size": args.batch_size,
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "sequence_length": len(sequence), "split": "held-out mutation positions",
        "train_variants": len(train), "validation_variants": len(valid),
        "train_positions": sorted({labeled[i]["position"] for i in train}),
        "validation_positions": sorted({labeled[i]["position"] for i in valid}),
        "candidate_variants": len(candidates), "train_mse_by_epoch": losses,
        "validation": regression_metrics(targets[valid].numpy(), validation_predictions),
        "training_mean_baseline": regression_metrics(targets[valid].numpy(),
                                                     np.full(len(valid), targets[train].mean().item())),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "predictions.csv", ["mutant", "DMS_score_predicted"], [
        {"mutant": row["mutant"], "DMS_score_predicted": float(value)}
        for row, value in zip(candidates, candidate_predictions)
    ])
    write_csv(output / "validation.csv", ["mutant", "position", "DMS_score", "DMS_score_predicted"], [
        {"mutant": labeled[i]["mutant"], "position": labeled[i]["position"],
         "DMS_score": labeled[i]["DMS_score"], "DMS_score_predicted": float(value)}
        for i, value in zip(valid, validation_predictions)
    ])
    ranked = sorted(range(len(candidates)), key=lambda i: (-float(candidate_predictions[i]), candidates[i]["mutant"]))
    (output / "top10.txt").write_text("".join(candidates[i]["mutant"] + "\n" for i in ranked[:args.top_k]))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics["validation"], indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
