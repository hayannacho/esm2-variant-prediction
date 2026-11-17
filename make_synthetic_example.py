"""Generate artificial sequence, substitutions, and nonexperimental target scores."""

from pathlib import Path

import numpy as np

from variant_prediction import AMINO_ACIDS, write_csv


def main():
    rng = np.random.default_rng(2025)
    sequence = "".join(rng.choice(list(AMINO_ACIDS), 60))
    labeled, candidates = [], []
    for position in range(0, 60, 3):
        wild = sequence[position]
        alternatives = rng.choice([aa for aa in AMINO_ACIDS if aa != wild], 4, replace=False)
        for index, alternate in enumerate(alternatives):
            mutation = f"{wild}{position}{alternate}"
            if index < 3:
                # Arbitrary numerical rule: not a biological fitness model or DMS measurement.
                score = (AMINO_ACIDS.index(alternate) - AMINO_ACIDS.index(wild)) / 19
                score += 0.2 * np.sin(position)
                labeled.append({"mutant": mutation, "DMS_score": round(float(score), 6)})
            else:
                candidates.append({"mutant": mutation})
    output = Path(__file__).parent / "examples" / "synthetic"
    output.mkdir(parents=True, exist_ok=True)
    (output / "sequence.fasta").write_text(">artificial_sequence_seed2025_not_a_biological_sample\n" + sequence + "\n")
    write_csv(output / "train.csv", ["mutant", "DMS_score"], labeled)
    write_csv(output / "candidates.csv", ["mutant"], candidates)


if __name__ == "__main__":
    main()
