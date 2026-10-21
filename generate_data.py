"""Create a clearly labeled toy CSV for interface development only.
Do not use this data to train, evaluate, or report fraud metrics.
"""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_toy_data(output_path="data/demo/toy_transactions.csv", n_samples=1000):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    data = {"Time": np.sort(rng.integers(0, 172800, n_samples))}
    data.update({f"V{i}": rng.normal(size=n_samples) for i in range(1, 29)})
    data["Amount"] = rng.exponential(75.0, n_samples)
    # Labels are randomly assigned for UI wiring only; they have no learnable signal.
    data["Class"] = rng.choice([0, 1], size=n_samples, p=[0.98, 0.02])
    pd.DataFrame(data).to_csv(output, index=False)
    print(f"Toy data saved to {output}. This is not the ULB benchmark and is not for evaluation.")

if __name__ == "__main__":
    generate_toy_data()
