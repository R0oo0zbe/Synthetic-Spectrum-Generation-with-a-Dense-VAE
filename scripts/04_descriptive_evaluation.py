#!/usr/bin/env python3
"""Descriptive real-vs-synthetic analysis for the selected beta=0.01 model.

This merges the earlier raw-synthetic analysis and TIC-correction analysis.
No model training occurs here.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maldi_vae.config import MZ_AXIS, TRAINING_SEED, project_paths, beta_tag
from maldi_vae.data import load_vae_arrays, load_saved_split, tic_normalize
from maldi_vae.metrics import (
    per_spectrum_statistics, safe_pearson, vector_cosine,
    pairwise_diversity, pairwise_summary_table, nearest_neighbor_summary,
)
from maldi_vae.plotting import (
    plot_mean_and_std, plot_pairwise_histograms, plot_pca,
)


BETA = 0.01
PCA_COMPONENTS = 10
LOW_INTENSITY_THRESHOLD = 1e-5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driams-root", default="DRIAMS-B")
    parser.add_argument("--year", default="2018")
    return parser.parse_args()


def compare_distribution(real_stats, synthetic_stats, metrics):
    rows = []
    for metric in metrics:
        rv = real_stats[metric].to_numpy(dtype=np.float64)
        sv = synthetic_stats[metric].to_numpy(dtype=np.float64)
        ks = ks_2samp(rv, sv)
        rows.append(
            {
                "metric": metric,
                "real_mean": float(rv.mean()),
                "synthetic_mean": float(sv.mean()),
                "real_std": float(rv.std()),
                "synthetic_std": float(sv.std()),
                "real_median": float(np.median(rv)),
                "synthetic_median": float(np.median(sv)),
                "wasserstein_distance": float(wasserstein_distance(rv, sv)),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    paths = project_paths(Path(args.driams_root), args.year)
    beta_dir = paths["sweep"] / beta_tag(BETA)
    output_dir = beta_dir / "descriptive_evaluation"
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    X, y, codes = load_vae_arrays(paths["vae_data"], dtype=np.float64)
    split_table, train_idx, val_idx, test_idx = load_saved_split(
        paths["baseline"] / "split_assignments.csv", codes
    )
    X_train = X[train_idx]

    synthetic_raw = np.load(beta_dir / "synthetic_spectra.npy").astype(
        np.float64, copy=False
    )
    synthetic_tic = tic_normalize(synthetic_raw, "synthetic beta=0.01")

    np.save(output_dir / "synthetic_spectra_tic.npy", synthetic_tic)
    np.savez_compressed(
        output_dir / "synthetic_spectra_tic.npz",
        spectra=synthetic_tic,
        mz=MZ_AXIS,
    )

    # Per-spectrum statistics before and after TIC.
    real_stats = per_spectrum_statistics(X, "real", LOW_INTENSITY_THRESHOLD)
    raw_stats = per_spectrum_statistics(
        synthetic_raw, "synthetic_raw", LOW_INTENSITY_THRESHOLD
    )
    tic_stats = per_spectrum_statistics(
        synthetic_tic, "synthetic_tic", LOW_INTENSITY_THRESHOLD
    )
    pd.concat([real_stats, raw_stats, tic_stats], ignore_index=True).to_csv(
        output_dir / "per_spectrum_statistics.csv", index=False
    )

    metrics = [
        "total_intensity", "mean_intensity", "std_intensity",
        "max_intensity", "median_intensity", "fraction_exact_zero",
        "fraction_below_low_threshold",
    ]
    compare_distribution(real_stats, raw_stats, metrics).assign(
        comparison="real_vs_synthetic_raw"
    ).to_csv(output_dir / "distribution_raw.csv", index=False)
    compare_distribution(real_stats, tic_stats, metrics).assign(
        comparison="real_vs_synthetic_tic"
    ).to_csv(output_dir / "distribution_tic.csv", index=False)

    # Per-bin population shape.
    real_mean, raw_mean, tic_mean = X.mean(0), synthetic_raw.mean(0), synthetic_tic.mean(0)
    real_std, raw_std, tic_std = X.std(0), synthetic_raw.std(0), synthetic_tic.std(0)

    pd.DataFrame(
        {
            "bin_index": np.arange(X.shape[1]),
            "mz": MZ_AXIS,
            "real_mean": real_mean,
            "raw_synthetic_mean": raw_mean,
            "tic_synthetic_mean": tic_mean,
            "real_std": real_std,
            "raw_synthetic_std": raw_std,
            "tic_synthetic_std": tic_std,
        }
    ).to_csv(output_dir / "per_bin_before_after_tic.csv", index=False)

    # Pairwise diversity with matched real/synthetic sample count.
    rng = np.random.default_rng(TRAINING_SEED)
    n_pairwise = min(len(X), len(synthetic_tic))
    subset = rng.choice(len(synthetic_tic), size=n_pairwise, replace=False)
    pairwise = pairwise_diversity(X, synthetic_tic[subset])
    pairwise_summary_table(pairwise).to_csv(
        output_dir / "pairwise_diversity_summary_tic.csv", index=False
    )
    plot_pairwise_histograms(pairwise, plot_dir)

    # Nearest-neighbor analysis against real training spectra.
    nn_table, nn_arrays = nearest_neighbor_summary(X_train, synthetic_tic)
    nn_table.to_csv(output_dir / "nearest_neighbor_summary_tic.csv", index=False)

    # PCA fitted only on real training spectra.
    n_components = min(PCA_COMPONENTS, len(train_idx), X.shape[1])
    pca = PCA(n_components=n_components, random_state=TRAINING_SEED)
    pca.fit(X_train)
    real_pca = pca.transform(X)
    synthetic_pca = pca.transform(synthetic_tic)

    pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    ).to_csv(output_dir / "pca_explained_variance.csv", index=False)

    real_pca_table = pd.DataFrame(
        {"code": codes.astype(str), "label": y, "split": split_table["split"].to_numpy()}
    )
    synthetic_pca_table = pd.DataFrame({"synthetic_index": np.arange(len(synthetic_tic))})
    for i in range(n_components):
        real_pca_table[f"PC{i+1}"] = real_pca[:, i]
        synthetic_pca_table[f"PC{i+1}"] = synthetic_pca[:, i]
    real_pca_table.to_csv(output_dir / "real_pca_coordinates.csv", index=False)
    synthetic_pca_table.to_csv(
        output_dir / "synthetic_tic_pca_coordinates.csv", index=False
    )

    plot_mean_and_std(X, synthetic_tic, MZ_AXIS, plot_dir, prefix="tic")
    plot_pca(
        real_pca, synthetic_pca, pca.explained_variance_ratio_,
        plot_dir / "pca_real_vs_tic_synthetic.png",
        real_label="Real",
    )

    summary = {
        "beta": BETA,
        "n_real": len(X),
        "n_synthetic": len(synthetic_tic),
        "raw_mean_total_intensity": float(synthetic_raw.sum(axis=1).mean()),
        "tic_mean_total_intensity": float(synthetic_tic.sum(axis=1).mean()),
        "mean_spectrum_pearson_raw": safe_pearson(real_mean, raw_mean),
        "mean_spectrum_pearson_tic": safe_pearson(real_mean, tic_mean),
        "std_spectrum_pearson_raw": safe_pearson(real_std, raw_std),
        "std_spectrum_pearson_tic": safe_pearson(real_std, tic_std),
        "mean_spectrum_cosine_tic": vector_cosine(real_mean, tic_mean),
        "std_spectrum_cosine_tic": vector_cosine(real_std, tic_std),
        "median_real_real_pearson": float(
            pairwise_summary_table(pairwise).loc[
                lambda d: (d["metric"] == "pearson_r") & (d["comparison"] == "real_real"),
                "median",
            ].iloc[0]
        ),
        "median_synthetic_synthetic_pearson": float(
            pairwise_summary_table(pairwise).loc[
                lambda d: (d["metric"] == "pearson_r") & (d["comparison"] == "synthetic_synthetic"),
                "median",
            ].iloc[0]
        ),
    }
    pd.DataFrame([summary]).to_csv(
        output_dir / "descriptive_summary.csv", index=False
    )

    with open(output_dir / "analysis_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "beta": BETA,
                "seed": TRAINING_SEED,
                "pca_components": n_components,
                "pca_fit_data": "real training spectra only",
                "tic_formula": "x_corrected = x_raw / row_sum",
            },
            handle,
            indent=4,
        )

    print("\nDESCRIPTIVE EVALUATION COMPLETE")
    print(pd.DataFrame([summary]).to_string(index=False))
    print("Output:", output_dir)


if __name__ == "__main__":
    main()
