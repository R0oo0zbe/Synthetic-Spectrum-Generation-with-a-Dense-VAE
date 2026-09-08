"""Higher-level evaluation tables built from reusable metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import safe_pearson, vector_cosine


def reconstruction_metrics_table(real_X, reconstructed, original_indices, codes, y):
    rows = []
    for local_index, original_index in enumerate(original_indices):
        real = real_X[original_index].astype(np.float64, copy=False)
        pred = reconstructed[local_index].astype(np.float64, copy=False)
        difference = pred - real

        rows.append(
            {
                "index": int(original_index),
                "code": str(codes[original_index]),
                "label": int(y[original_index]),
                "pearson_r": safe_pearson(real, pred),
                "cosine_similarity": vector_cosine(real, pred),
                "mae": float(np.mean(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(difference**2))),
                "real_max": float(real.max()),
                "reconstructed_max": float(pred.max()),
                "real_sum": float(real.sum()),
                "reconstructed_sum": float(pred.sum()),
            }
        )
    return pd.DataFrame(rows)


def latent_statistics_table(latent_mu, latent_logvar):
    rows = []
    for dimension in range(latent_mu.shape[1]):
        mu_values = latent_mu[:, dimension]
        logvar_values = latent_logvar[:, dimension]
        rows.append(
            {
                "latent_dimension": dimension,
                "mu_mean": float(np.mean(mu_values)),
                "mu_std": float(np.std(mu_values)),
                "mu_min": float(np.min(mu_values)),
                "mu_max": float(np.max(mu_values)),
                "mean_logvar": float(np.mean(logvar_values)),
                "mean_posterior_variance": float(np.mean(np.exp(logvar_values))),
            }
        )
    return pd.DataFrame(rows)


def population_summary(real, synthetic):
    real_mean = real.mean(axis=0)
    syn_mean = synthetic.mean(axis=0)

    return {
        "real_global_mean": float(real.mean()),
        "synthetic_global_mean": float(synthetic.mean()),
        "real_global_std": float(real.std()),
        "synthetic_global_std": float(synthetic.std()),
        "real_global_min": float(real.min()),
        "synthetic_global_min": float(synthetic.min()),
        "real_global_max": float(real.max()),
        "synthetic_global_max": float(synthetic.max()),
        "real_mean_spectrum_sum": float(real.sum(axis=1).mean()),
        "synthetic_mean_spectrum_sum": float(synthetic.sum(axis=1).mean()),
        "real_std_spectrum_sum": float(real.sum(axis=1).std()),
        "synthetic_std_spectrum_sum": float(synthetic.sum(axis=1).std()),
        "real_mean_peak_max": float(real.max(axis=1).mean()),
        "synthetic_mean_peak_max": float(synthetic.max(axis=1).mean()),
        "real_std_peak_max": float(real.max(axis=1).std()),
        "synthetic_std_peak_max": float(synthetic.max(axis=1).std()),
        "mean_spectrum_pearson": safe_pearson(real_mean, syn_mean),
        "mean_spectrum_cosine": vector_cosine(real_mean, syn_mean),
    }


def summary_across_seeds(metrics_per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean, sample SD, median, min, max for all numeric columns except seed."""
    rows = []
    for column in metrics_per_seed.columns:
        if column == "seed" or not np.issubdtype(metrics_per_seed[column].dtype, np.number):
            continue

        values = metrics_per_seed[column].to_numpy(dtype=np.float64)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        rows.append(
            {
                "metric": column,
                "n_seeds": len(values),
                "mean": mean,
                "std": std,
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean_plus_minus_std": f"{mean:.6g} ± {std:.6g}",
            }
        )
    return pd.DataFrame(rows)
