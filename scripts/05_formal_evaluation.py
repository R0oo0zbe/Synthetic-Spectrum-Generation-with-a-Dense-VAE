#!/usr/bin/env python3
"""Formal held-out evaluation of the frozen beta=0.01 Dense beta-VAE.

Implements the supervisor-requested:
- held-out test evaluation
- PCA fitted on real training spectra only
- PCA Fréchet Distance (not image FID)
- precision, recall, density, coverage
- real-vs-real baseline
- matched sample sizes
- 10 evaluation seeds
- TIC normalization
- mean ± SD summaries and plots
"""

from pathlib import Path
import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maldi_vae.config import (
    INPUT_DIM, HIDDEN_1, HIDDEN_2, HIDDEN_3, LATENT_DIM, DROPOUT,
    PCA_DIM, PRDC_K, FORMAL_EVALUATION_SEEDS, beta_tag, project_paths,
)
from maldi_vae.data import (
    set_seed, load_vae_arrays, load_saved_split, tic_normalize,
)
from maldi_vae.models import DenseBetaVAE
from maldi_vae.generation import generate_synthetic
from maldi_vae.metrics import (
    pca_frechet_distance,
    precision_recall_density_coverage,
    real_real_prdc_baseline,
)
from maldi_vae.evaluation import summary_across_seeds
from maldi_vae.plotting import (
    plot_formal_fd, plot_prdc_comparison, plot_pca,
)


BETA = 0.01
EXPECTED_TEST_SIZE = 22


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driams-root", default="DRIAMS-B")
    parser.add_argument("--year", default="2018")
    return parser.parse_args()


def mean_std(table, column):
    values = table[column].to_numpy(dtype=np.float64)
    return float(np.mean(values)), float(np.std(values, ddof=1))


def main():
    args = parse_args()
    paths = project_paths(Path(args.driams_root), args.year)
    beta_dir = paths["sweep"] / beta_tag(BETA)
    output_dir = beta_dir / "formal_distribution_evaluation"
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    X, y, codes = load_vae_arrays(paths["vae_data"], dtype=np.float64)
    split_table, train_idx, val_idx, test_idx = load_saved_split(
        paths["baseline"] / "split_assignments.csv", codes
    )

    if len(test_idx) != EXPECTED_TEST_SIZE:
        warnings.warn(
            f"Expected {EXPECTED_TEST_SIZE} test spectra, found {len(test_idx)}."
        )

    # Put real and synthetic data in the same TIC-normalized representation.
    X_tic = tic_normalize(X, "real X")
    X_train = X_tic[train_idx]
    X_test = X_tic[test_idx]

    checkpoint_path = beta_dir / "best_model.pt"
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    checkpoint_beta = float(checkpoint.get("beta", BETA))
    if not np.isclose(checkpoint_beta, BETA):
        raise ValueError(
            f"Expected beta={BETA}, checkpoint contains beta={checkpoint_beta}."
        )

    model = DenseBetaVAE(
        input_dim=int(checkpoint.get("input_dim", INPUT_DIM)),
        hidden_1=int(checkpoint.get("hidden_1", HIDDEN_1)),
        hidden_2=int(checkpoint.get("hidden_2", HIDDEN_2)),
        hidden_3=int(checkpoint.get("hidden_3", HIDDEN_3)),
        latent_dim=int(checkpoint.get("latent_dim", LATENT_DIM)),
        dropout=float(checkpoint.get("dropout", DROPOUT)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    global_scale = float(checkpoint["global_scale"])
    latent_dim = int(checkpoint.get("latent_dim", LATENT_DIM))

    # PCA is fit ONLY on real training spectra.
    pca = PCA(n_components=PCA_DIM, whiten=False, random_state=0)
    pca.fit(X_train)
    test_features = pca.transform(X_test)

    pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(PCA_DIM)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    ).to_csv(output_dir / "pca_explained_variance.csv", index=False)

    np.savez_compressed(
        output_dir / "pca_model_parameters.npz",
        mean=pca.mean_,
        components=pca.components_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )

    per_seed_rows = []
    example_synthetic_features = None

    for seed in FORMAL_EVALUATION_SEEDS:
        print(f"\nEvaluation seed {seed}")
        set_seed(seed)
        rng = np.random.default_rng(seed)

        synthetic_22 = generate_synthetic(
            model,
            len(X_test),
            latent_dim,
            global_scale,
            device,
            seed=seed,
        )
        synthetic_22 = tic_normalize(synthetic_22, f"synthetic seed {seed}")
        synthetic_features = pca.transform(synthetic_22)

        fd_full = pca_frechet_distance(test_features, synthetic_features)
        prdc_full = precision_recall_density_coverage(
            test_features, synthetic_features, k=PRDC_K
        )

        permutation = rng.permutation(len(X_test))
        midpoint = len(X_test) // 2
        half_a = permutation[:midpoint]
        half_b = permutation[midpoint:]

        if len(half_a) != len(half_b):
            raise ValueError("Held-out test set must be even for the equal-half baseline.")

        real_a = test_features[half_a]
        real_b = test_features[half_b]
        synthetic_11 = synthetic_features[: len(real_a)]

        fd_rr = pca_frechet_distance(real_a, real_b)
        prdc_rr = real_real_prdc_baseline(real_a, real_b, k=PRDC_K)

        fd_matched = pca_frechet_distance(real_a, synthetic_11)
        prdc_matched = precision_recall_density_coverage(
            real_a, synthetic_11, k=PRDC_K
        )

        seed_dir = output_dir / f"seed_{seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.save(seed_dir / "synthetic_22_tic.npy", synthetic_22)
        np.save(seed_dir / "synthetic_11_for_matched_baseline_tic.npy", synthetic_22[:len(real_a)])

        half_a_set = set(half_a.tolist())
        pd.DataFrame(
            {
                "local_test_index": np.arange(len(X_test)),
                "global_index": test_idx,
                "code": codes[test_idx].astype(str),
                "label": y[test_idx],
                "baseline_half": [
                    "A" if i in half_a_set else "B" for i in range(len(X_test))
                ],
            }
        ).to_csv(seed_dir / "real_test_split.csv", index=False)

        per_seed_rows.append(
            {
                "seed": seed,
                "n_real_test_full": len(test_features),
                "n_synthetic_full": len(synthetic_features),
                "pca_fd_real_test_vs_synthetic_22": fd_full["pca_fd"],
                "pca_fd_mean_term_real_test_vs_synthetic_22": fd_full["pca_fd_mean_term"],
                "pca_fd_covariance_term_real_test_vs_synthetic_22": fd_full["pca_fd_covariance_term"],
                "precision_real_test_vs_synthetic_22": prdc_full["precision"],
                "recall_real_test_vs_synthetic_22": prdc_full["recall"],
                "density_real_test_vs_synthetic_22": prdc_full["density"],
                "coverage_real_test_vs_synthetic_22": prdc_full["coverage"],
                "n_real_half_a": len(real_a),
                "n_real_half_b": len(real_b),
                "pca_fd_real_half_a_vs_real_half_b": fd_rr["pca_fd"],
                "precision_real_real_baseline": prdc_rr["precision"],
                "recall_real_real_baseline": prdc_rr["recall"],
                "density_real_real_baseline": prdc_rr["density"],
                "coverage_real_real_baseline": prdc_rr["coverage"],
                "n_real_matched": len(real_a),
                "n_synthetic_matched": len(synthetic_11),
                "pca_fd_real_half_a_vs_synthetic_11": fd_matched["pca_fd"],
                "precision_real_half_a_vs_synthetic_11": prdc_matched["precision"],
                "recall_real_half_a_vs_synthetic_11": prdc_matched["recall"],
                "density_real_half_a_vs_synthetic_11": prdc_matched["density"],
                "coverage_real_half_a_vs_synthetic_11": prdc_matched["coverage"],
            }
        )

        if example_synthetic_features is None:
            example_synthetic_features = synthetic_features.copy()

    metrics_per_seed = pd.DataFrame(per_seed_rows)
    metrics_per_seed.to_csv(
        output_dir / "formal_metrics_per_seed.csv", index=False
    )
    metrics_summary = summary_across_seeds(metrics_per_seed)
    metrics_summary.to_csv(
        output_dir / "formal_metrics_summary.csv", index=False
    )

    rr_fd_mean, rr_fd_std = mean_std(
        metrics_per_seed, "pca_fd_real_half_a_vs_real_half_b"
    )
    rs11_fd_mean, rs11_fd_std = mean_std(
        metrics_per_seed, "pca_fd_real_half_a_vs_synthetic_11"
    )
    rs22_fd_mean, rs22_fd_std = mean_std(
        metrics_per_seed, "pca_fd_real_test_vs_synthetic_22"
    )

    formal_names = ["precision", "recall", "density", "coverage"]
    full_means, full_stds, rr_means, rr_stds = [], [], [], []
    for name in formal_names:
        mean, std = mean_std(metrics_per_seed, f"{name}_real_test_vs_synthetic_22")
        full_means.append(mean)
        full_stds.append(std)
        mean, std = mean_std(metrics_per_seed, f"{name}_real_real_baseline")
        rr_means.append(mean)
        rr_stds.append(std)

    supervisor_summary = pd.DataFrame(
        [
            {"result": "PCA-FD real-real baseline (11 vs 11)", "mean": rr_fd_mean, "std": rr_fd_std},
            {"result": "PCA-FD real-synthetic matched (11 vs 11)", "mean": rs11_fd_mean, "std": rs11_fd_std},
            {"result": "PCA-FD real test-synthetic (22 vs 22)", "mean": rs22_fd_mean, "std": rs22_fd_std},
            *[
                {
                    "result": f"Generative {name} (22 vs 22)",
                    "mean": full_means[i],
                    "std": full_stds[i],
                }
                for i, name in enumerate(formal_names)
            ],
        ]
    )
    supervisor_summary["mean_plus_minus_std"] = supervisor_summary.apply(
        lambda row: f"{row['mean']:.6g} ± {row['std']:.6g}", axis=1
    )
    supervisor_summary.to_csv(output_dir / "SUPERVISOR_SUMMARY.csv", index=False)

    plot_formal_fd(
        rr_fd_mean, rr_fd_std, rs11_fd_mean, rs11_fd_std,
        plot_dir / "pca_fd_real_real_vs_real_synthetic.png",
    )
    plot_prdc_comparison(
        rr_means, rr_stds, full_means, full_stds,
        plot_dir / "prdc_real_real_vs_real_synthetic.png",
    )
    plot_pca(
        test_features,
        example_synthetic_features,
        pca.explained_variance_ratio_,
        plot_dir / "pca_example_seed_00.png",
    )

    with open(output_dir / "evaluation_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "year": args.year,
                "beta": BETA,
                "checkpoint": str(checkpoint_path),
                "model_frozen": True,
                "vae_retrained": False,
                "n_train": len(train_idx),
                "n_validation": len(val_idx),
                "n_test": len(test_idx),
                "pca_fit_data": "real training spectra only",
                "pca_dim": PCA_DIM,
                "formal_frechet_metric_name": "PCA Frechet Distance (PCA-FD)",
                "inception_fid_used": False,
                "prdc_feature_space": f"same {PCA_DIM}-D PCA space",
                "knn_k": PRDC_K,
                "evaluation_seeds": list(FORMAL_EVALUATION_SEEDS),
                "synthetic_per_seed_full": len(X_test),
                "real_real_baseline": "held-out test split into equal halves",
                "tic_normalization": "per spectrum before PCA",
            },
            handle,
            indent=4,
        )

    print("\nFORMAL GENERATIVE EVALUATION COMPLETE")
    print(f"Real vs Real baseline (11 vs 11): {rr_fd_mean:.6g} ± {rr_fd_std:.6g}")
    print(f"Real vs Synthetic matched (11 vs 11): {rs11_fd_mean:.6g} ± {rs11_fd_std:.6g}")
    print(f"Real test vs Synthetic full (22 vs 22): {rs22_fd_mean:.6g} ± {rs22_fd_std:.6g}")
    for name, mean, std in zip(formal_names, full_means, full_stds):
        print(f"{name.capitalize()}: {mean:.6g} ± {std:.6g}")
    print("Output:", output_dir)


if __name__ == "__main__":
    main()
