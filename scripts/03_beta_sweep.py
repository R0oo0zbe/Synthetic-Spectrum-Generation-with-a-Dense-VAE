#!/usr/bin/env python3
"""Controlled beta sweep using the exact baseline split and global scale."""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maldi_vae.config import (
    INPUT_DIM, HIDDEN_1, HIDDEN_2, HIDDEN_3, LATENT_DIM, DROPOUT,
    MZ_AXIS, TRAINING_SEED, beta_tag, project_paths,
)
from maldi_vae.data import set_seed, load_vae_arrays, load_saved_split
from maldi_vae.models import DenseBetaVAE
from maldi_vae.training import (
    make_loaders, train_model, evaluate_epoch, reconstruct_indices, encode_all,
)
from maldi_vae.generation import generate_synthetic, save_synthetic_arrays
from maldi_vae.evaluation import (
    reconstruction_metrics_table, latent_statistics_table, population_summary,
)
from maldi_vae.metrics import vector_cosine
from maldi_vae.plotting import (
    plot_training_history, plot_reconstructions, plot_synthetic_examples,
)


BASELINE_BETA = 0.10
BETAS_TO_TRAIN = [0.001, 0.003, 0.01, 0.03]
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 400
KL_WARMUP_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 40
N_SYNTHETIC = 1000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driams-root", default="DRIAMS-B")
    parser.add_argument("--year", default="2018")
    return parser.parse_args()


def make_model(device):
    return DenseBetaVAE(
        input_dim=INPUT_DIM,
        hidden_1=HIDDEN_1,
        hidden_2=HIDDEN_2,
        hidden_3=HIDDEN_3,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
    ).to(device)


def evaluate_and_save(
    beta, model, model_dir, history, checkpoint, source,
    X, y, codes, X_scaled, train_idx, val_idx, test_idx,
    global_scale, device,
):
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = model_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = make_loaders(
        X_scaled, train_idx, val_idx, test_idx, batch_size=BATCH_SIZE
    )
    test_metrics = evaluate_epoch(model, test_loader, beta, device)

    reconstructed = reconstruct_indices(
        model, X_scaled, test_idx, global_scale, device
    )
    np.save(model_dir / "test_reconstructions.npy", reconstructed)

    recon = reconstruction_metrics_table(X, reconstructed, test_idx, codes, y)
    recon.to_csv(model_dir / "test_reconstruction_metrics.csv", index=False)

    latent_mu, latent_logvar = encode_all(model, X_scaled, device)
    latent_stats = latent_statistics_table(latent_mu, latent_logvar)
    latent_stats.to_csv(model_dir / "latent_dimension_statistics.csv", index=False)

    latent_table = pd.DataFrame(
        latent_mu, columns=[f"z_{i}" for i in range(LATENT_DIM)]
    )
    latent_table.insert(0, "label", y)
    latent_table.insert(0, "code", codes.astype(str))
    latent_table.to_csv(model_dir / "latent_means.csv", index=False)

    synthetic = generate_synthetic(
        model,
        N_SYNTHETIC,
        LATENT_DIM,
        global_scale,
        device,
        seed=TRAINING_SEED + 1000,
    )
    save_synthetic_arrays(synthetic, model_dir)

    population = population_summary(X, synthetic)
    pd.DataFrame([population]).to_csv(
        model_dir / "real_vs_synthetic_summary.csv", index=False
    )

    summary = {
        "beta": float(beta),
        "source": source,
        "best_epoch": checkpoint.get("best_epoch", np.nan),
        "best_validation_loss": checkpoint.get("best_validation_loss", np.nan),
        "test_total_loss": float(test_metrics["loss"]),
        "test_reconstruction_loss": float(test_metrics["reconstruction"]),
        "test_kl": float(test_metrics["kl"]),
        "median_test_pearson": float(recon["pearson_r"].median()),
        "median_test_cosine": float(recon["cosine_similarity"].median()),
        "median_test_mae": float(recon["mae"].median()),
        "median_test_rmse": float(recon["rmse"].median()),
        "mean_latent_mu_std": float(latent_stats["mu_std"].mean()),
        "max_latent_mu_std": float(latent_stats["mu_std"].max()),
        **{k: population[k] for k in [
            "synthetic_global_mean", "synthetic_global_std",
            "synthetic_global_min", "synthetic_global_max",
            "synthetic_mean_spectrum_sum", "synthetic_std_spectrum_sum",
            "synthetic_mean_peak_max", "synthetic_std_peak_max",
            "mean_spectrum_pearson", "mean_spectrum_cosine",
        ]},
        "n_train": len(train_idx),
        "n_validation": len(val_idx),
        "n_test": len(test_idx),
        "n_synthetic": N_SYNTHETIC,
        "latent_dim": LATENT_DIM,
        "global_scale": global_scale,
        "seed": TRAINING_SEED,
    }
    pd.DataFrame([summary]).to_csv(model_dir / "model_summary.csv", index=False)

    if history is not None and len(history):
        plot_training_history(history, plot_dir, title_prefix=f"beta={beta:g}")
    plot_reconstructions(X, reconstructed, test_idx, codes, MZ_AXIS, plot_dir, n=3)
    plot_synthetic_examples(synthetic, MZ_AXIS, plot_dir, n=3)

    return summary


def main():
    args = parse_args()
    paths = project_paths(Path(args.driams_root), args.year)
    paths["sweep"].mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    X, y, codes = load_vae_arrays(paths["vae_data"], dtype=np.float32)
    _, train_idx, val_idx, test_idx = load_saved_split(
        paths["baseline"] / "split_assignments.csv", codes
    )

    with open(paths["baseline"] / "config.json", "r", encoding="utf-8") as handle:
        baseline_config = json.load(handle)

    global_scale = float(baseline_config["global_scale"])
    X_scaled = (X / global_scale).astype(np.float32, copy=False)

    all_summaries = []

    for beta in BETAS_TO_TRAIN:
        print(f"\n=== Training beta={beta:g} ===")
        set_seed(TRAINING_SEED)
        train_loader, val_loader, _ = make_loaders(
            X_scaled, train_idx, val_idx, test_idx, batch_size=BATCH_SIZE
        )
        model = make_model(device)
        model_dir = paths["sweep"] / beta_tag(beta)
        checkpoint_path = model_dir / "best_model.pt"

        history, checkpoint = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            target_beta=beta,
            device=device,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata={
                "input_dim": INPUT_DIM,
                "hidden_1": HIDDEN_1,
                "hidden_2": HIDDEN_2,
                "hidden_3": HIDDEN_3,
                "latent_dim": LATENT_DIM,
                "dropout": DROPOUT,
                "global_scale": global_scale,
                "seed": TRAINING_SEED,
            },
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            max_epochs=MAX_EPOCHS,
            warmup_epochs=KL_WARMUP_EPOCHS,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
        )
        history.to_csv(model_dir / "training_history.csv", index=False)

        with open(model_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "beta": beta,
                    "global_scale": global_scale,
                    "seed": TRAINING_SEED,
                    "same_split_as_baseline": True,
                    "baseline_split_path": str(paths["baseline"] / "split_assignments.csv"),
                },
                handle,
                indent=4,
            )

        all_summaries.append(
            evaluate_and_save(
                beta, model, model_dir, history, checkpoint, "new_training",
                X, y, codes, X_scaled, train_idx, val_idx, test_idx,
                global_scale, device,
            )
        )

    # Reuse beta=0.10 baseline, without retraining.
    print("\n=== Reusing beta=0.10 baseline (NO RETRAINING) ===")
    baseline_checkpoint_path = paths["baseline"] / "best_dense_beta_vae.pt"
    try:
        baseline_checkpoint = torch.load(
            baseline_checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        baseline_checkpoint = torch.load(baseline_checkpoint_path, map_location=device)

    baseline_model = make_model(device)
    baseline_model.load_state_dict(baseline_checkpoint["model_state_dict"])
    baseline_model.eval()

    history_path = paths["baseline"] / "training_history.csv"
    baseline_history = pd.read_csv(history_path) if history_path.exists() else None

    all_summaries.append(
        evaluate_and_save(
            BASELINE_BETA,
            baseline_model,
            paths["sweep"] / beta_tag(BASELINE_BETA),
            baseline_history,
            baseline_checkpoint,
            "existing_baseline_no_retraining",
            X, y, codes, X_scaled, train_idx, val_idx, test_idx,
            global_scale, device,
        )
    )

    comparison = pd.DataFrame(all_summaries).sort_values("beta").reset_index(drop=True)
    comparison.to_csv(paths["sweep"] / "beta_model_comparison.csv", index=False)
    comparison.sort_values(
        ["median_test_pearson", "test_kl"], ascending=[False, False]
    ).to_csv(paths["sweep"] / "beta_model_comparison_sorted.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(comparison["beta"], comparison["median_test_pearson"], marker="o")
    plt.xscale("log")
    plt.xlabel("beta")
    plt.ylabel("Median test reconstruction Pearson r")
    plt.title("Reconstruction quality across beta")
    plt.tight_layout()
    plt.savefig(paths["sweep"] / "comparison_pearson_vs_beta.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(comparison["beta"], comparison["test_kl"], marker="o")
    plt.xscale("log")
    plt.xlabel("beta")
    plt.ylabel("Test KL divergence")
    plt.title("Latent regularization across beta")
    plt.tight_layout()
    plt.savefig(paths["sweep"] / "comparison_kl_vs_beta.png", dpi=300)
    plt.close()

    print("\nBETA SWEEP COMPLETE")
    print(
        comparison[
            [
                "beta", "source", "best_epoch", "test_reconstruction_loss",
                "test_kl", "median_test_pearson", "median_test_cosine",
                "mean_latent_mu_std", "mean_spectrum_pearson",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
