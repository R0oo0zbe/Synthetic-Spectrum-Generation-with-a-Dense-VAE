#!/usr/bin/env python3
"""Train the original Dense beta-VAE baseline at beta=0.10."""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maldi_vae.config import (
    INPUT_DIM, HIDDEN_1, HIDDEN_2, HIDDEN_3, LATENT_DIM, DROPOUT,
    MZ_AXIS, TRAINING_SEED, project_paths,
)
from maldi_vae.data import (
    set_seed, load_vae_arrays, create_stratified_split,
    save_split_assignments, training_global_scale,
)
from maldi_vae.models import DenseBetaVAE
from maldi_vae.training import (
    make_loaders, train_model, evaluate_epoch, reconstruct_indices, encode_all,
)
from maldi_vae.generation import generate_synthetic, save_synthetic_arrays, save_synthetic_txt
from maldi_vae.evaluation import reconstruction_metrics_table, latent_statistics_table
from maldi_vae.plotting import (
    plot_training_history, plot_reconstructions, plot_synthetic_examples,
)


BETA = 0.10
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 400
KL_WARMUP_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 40
TEST_SIZE = 0.10
VALIDATION_SIZE = 0.10
N_SYNTHETIC = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driams-root", default="DRIAMS-B")
    parser.add_argument("--year", default="2018")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = project_paths(Path(args.driams_root), args.year)
    results_dir = paths["baseline"]
    plot_dir = results_dir / "plots"
    synthetic_dir = results_dir / "synthetic_spectra"
    for directory in [results_dir, plot_dir, synthetic_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    set_seed(TRAINING_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    X, y, codes = load_vae_arrays(paths["vae_data"], dtype=np.float32)

    train_idx, val_idx, test_idx = create_stratified_split(
        y,
        seed=TRAINING_SEED,
        test_size=TEST_SIZE,
        validation_size=VALIDATION_SIZE,
    )
    save_split_assignments(
        results_dir / "split_assignments.csv",
        codes, y, train_idx, val_idx, test_idx,
    )

    global_scale = training_global_scale(X, train_idx)
    X_scaled = (X / global_scale).astype(np.float32, copy=False)

    train_loader, val_loader, test_loader = make_loaders(
        X_scaled, train_idx, val_idx, test_idx, batch_size=BATCH_SIZE
    )

    model = DenseBetaVAE(
        input_dim=INPUT_DIM,
        hidden_1=HIDDEN_1,
        hidden_2=HIDDEN_2,
        hidden_3=HIDDEN_3,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
    ).to(device)

    checkpoint_path = results_dir / "best_dense_beta_vae.pt"
    checkpoint_metadata = {
        "input_dim": INPUT_DIM,
        "hidden_1": HIDDEN_1,
        "hidden_2": HIDDEN_2,
        "hidden_3": HIDDEN_3,
        "latent_dim": LATENT_DIM,
        "dropout": DROPOUT,
        "global_scale": global_scale,
        "seed": TRAINING_SEED,
    }

    history, checkpoint = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        target_beta=BETA,
        device=device,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_epochs=MAX_EPOCHS,
        warmup_epochs=KL_WARMUP_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )
    history.to_csv(results_dir / "training_history.csv", index=False)

    test_metrics = evaluate_epoch(model, test_loader, BETA, device)

    reconstructed = reconstruct_indices(
        model, X_scaled, test_idx, global_scale, device
    )
    np.save(results_dir / "test_reconstructions.npy", reconstructed)

    recon_table = reconstruction_metrics_table(
        X, reconstructed, test_idx, codes, y
    )
    recon_table.to_csv(results_dir / "test_reconstruction_metrics.csv", index=False)

    latent_mu, latent_logvar = encode_all(model, X_scaled, device)
    latent_table = pd.DataFrame(
        latent_mu,
        columns=[f"z_{i}" for i in range(LATENT_DIM)],
    )
    latent_table.insert(0, "label", y)
    latent_table.insert(0, "code", codes.astype(str))
    latent_table.to_csv(results_dir / "latent_means.csv", index=False)
    latent_statistics_table(latent_mu, latent_logvar).to_csv(
        results_dir / "latent_dimension_statistics.csv", index=False
    )

    synthetic = generate_synthetic(
        model, N_SYNTHETIC, LATENT_DIM, global_scale, device
    )
    save_synthetic_arrays(synthetic, results_dir)
    save_synthetic_txt(synthetic, synthetic_dir)

    config = {
        "year": args.year,
        "input_dim": INPUT_DIM,
        "hidden_1": HIDDEN_1,
        "hidden_2": HIDDEN_2,
        "hidden_3": HIDDEN_3,
        "latent_dim": LATENT_DIM,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "beta": BETA,
        "kl_warmup_epochs": KL_WARMUP_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
        "global_scale": global_scale,
        "seed": TRAINING_SEED,
        "number_of_samples": len(X),
        "number_of_train_samples": len(train_idx),
        "number_of_validation_samples": len(val_idx),
        "number_of_test_samples": len(test_idx),
        "number_of_synthetic_spectra": N_SYNTHETIC,
    }
    with open(results_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)

    plot_training_history(history, plot_dir)
    plot_reconstructions(X, reconstructed, test_idx, codes, MZ_AXIS, plot_dir, n=5)
    plot_synthetic_examples(synthetic, MZ_AXIS, plot_dir, n=5)

    print("\nBASELINE TRAINING COMPLETE")
    print("Train / validation / test:", len(train_idx), len(val_idx), len(test_idx))
    print("Best epoch:", checkpoint["best_epoch"])
    print("Best validation loss:", checkpoint["best_validation_loss"])
    print("Test loss:", test_metrics)
    print("Median test Pearson:", recon_table["pearson_r"].median())
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
