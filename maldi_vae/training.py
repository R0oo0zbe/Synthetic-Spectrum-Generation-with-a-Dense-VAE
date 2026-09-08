"""Shared beta-VAE training and reconstruction utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SpectrumDataset(Dataset):
    def __init__(self, matrix, indices):
        self.matrix = torch.from_numpy(matrix[np.asarray(indices)])
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.matrix[index], int(self.indices[index])


def make_loaders(
    matrix_scaled,
    train_idx,
    val_idx,
    test_idx,
    batch_size=16,
    num_workers=0,
):
    """Construct loaders from the exact saved split."""
    train = SpectrumDataset(matrix_scaled, train_idx)
    val = SpectrumDataset(matrix_scaled, val_idx)
    test = SpectrumDataset(matrix_scaled, test_idx)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def beta_vae_loss(reconstruction, x, mu, logvar, beta):
    """MSE-sum reconstruction + beta * KL, averaged across the batch."""
    reconstruction_loss = (
        F.mse_loss(reconstruction, x, reduction="none").sum(dim=1).mean()
    )
    kl_loss = (
        -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    ).mean()
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss


def beta_for_epoch(target_beta, epoch, warmup_epochs=50):
    if warmup_epochs <= 0:
        return target_beta
    fraction = min(1.0, epoch / warmup_epochs)
    return target_beta * fraction


def train_one_epoch(model, loader, optimizer, beta, device):
    model.train()
    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
    n_samples = 0

    for batch_x, _ in loader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad(set_to_none=True)
        reconstruction, mu, logvar = model(batch_x)
        loss, recon, kl = beta_vae_loss(reconstruction, batch_x, mu, logvar, beta)
        loss.backward()
        optimizer.step()

        n = batch_x.size(0)
        totals["loss"] += loss.item() * n
        totals["reconstruction"] += recon.item() * n
        totals["kl"] += kl.item() * n
        n_samples += n

    return {key: value / n_samples for key, value in totals.items()}


@torch.no_grad()
def evaluate_epoch(model, loader, beta, device):
    model.eval()
    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
    n_samples = 0

    for batch_x, _ in loader:
        batch_x = batch_x.to(device)
        reconstruction, mu, logvar = model(batch_x)
        loss, recon, kl = beta_vae_loss(reconstruction, batch_x, mu, logvar, beta)

        n = batch_x.size(0)
        totals["loss"] += loss.item() * n
        totals["reconstruction"] += recon.item() * n
        totals["kl"] += kl.item() * n
        n_samples += n

    return {key: value / n_samples for key, value in totals.items()}


def train_model(
    model,
    train_loader,
    val_loader,
    target_beta,
    device,
    checkpoint_path: Path,
    checkpoint_metadata: dict,
    learning_rate=1e-3,
    weight_decay=1e-5,
    max_epochs=400,
    warmup_epochs=50,
    early_stopping_patience=40,
    print_every=10,
):
    """Train one beta-VAE and save the best validation checkpoint."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    history_rows = []
    best_validation_loss = np.inf
    best_epoch = None
    epochs_without_improvement = 0

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        current_beta = beta_for_epoch(target_beta, epoch, warmup_epochs)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, current_beta, device
        )
        val_metrics = evaluate_epoch(model, val_loader, current_beta, device)

        history_rows.append(
            {
                "epoch": epoch,
                "beta": current_beta,
                "train_loss": train_metrics["loss"],
                "train_reconstruction": train_metrics["reconstruction"],
                "train_kl": train_metrics["kl"],
                "val_loss": val_metrics["loss"],
                "val_reconstruction": val_metrics["reconstruction"],
                "val_kl": val_metrics["kl"],
            }
        )

        if epoch == 1 or epoch % print_every == 0:
            print(
                f"Epoch {epoch:4d} | beta={current_beta:.6f} | "
                f"train={train_metrics['loss']:.6f} | "
                f"val={val_metrics['loss']:.6f} | "
                f"recon={val_metrics['reconstruction']:.6f} | "
                f"KL={val_metrics['kl']:.6f}"
            )

        if epoch >= warmup_epochs:
            if val_metrics["loss"] < best_validation_loss:
                best_validation_loss = val_metrics["loss"]
                best_epoch = epoch
                epochs_without_improvement = 0

                payload = {
                    "model_state_dict": model.state_dict(),
                    "beta": float(target_beta),
                    "best_epoch": int(best_epoch),
                    "best_validation_loss": float(best_validation_loss),
                    **checkpoint_metadata,
                }
                torch.save(payload, checkpoint_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    history = pd.DataFrame(history_rows)
    if not checkpoint_path.exists():
        raise RuntimeError("No best-model checkpoint was saved.")

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return history, checkpoint


@torch.no_grad()
def reconstruct_indices(model, matrix_scaled, indices, global_scale, device):
    model.eval()
    tensor = torch.from_numpy(matrix_scaled[np.asarray(indices)]).to(device)
    reconstructed = model.reconstruct_from_mean(tensor).cpu().numpy()
    return reconstructed * global_scale


@torch.no_grad()
def encode_all(model, matrix_scaled, device, batch_size=64):
    model.eval()
    mu_rows = []
    logvar_rows = []

    for start in range(0, len(matrix_scaled), batch_size):
        stop = min(start + batch_size, len(matrix_scaled))
        tensor = torch.from_numpy(matrix_scaled[start:stop]).to(device)
        mu, logvar = model.encode(tensor)
        mu_rows.append(mu.cpu().numpy())
        logvar_rows.append(logvar.cpu().numpy())

    return np.concatenate(mu_rows), np.concatenate(logvar_rows)
