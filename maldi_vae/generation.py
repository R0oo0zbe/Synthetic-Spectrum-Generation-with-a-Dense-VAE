"""Synthetic spectrum generation and export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import INPUT_DIM, MZ_AXIS


@torch.no_grad()
def generate_synthetic(
    model,
    n_samples,
    latent_dim,
    global_scale,
    device,
    seed=None,
):
    """Sample z~N(0,I), decode, and reverse the training-only global scale."""
    model.eval()

    if seed is None:
        z = torch.randn(n_samples, latent_dim, device=device)
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        z = torch.randn(
            n_samples,
            latent_dim,
            generator=generator,
            device=device,
        )

    synthetic_scaled = model.decode(z).cpu().numpy()
    return synthetic_scaled * global_scale


def save_synthetic_arrays(synthetic, output_dir: Path, stem="synthetic_spectra"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{stem}.npy", synthetic)
    np.savez_compressed(
        output_dir / f"{stem}.npz",
        spectra=synthetic,
        mz=MZ_AXIS,
    )


def save_synthetic_txt(synthetic, output_dir: Path):
    """Save generated spectra in DRIAMS-like binned format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, spectrum in enumerate(synthetic, start=1):
        pd.DataFrame(
            {
                "bin_index": np.arange(INPUT_DIM, dtype=np.int64),
                "binned_intensity": spectrum,
            }
        ).to_csv(
            output_dir / f"synthetic_{index:04d}.txt",
            sep=" ",
            index=False,
        )
