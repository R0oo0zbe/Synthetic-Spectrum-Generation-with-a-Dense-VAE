"""Stable project constants shared across experiments."""

from pathlib import Path
import numpy as np

YEAR = "2018"

# Spectrum representation
INPUT_DIM = 6000
MZ_MIN = 2000
MZ_MAX = 20000
BIN_WIDTH = 3
MZ_AXIS = MZ_MIN + BIN_WIDTH * np.arange(INPUT_DIM)

# Cohort
SPECIES = "Escherichia coli"
ANTIBIOTIC = "Ceftriaxone"
VALID_AST_LABELS = ("S", "I", "R")
LABEL_MAP = {"S": 0, "I": 1, "R": 1}

# Dense beta-VAE architecture
HIDDEN_1 = 512
HIDDEN_2 = 128
HIDDEN_3 = 64
LATENT_DIM = 16
DROPOUT = 0.10

# Reproducibility / formal evaluation
TRAINING_SEED = 42
PCA_DIM = 5
PRDC_K = 3
FORMAL_EVALUATION_SEEDS = tuple(range(10))


def beta_tag(beta: float) -> str:
    """Convert beta=0.01 to the directory tag ``beta_0p01``."""
    return "beta_" + f"{beta:g}".replace(".", "p")


def project_paths(driams_root=Path("DRIAMS-B"), year=YEAR):
    """Return the canonical paths used by the repository."""
    driams_root = Path(driams_root)
    vae_results = driams_root / "vae_results" / year

    return {
        "root": driams_root,
        "metadata": driams_root / "id" / year / f"{year}_clean.csv",
        "raw": driams_root / "raw" / year,
        "official_preprocessed": driams_root / "preprocessed" / year,
        "official_binned": driams_root / "binned_6000" / year,
        "generated_preprocessed": driams_root / "preprocess_MaldiAMRKit" / year,
        "generated_binned": driams_root / "binned_6000_MaldiAMRKit" / year,
        "vae_data": driams_root / "vae_data" / year,
        "baseline": vae_results / "dense_beta_vae",
        "sweep": vae_results / "dense_beta_sweep",
    }
