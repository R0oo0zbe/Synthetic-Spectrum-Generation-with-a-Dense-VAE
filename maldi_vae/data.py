"""Dataset selection, validation, split handling, and normalization."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .config import (
    INPUT_DIM,
    SPECIES,
    ANTIBIOTIC,
    VALID_AST_LABELS,
    LABEL_MAP,
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch consistently."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_ecoli_ceftriaxone(metadata_path: Path) -> pd.DataFrame:
    """Select unique E. coli spectra with valid ceftriaxone S/I/R labels."""
    metadata = pd.read_csv(
        metadata_path,
        usecols=["code", "species", ANTIBIOTIC],
    )

    metadata["code"] = metadata["code"].astype("string").str.strip()
    metadata["species"] = metadata["species"].astype("string").str.strip()
    metadata[f"{ANTIBIOTIC}_clean"] = (
        metadata[ANTIBIOTIC].astype("string").str.strip().str.upper()
    )

    clean_label = f"{ANTIBIOTIC}_clean"
    selected = metadata.loc[
        (metadata["species"] == SPECIES)
        & metadata[clean_label].isin(VALID_AST_LABELS),
        ["code", clean_label],
    ].copy()

    return (
        selected.dropna(subset=["code"])
        .drop_duplicates(subset=["code"])
        .reset_index(drop=True)
    )


def load_vae_arrays(vae_data_dir: Path, dtype=np.float32):
    """Load X, y, and codes saved by the data-preparation stage."""
    vae_data_dir = Path(vae_data_dir)
    X = np.load(vae_data_dir / "X.npy").astype(dtype, copy=False)
    y = np.load(vae_data_dir / "y.npy").astype(np.int64, copy=False)
    codes = np.load(vae_data_dir / "codes.npy", allow_pickle=True)
    validate_vae_arrays(X, y, codes)
    return X, y, codes


def validate_vae_arrays(X, y, codes, input_dim: int = INPUT_DIM) -> None:
    """Validate the matrix used by the VAE."""
    if X.ndim != 2 or X.shape[1] != input_dim:
        raise ValueError(f"Expected X shape (N, {input_dim}), got {X.shape}.")
    if len(X) != len(y) or len(X) != len(codes):
        raise ValueError("X, y, and codes must contain the same number of samples.")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")
    if (X < 0).any():
        raise ValueError("X contains negative intensities.")


def encode_ast_labels(labels) -> np.ndarray:
    """Map S -> 0 and I/R -> 1."""
    try:
        return np.asarray([LABEL_MAP[str(v)] for v in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unexpected AST label: {exc.args[0]}") from exc


def create_stratified_split(
    y: np.ndarray,
    seed: int = 42,
    test_size: float = 0.10,
    validation_size: float = 0.10,
):
    """Create the original 80/10/10-ish stratified split."""
    all_indices = np.arange(len(y))

    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    validation_fraction = validation_size / (1.0 - test_size)

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y[train_val_idx],
    )

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def save_split_assignments(path: Path, codes, y, train_idx, val_idx, test_idx):
    """Persist exact split membership for all later experiments."""
    split_name = np.empty(len(y), dtype=object)
    split_name[train_idx] = "train"
    split_name[val_idx] = "validation"
    split_name[test_idx] = "test"

    table = pd.DataFrame(
        {
            "index": np.arange(len(y)),
            "code": np.asarray(codes).astype(str),
            "label": y,
            "split": split_name,
        }
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def load_saved_split(path: Path, codes):
    """Load and validate a previously saved split table."""
    table = pd.read_csv(path).sort_values("index").reset_index(drop=True)

    if len(table) != len(codes):
        raise ValueError("Saved split length does not match the dataset.")

    if not np.array_equal(table["code"].astype(str).to_numpy(), np.asarray(codes).astype(str)):
        raise ValueError("Saved split codes do not match codes.npy.")

    train_idx = table.loc[table["split"] == "train", "index"].to_numpy(dtype=np.int64)
    val_idx = table.loc[table["split"] == "validation", "index"].to_numpy(dtype=np.int64)
    test_idx = table.loc[table["split"] == "test", "index"].to_numpy(dtype=np.int64)

    return table, train_idx, val_idx, test_idx


def tic_normalize(matrix, name: str = "matrix") -> np.ndarray:
    """TIC-normalize every spectrum independently."""
    matrix = np.asarray(matrix, dtype=np.float64)
    sums = matrix.sum(axis=1, keepdims=True)

    if not np.isfinite(sums).all():
        raise ValueError(f"{name}: non-finite TIC sums.")
    if np.any(sums <= 0):
        bad = np.where(sums[:, 0] <= 0)[0]
        raise ValueError(f"{name}: non-positive TIC for rows {bad[:20].tolist()}.")

    normalized = matrix / sums
    if not np.isfinite(normalized).all():
        raise ValueError(f"{name}: non-finite values after TIC normalization.")
    return normalized


def training_global_scale(X: np.ndarray, train_idx) -> float:
    """Learn the single reversible numerical scale from training data only."""
    scale = float(np.max(X[train_idx]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid global scale: {scale}")
    return scale
