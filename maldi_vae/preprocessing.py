"""MaldiAMRKit preprocessing, 3 Da binning, and validation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from maldiamrkit import MaldiSpectrum
from maldiamrkit.preprocessing import (
    PreprocessingPipeline,
    SqrtTransform,
    SavitzkyGolaySmooth,
    SNIPBaseline,
    TICNormalizer,
    MzTrimmer,
)

from .config import INPUT_DIM, MZ_MIN, MZ_MAX, BIN_WIDTH


def build_preprocessing_pipeline():
    """Recreate the pipeline used in the project."""
    return PreprocessingPipeline(
        [
            ("sqrt", SqrtTransform()),
            ("smooth", SavitzkyGolaySmooth(window_length=21, polyorder=3)),
            ("baseline", SNIPBaseline(half_window=20)),
            ("tic", TICNormalizer()),
            ("trim", MzTrimmer(mz_min=MZ_MIN, mz_max=MZ_MAX)),
        ]
    )


def read_two_column_spectrum(path: Path, comment="#"):
    """Read a two-column MALDI spectrum into NumPy arrays."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment=comment,
        usecols=[0, 1],
        engine="c",
    )
    values = df.to_numpy(dtype=np.float64, copy=False)
    return values[:, 0], values[:, 1]


def read_binned_spectrum(path: Path):
    """Read a DRIAMS-like 6000-bin spectrum."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        usecols=["bin_index", "binned_intensity"],
    )
    return (
        df["bin_index"].to_numpy(dtype=np.int64),
        df["binned_intensity"].to_numpy(dtype=np.float64),
    )


def validate_binned_arrays(bin_index, intensity):
    """Validate the canonical 6000-bin representation."""
    if len(intensity) != INPUT_DIM:
        raise ValueError(f"Expected {INPUT_DIM} bins, got {len(intensity)}.")
    if not np.array_equal(bin_index, np.arange(INPUT_DIM)):
        raise ValueError("Bin indices are not exactly 0..5999.")
    if not np.isfinite(intensity).all():
        raise ValueError("Binned intensities contain NaN or infinity.")
    if (intensity < 0).any():
        raise ValueError("Binned intensities contain negative values.")


def preprocess_spectrum(raw_path: Path, pipeline=None):
    """Load and preprocess one raw spectrum with MaldiAMRKit."""
    if pipeline is None:
        pipeline = build_preprocessing_pipeline()

    spec = MaldiSpectrum(raw_path, pipeline=pipeline)
    spec.preprocess()
    processed = spec.preprocessed

    if processed is None or len(processed) == 0:
        raise ValueError("MaldiAMRKit returned an empty preprocessed spectrum.")

    mz = processed["mass"].to_numpy(dtype=np.float64, copy=False)
    intensity = processed["intensity"].to_numpy(dtype=np.float64, copy=False)

    if not np.isfinite(mz).all() or not np.isfinite(intensity).all():
        raise ValueError("Preprocessed spectrum contains non-finite values.")

    return spec, processed, mz, intensity


def bin_spectrum(spec: MaldiSpectrum):
    """Bin a MaldiSpectrum at 3 Da and validate the exact 6000-bin grid."""
    spec.bin(bin_width=BIN_WIDTH)
    binned = spec.binned

    if binned is None:
        raise ValueError("No binned spectrum was generated.")
    if len(binned) != INPUT_DIM:
        raise ValueError(f"Expected {INPUT_DIM} bins, got {len(binned)}.")

    expected_mz = np.arange(MZ_MIN, MZ_MAX, BIN_WIDTH, dtype=float)
    binned_mz = binned["mass"].to_numpy(dtype=float)
    if not np.allclose(binned_mz, expected_mz):
        raise ValueError("Incorrect 3 Da bin positions.")

    intensity = binned["intensity"].to_numpy(dtype=np.float64)
    if not np.isfinite(intensity).all():
        raise ValueError("Binned intensities contain non-finite values.")

    return binned_mz, intensity


def save_preprocessed(processed: pd.DataFrame, path: Path):
    """Save the two-column reconstructed preprocessed spectrum."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    processed[["mass", "intensity"]].to_csv(path, sep=" ", index=False)


def save_binned(intensity, path: Path):
    """Save in the same two-column format used by DRIAMS binned_6000."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(
        {
            "bin_index": np.arange(INPUT_DIM, dtype=int),
            "binned_intensity": np.asarray(intensity),
        }
    )
    output.to_csv(path, sep=" ", index=False)


def compare_preprocessed_arrays(
    mz_generated,
    y_generated,
    mz_official,
    y_official,
    mz_atol: float = 1e-8,
):
    """Compare reconstructed preprocessing against official DRIAMS preprocessing."""
    result = {
        "n_generated": len(mz_generated),
        "n_official": len(mz_official),
        "same_length": len(mz_generated) == len(mz_official),
    }
    if not result["same_length"]:
        result.update({"mz_match": False, "mz_max_abs_diff": np.nan})
        return result

    generated_finite = np.isfinite(mz_generated).all() and np.isfinite(y_generated).all()
    official_finite = np.isfinite(mz_official).all() and np.isfinite(y_official).all()
    result.update(
        {"generated_finite": generated_finite, "official_finite": official_finite}
    )
    if not generated_finite or not official_finite:
        return result

    mz_max_abs_diff = float(np.max(np.abs(mz_generated - mz_official)))
    result["mz_max_abs_diff"] = mz_max_abs_diff
    result["mz_match"] = mz_max_abs_diff <= mz_atol
    if not result["mz_match"]:
        return result

    result.update(compare_intensity_vectors(y_generated, y_official))
    return result


def compare_intensity_vectors(generated, official):
    """Core point-wise comparison used for preprocessed and binned spectra."""
    generated = np.asarray(generated, dtype=np.float64)
    official = np.asarray(official, dtype=np.float64)

    difference = generated - official
    mae = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference**2)))
    max_abs_error = float(np.max(np.abs(difference)))

    generated_sum = float(generated.sum())
    official_sum = float(official.sum())

    gg = float(np.dot(generated, generated))
    oo = float(np.dot(official, official))
    go = float(np.dot(generated, official))

    cosine = go / np.sqrt(gg * oo) if gg > 0 and oo > 0 else np.nan
    official_rms = np.sqrt(oo / len(official)) if len(official) else np.nan
    nrmse = rmse / official_rms if official_rms > 0 else np.nan

    g_centered = generated - generated.mean()
    o_centered = official - official.mean()
    denom = np.sqrt(np.dot(g_centered, g_centered) * np.dot(o_centered, o_centered))
    pearson = float(np.dot(g_centered, o_centered) / denom) if denom > 0 else np.nan

    if gg > 0:
        scale = go / gg
        scaled_error = generated * scale - official
        scaled_mae = float(np.mean(np.abs(scaled_error)))
        scaled_rmse = float(np.sqrt(np.mean(scaled_error**2)))
    else:
        scale = scaled_mae = scaled_rmse = np.nan

    return {
        "pearson_r": pearson,
        "cosine_similarity": float(cosine),
        "mae": mae,
        "rmse": rmse,
        "nrmse": float(nrmse),
        "max_abs_error": max_abs_error,
        "generated_sum": generated_sum,
        "official_sum": official_sum,
        "sum_ratio_official_to_generated": (
            official_sum / generated_sum if generated_sum != 0 else np.nan
        ),
        "optimal_scale": float(scale),
        "scaled_mae": scaled_mae,
        "scaled_rmse": scaled_rmse,
    }


def compare_binned_spectra(
    generated_bins,
    generated_intensity,
    official_bins,
    official_intensity,
):
    """Compare two DRIAMS-like binned spectra."""
    result = {
        "n_generated": len(generated_intensity),
        "n_official": len(official_intensity),
        "same_length": len(generated_intensity) == len(official_intensity),
    }

    if not result["same_length"]:
        result["same_bins"] = False
        return result

    result["same_bins"] = bool(np.array_equal(generated_bins, official_bins))
    if not result["same_bins"]:
        return result

    result["exactly_6000"] = len(generated_intensity) == INPUT_DIM

    generated_finite = np.isfinite(generated_intensity).all()
    official_finite = np.isfinite(official_intensity).all()
    result["generated_finite"] = bool(generated_finite)
    result["official_finite"] = bool(official_finite)
    if not generated_finite or not official_finite:
        return result

    result.update(compare_intensity_vectors(generated_intensity, official_intensity))
    return result
