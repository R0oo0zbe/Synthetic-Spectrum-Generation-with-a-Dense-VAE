#!/usr/bin/env python3
"""Prepare the E. coli / ceftriaxone MALDI dataset from DRIAMS-B.

This combines the original preprocessing, bin-comparison, and X-building
scripts into one reproducible data-preparation stage while keeping the
scientific operations unchanged.
"""

from pathlib import Path
import argparse
import gc
import sys

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from maldiamrkit import MaldiSpectrum

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maldi_vae.config import INPUT_DIM, ANTIBIOTIC, project_paths
from maldi_vae.data import select_ecoli_ceftriaxone, encode_ast_labels
from maldi_vae.preprocessing import (
    build_preprocessing_pipeline,
    read_two_column_spectrum,
    read_binned_spectrum,
    preprocess_spectrum,
    bin_spectrum,
    save_preprocessed,
    save_binned,
    compare_preprocessed_arrays,
    compare_binned_spectra,
    validate_binned_arrays,
)
from maldi_vae.plotting import (
    plot_binned_comparison,
    plot_preprocessing_metric_distributions,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driams-root", default="DRIAMS-B")
    parser.add_argument("--year", default="2018")
    parser.add_argument("--force", action="store_true", help="Recompute existing generated spectra.")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = project_paths(Path(args.driams_root), args.year)

    for key in ["generated_preprocessed", "generated_binned", "vae_data"]:
        paths[key].mkdir(parents=True, exist_ok=True)

    selected = select_ecoli_ceftriaxone(paths["metadata"])
    label_col = f"{ANTIBIOTIC}_clean"

    selected["raw_path"] = selected["code"].map(lambda c: paths["raw"] / f"{c}.txt")
    selected["official_preprocessed_path"] = selected["code"].map(
        lambda c: paths["official_preprocessed"] / f"{c}.txt"
    )
    selected["official_binned_path"] = selected["code"].map(
        lambda c: paths["official_binned"] / f"{c}.txt"
    )
    selected["generated_preprocessed_path"] = selected["code"].map(
        lambda c: paths["generated_preprocessed"] / f"{c}.txt"
    )
    selected["generated_binned_path"] = selected["code"].map(
        lambda c: paths["generated_binned"] / f"{c}.txt"
    )

    usable = selected.loc[
        selected["raw_path"].map(Path.exists)
        & selected["official_preprocessed_path"].map(Path.exists)
    ].reset_index(drop=True)

    print(f"Selected E. coli / ceftriaxone spectra: {len(selected)}")
    print(selected[label_col].value_counts())
    print(f"Usable raw + official-preprocessed spectra: {len(usable)}")

    pipeline = build_preprocessing_pipeline()

    processing_rows = []
    preprocessing_comparison_rows = []

    for row in tqdm(usable.itertuples(index=False), total=len(usable), desc="Preprocess + bin"):
        code = row.code
        generated_path = row.generated_preprocessed_path
        binned_path = row.generated_binned_path

        try:
            use_existing = generated_path.exists() and not args.force

            if use_existing:
                mz_generated, y_generated = read_two_column_spectrum(
                    generated_path, comment=None
                )
                processing_status = "skipped_existing"
                spec = None
            else:
                spec, processed, mz_generated, y_generated = preprocess_spectrum(
                    row.raw_path, pipeline
                )
                save_preprocessed(processed, generated_path)
                processing_status = "generated"

            if binned_path.exists() and not args.force:
                bin_index, binned_intensity = read_binned_spectrum(binned_path)
                validate_binned_arrays(bin_index, binned_intensity)
            else:
                if spec is None:
                    bin_spec = MaldiSpectrum(generated_path)
                    _, binned_intensity = bin_spectrum(bin_spec)
                else:
                    _, binned_intensity = bin_spectrum(spec)
                save_binned(binned_intensity, binned_path)

            processing_rows.append(
                {
                    "code": code,
                    "status": processing_status,
                    "n_measurements": len(mz_generated),
                    "mz_min": float(mz_generated[0]),
                    "mz_max": float(mz_generated[-1]),
                    "error": None,
                }
            )

            mz_official, y_official = read_two_column_spectrum(
                row.official_preprocessed_path, comment="#"
            )
            metrics = compare_preprocessed_arrays(
                mz_generated, y_generated, mz_official, y_official
            )
            metrics.update({"code": code, "status": "success", "error": None})
            preprocessing_comparison_rows.append(metrics)

        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            processing_rows.append(
                {
                    "code": code,
                    "status": "failed",
                    "n_measurements": np.nan,
                    "mz_min": np.nan,
                    "mz_max": np.nan,
                    "error": message,
                }
            )
            preprocessing_comparison_rows.append(
                {"code": code, "status": "failed", "error": message}
            )
        finally:
            gc.collect()

    prep_root = paths["generated_preprocessed"].parent
    pd.DataFrame(processing_rows).to_csv(
        prep_root / f"processing_log_{args.year}.csv", index=False
    )
    prep_comparison = pd.DataFrame(preprocessing_comparison_rows)
    prep_comparison.to_csv(
        prep_root / f"preprocessing_assessment_{args.year}.csv", index=False
    )

    successful_prep = prep_comparison.loc[
        prep_comparison.get("status", pd.Series(dtype=str)) == "success"
    ].copy()
    prep_summary_cols = [
        "pearson_r", "cosine_similarity", "mae", "rmse", "nrmse",
        "max_abs_error", "generated_sum", "official_sum",
        "sum_ratio_official_to_generated", "optimal_scale",
        "scaled_mae", "scaled_rmse",
    ]
    prep_summary_cols = [c for c in prep_summary_cols if c in successful_prep.columns]
    if len(successful_prep) and prep_summary_cols:
        successful_prep[prep_summary_cols].describe().T.to_csv(
            prep_root / f"preprocessing_summary_{args.year}.csv"
        )

    # --------------------------------------------------------
    # Compare generated 6000-bin spectra with official DRIAMS bins
    # --------------------------------------------------------
    binned_rows = []
    for row in tqdm(selected.itertuples(index=False), total=len(selected), desc="Compare binned spectra"):
        if not row.generated_binned_path.exists() or not row.official_binned_path.exists():
            continue

        try:
            gb, gi = read_binned_spectrum(row.generated_binned_path)
            ob, oi = read_binned_spectrum(row.official_binned_path)
            metrics = compare_binned_spectra(gb, gi, ob, oi)
            metrics.update(
                {
                    "code": row.code,
                    "Ceftriaxone": getattr(row, label_col),
                    "status": "success",
                    "error": None,
                }
            )
        except Exception as exc:
            metrics = {
                "code": row.code,
                "Ceftriaxone": getattr(row, label_col),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        binned_rows.append(metrics)

    bin_comparison_dir = paths["root"] / "bin_comparison"
    bin_comparison_dir.mkdir(parents=True, exist_ok=True)
    binned_comparison = pd.DataFrame(binned_rows)
    binned_comparison.to_csv(
        bin_comparison_dir / f"binned_MaldiAMRKit_vs_official_{args.year}.csv",
        index=False,
    )

    valid = binned_comparison.loc[
        (binned_comparison["status"] == "success")
        & binned_comparison.get("same_bins", False)
    ].copy()

    summary_cols = [
        "pearson_r", "cosine_similarity", "mae", "rmse", "nrmse",
        "max_abs_error", "generated_sum", "official_sum",
        "sum_ratio_official_to_generated", "optimal_scale",
        "scaled_mae", "scaled_rmse",
    ]
    summary_cols = [c for c in summary_cols if c in valid.columns]
    if len(valid) and summary_cols:
        valid[summary_cols].describe().T.to_csv(
            bin_comparison_dir / f"binned_comparison_summary_{args.year}.csv"
        )

        comparison_plot_dir = bin_comparison_dir / "plots"
        plot_preprocessing_metric_distributions(valid, comparison_plot_dir)

        # Recreate representative best/worst comparison plots without
        # hard-coding a spectrum UUID.
        ranked = valid.dropna(subset=["pearson_r"]).sort_values("pearson_r")
        if len(ranked):
            representative_codes = {
                "worst": str(ranked.iloc[0]["code"]),
                "best": str(ranked.iloc[-1]["code"]),
            }
            for _, representative_code in representative_codes.items():
                row = selected.loc[selected["code"].astype(str) == representative_code].iloc[0]
                gb, gi = read_binned_spectrum(row["generated_binned_path"])
                ob, oi = read_binned_spectrum(row["official_binned_path"])
                metrics = compare_binned_spectra(gb, gi, ob, oi)
                plot_binned_comparison(
                    representative_code,
                    gb,
                    gi,
                    oi,
                    metrics,
                    comparison_plot_dir,
                    zoom_range=(4000, 6000),
                )

    # --------------------------------------------------------
    # Build X, y, codes from the generated binned spectra
    # --------------------------------------------------------
    spectra, codes, labels, failed = [], [], [], []

    for row in tqdm(selected.itertuples(index=False), total=len(selected), desc="Build VAE matrix"):
        path = row.generated_binned_path
        if not path.exists():
            failed.append({"code": row.code, "error": "Generated binned file missing."})
            continue
        try:
            bin_index, intensity = read_binned_spectrum(path)
            validate_binned_arrays(bin_index, intensity)
            spectra.append(intensity.astype(np.float32, copy=False))
            codes.append(str(row.code))
            labels.append(getattr(row, label_col))
        except Exception as exc:
            failed.append({"code": row.code, "error": f"{type(exc).__name__}: {exc}"})

    if not spectra:
        raise RuntimeError("No valid binned spectra were available to build X.")

    X = np.stack(spectra, axis=0)
    codes = np.asarray(codes)
    labels = np.asarray(labels)
    y = encode_ast_labels(labels)

    np.save(paths["vae_data"] / "X.npy", X)
    np.save(paths["vae_data"] / "y.npy", y)
    np.save(paths["vae_data"] / "codes.npy", codes)
    np.savez_compressed(
        paths["vae_data"] / "ecoli_ceftriaxone_6000.npz",
        X=X,
        y=y,
        codes=codes,
    )
    pd.DataFrame(
        {"code": codes, "Ceftriaxone": labels, "binary_label": y}
    ).to_csv(paths["vae_data"] / "vae_metadata.csv", index=False)

    if failed:
        pd.DataFrame(failed).to_csv(paths["vae_data"] / "failed_spectra.csv", index=False)

    print("\nDATA PREPARATION COMPLETE")
    print("X shape:", X.shape)
    print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))
    print("Mean TIC:", X.sum(axis=1).mean())
    print("VAE data:", paths["vae_data"])


if __name__ == "__main__":
    main()
