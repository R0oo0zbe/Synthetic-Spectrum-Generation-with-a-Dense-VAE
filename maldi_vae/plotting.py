"""Plotting helpers used by the experiment scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _save(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_history(history, plot_dir, title_prefix="Dense beta-VAE"):
    plot_dir = Path(plot_dir)

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Training")
    plt.plot(history["epoch"], history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("beta-VAE loss")
    plt.title(f"{title_prefix}: total loss")
    plt.legend()
    _save(plot_dir / "training_total_loss.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_reconstruction"], label="Training")
    plt.plot(history["epoch"], history["val_reconstruction"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction loss")
    plt.title(f"{title_prefix}: reconstruction")
    plt.legend()
    _save(plot_dir / "reconstruction_loss.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_kl"], label="Training")
    plt.plot(history["epoch"], history["val_kl"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("KL divergence")
    plt.title(f"{title_prefix}: KL")
    plt.legend()
    _save(plot_dir / "kl_loss.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["beta"])
    plt.xlabel("Epoch")
    plt.ylabel("beta")
    plt.title("KL warm-up schedule")
    _save(plot_dir / "beta_schedule.png")


def plot_reconstructions(real_X, reconstructed, indices, codes, mz_axis, plot_dir, n=5):
    plot_dir = Path(plot_dir)
    for local_index, original_index in enumerate(indices[:n]):
        plt.figure(figsize=(14, 5))
        plt.plot(mz_axis, real_X[original_index], label="Real spectrum", linewidth=1)
        plt.plot(
            mz_axis,
            reconstructed[local_index],
            label="VAE reconstruction",
            linewidth=1,
            alpha=0.8,
        )
        plt.xlabel("m/z")
        plt.ylabel("Binned intensity")
        plt.title(f"Real vs reconstruction\n{codes[original_index]}")
        plt.legend()
        _save(plot_dir / f"reconstruction_{local_index + 1:02d}.png")


def plot_synthetic_examples(synthetic, mz_axis, plot_dir, n=3):
    plot_dir = Path(plot_dir)
    for index in range(min(n, len(synthetic))):
        plt.figure(figsize=(14, 5))
        plt.plot(mz_axis, synthetic[index], linewidth=1)
        plt.xlabel("m/z")
        plt.ylabel("Generated binned intensity")
        plt.title(f"Synthetic spectrum {index + 1}")
        _save(plot_dir / f"synthetic_{index + 1:02d}.png")


def plot_mean_and_std(real, synthetic, mz_axis, plot_dir, prefix="tic"):
    plot_dir = Path(plot_dir)

    real_mean, syn_mean = real.mean(axis=0), synthetic.mean(axis=0)
    real_std, syn_std = real.std(axis=0), synthetic.std(axis=0)

    plt.figure(figsize=(14, 5))
    plt.plot(mz_axis, real_mean, label="Real", linewidth=1)
    plt.plot(mz_axis, syn_mean, label="Synthetic", linewidth=1, alpha=0.8)
    plt.xlabel("m/z")
    plt.ylabel("Mean binned intensity")
    plt.title("Mean spectrum: real vs synthetic")
    plt.legend()
    _save(plot_dir / f"{prefix}_mean_spectrum_real_vs_synthetic.png")

    plt.figure(figsize=(14, 5))
    plt.plot(mz_axis, real_std, label="Real", linewidth=1)
    plt.plot(mz_axis, syn_std, label="Synthetic", linewidth=1, alpha=0.8)
    plt.xlabel("m/z")
    plt.ylabel("Across-spectrum standard deviation")
    plt.title("Spectrum-to-spectrum variability")
    plt.legend()
    _save(plot_dir / f"{prefix}_std_spectrum_real_vs_synthetic.png")


def plot_pairwise_histograms(distributions, plot_dir):
    plot_dir = Path(plot_dir)
    for metric, (rr, ss, rs) in distributions.items():
        plt.figure(figsize=(8, 5))
        plt.hist(rr, bins=40, density=True, alpha=0.5, label="Real ↔ Real")
        plt.hist(ss, bins=40, density=True, alpha=0.5, label="Synthetic ↔ Synthetic")
        plt.hist(rs, bins=40, density=True, alpha=0.35, label="Real ↔ Synthetic")
        plt.xlabel(metric.replace("_", " "))
        plt.ylabel("Density")
        plt.title("Pairwise spectral diversity")
        plt.legend()
        _save(plot_dir / f"pairwise_{metric}.png")


def plot_pca(real_features, synthetic_features, explained_ratio, plot_path, real_label="Held-out real test"):
    if real_features.shape[1] < 2:
        return
    plt.figure(figsize=(8, 7))
    plt.scatter(real_features[:, 0], real_features[:, 1], s=55, alpha=0.8, label=real_label)
    plt.scatter(synthetic_features[:, 0], synthetic_features[:, 1], s=45, alpha=0.65, label="Synthetic")
    plt.xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}%)")
    plt.title("Real vs synthetic in training-fitted PCA space")
    plt.legend()
    _save(plot_path)


def plot_formal_fd(rr_mean, rr_std, rs_mean, rs_std, plot_path):
    labels = ["Real vs Real\n11 vs 11", "Real vs Synthetic\n11 vs 11"]
    means = [rr_mean, rs_mean]
    errors = [rr_std, rs_std]
    x = np.arange(2)

    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=errors, capsize=6)
    plt.xticks(x, labels)
    plt.ylabel("PCA Fréchet Distance")
    plt.title("Held-out distribution comparison\nMean ± SD across evaluation seeds")
    _save(plot_path)


def plot_prdc_comparison(real_real_means, real_real_stds, syn_means, syn_stds, plot_path):
    names = ["Precision", "Recall", "Density", "Coverage"]
    x = np.arange(len(names))
    width = 0.36

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, real_real_means, width, yerr=real_real_stds, capsize=5, label="Real vs Real baseline")
    plt.bar(x + width / 2, syn_means, width, yerr=syn_stds, capsize=5, label="Real test vs Synthetic")
    plt.xticks(x, names)
    plt.ylabel("Metric value")
    plt.title("Generative precision / recall / density / coverage\nMean ± SD across evaluation seeds")
    plt.legend()
    _save(plot_path)


def plot_binned_comparison(
    code,
    generated_bins,
    generated_intensity,
    official_intensity,
    metrics,
    plot_dir,
    zoom_range=(4000, 6000),
):
    """Reproduce the key official-vs-MaldiAMRKit binned comparison figures."""
    plot_dir = Path(plot_dir)
    mz = 2000 + np.asarray(generated_bins) * 3

    pearson = metrics.get("pearson_r", np.nan)
    cosine = metrics.get("cosine_similarity", np.nan)
    rmse = metrics.get("rmse", np.nan)

    plt.figure(figsize=(14, 5))
    plt.plot(mz, official_intensity, label="Official DRIAMS", linewidth=1, alpha=0.8)
    plt.plot(mz, generated_intensity, label="MaldiAMRKit", linewidth=1, alpha=0.7)
    plt.xlabel("m/z")
    plt.ylabel("Binned intensity")
    plt.title(
        f"Official vs MaldiAMRKit\n{code}\n"
        f"Pearson r={pearson:.6f}, Cosine={cosine:.6f}, RMSE={rmse:.3e}"
    )
    plt.legend()
    _save(plot_dir / f"{code}_full_overlay.png")

    if zoom_range is not None:
        mz_min, mz_max = zoom_range
        mask = (mz >= mz_min) & (mz <= mz_max)
        plt.figure(figsize=(14, 5))
        plt.plot(mz[mask], official_intensity[mask], label="Official DRIAMS", linewidth=1.2)
        plt.plot(mz[mask], generated_intensity[mask], label="MaldiAMRKit", linewidth=1.2, alpha=0.8)
        plt.xlabel("m/z")
        plt.ylabel("Binned intensity")
        plt.title(f"Zoomed comparison: {mz_min}–{mz_max} m/z\n{code}")
        plt.legend()
        _save(plot_dir / f"{code}_zoom_{mz_min}_{mz_max}.png")

    scale = metrics.get("optimal_scale", np.nan)
    if np.isfinite(scale):
        plt.figure(figsize=(14, 5))
        plt.plot(mz, official_intensity, label="Official DRIAMS", linewidth=1)
        plt.plot(
            mz,
            generated_intensity * scale,
            label="MaldiAMRKit after optimal scale correction",
            linewidth=1,
            alpha=0.8,
        )
        plt.xlabel("m/z")
        plt.ylabel("Binned intensity")
        plt.title(f"Scale-corrected comparison\n{code}\nOptimal scale={scale:.5f}")
        plt.legend()
        _save(plot_dir / f"{code}_scaled_overlay.png")

    plt.figure(figsize=(7, 7))
    plt.scatter(official_intensity, generated_intensity, s=10, alpha=0.4)
    minimum = min(official_intensity.min(), generated_intensity.min())
    maximum = max(official_intensity.max(), generated_intensity.max())
    plt.plot([minimum, maximum], [minimum, maximum], linestyle="--", label="Perfect agreement")
    plt.xlabel("Official DRIAMS intensity")
    plt.ylabel("MaldiAMRKit intensity")
    plt.title(f"Bin-by-bin intensity comparison\nPearson r={pearson:.6f}")
    plt.legend()
    _save(plot_dir / f"{code}_scatter.png")


def plot_preprocessing_metric_distributions(valid_comparison, plot_dir):
    """Summary histograms for binned preprocessing agreement."""
    plot_dir = Path(plot_dir)

    if "cosine_similarity" in valid_comparison:
        values = valid_comparison["cosine_similarity"].dropna()
        if len(values):
            plt.figure(figsize=(8, 5))
            plt.hist(values, bins=30)
            plt.xlabel("Cosine similarity")
            plt.ylabel("Number of spectra")
            plt.title("Cosine similarity across spectra")
            _save(plot_dir / "cosine_similarity_distribution.png")

    if "optimal_scale" in valid_comparison:
        values = valid_comparison["optimal_scale"].dropna()
        if len(values):
            plt.figure(figsize=(8, 5))
            plt.hist(values, bins=30)
            plt.xlabel("Optimal scale factor")
            plt.ylabel("Number of spectra")
            plt.title("Intensity scaling difference: official DRIAMS vs MaldiAMRKit")
            _save(plot_dir / "optimal_scale_distribution.png")
