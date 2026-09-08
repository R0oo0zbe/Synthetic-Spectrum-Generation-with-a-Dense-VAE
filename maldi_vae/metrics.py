"""Reusable reconstruction, diversity, PCA-FD, and PRDC metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors


def safe_pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) <= 0 or np.std(b) <= 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def vector_cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 0:
        return np.nan
    return float(np.dot(a, b) / denominator)


def per_spectrum_statistics(matrix, name, low_intensity_threshold=None):
    data = {
        "dataset": name,
        "spectrum_index": np.arange(len(matrix), dtype=np.int64),
        "total_intensity": matrix.sum(axis=1),
        "mean_intensity": matrix.mean(axis=1),
        "std_intensity": matrix.std(axis=1),
        "max_intensity": matrix.max(axis=1),
        "median_intensity": np.median(matrix, axis=1),
        "fraction_exact_zero": (matrix == 0).mean(axis=1),
    }
    if low_intensity_threshold is not None:
        data["fraction_below_low_threshold"] = (
            matrix < low_intensity_threshold
        ).mean(axis=1)
    return pd.DataFrame(data)


def upper_triangle_values(matrix):
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def center_rows(matrix):
    return matrix - matrix.mean(axis=1, keepdims=True)


def pairwise_diversity(real, synthetic):
    """Return cosine, Pearson, and Euclidean pairwise distributions."""
    rr_cos = upper_triangle_values(cosine_similarity(real))
    ss_cos = upper_triangle_values(cosine_similarity(synthetic))
    rs_cos = cosine_similarity(real, synthetic).ravel()

    rr_pea = upper_triangle_values(cosine_similarity(center_rows(real)))
    ss_pea = upper_triangle_values(cosine_similarity(center_rows(synthetic)))
    rs_pea = cosine_similarity(center_rows(real), center_rows(synthetic)).ravel()

    rr_euc = upper_triangle_values(euclidean_distances(real))
    ss_euc = upper_triangle_values(euclidean_distances(synthetic))
    rs_euc = euclidean_distances(real, synthetic).ravel()

    return {
        "cosine_similarity": (rr_cos, ss_cos, rs_cos),
        "pearson_r": (rr_pea, ss_pea, rs_pea),
        "euclidean_distance": (rr_euc, ss_euc, rs_euc),
    }


def summarize_values(metric, comparison, values):
    return {
        "metric": metric,
        "comparison": comparison,
        "n_pairs": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }


def pairwise_summary_table(distributions):
    rows = []
    for metric, (rr, ss, rs) in distributions.items():
        rows.extend(
            [
                summarize_values(metric, "real_real", rr),
                summarize_values(metric, "synthetic_synthetic", ss),
                summarize_values(metric, "real_synthetic", rs),
            ]
        )
    return pd.DataFrame(rows)


def nearest_neighbor_summary(real_train, synthetic):
    """Nearest-neighbor baselines in the original 6000-D space."""
    rr_euc = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(real_train)
    rr_euc_dist, _ = rr_euc.kneighbors(real_train)
    real_euc = rr_euc_dist[:, 1]

    syn_euc_model = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(real_train)
    syn_euc, _ = syn_euc_model.kneighbors(synthetic)
    syn_euc = syn_euc[:, 0]

    rr_cos = NearestNeighbors(n_neighbors=2, metric="cosine").fit(real_train)
    rr_cos_dist, _ = rr_cos.kneighbors(real_train)
    real_cos = 1.0 - rr_cos_dist[:, 1]

    syn_cos_model = NearestNeighbors(n_neighbors=1, metric="cosine").fit(real_train)
    syn_cos_dist, _ = syn_cos_model.kneighbors(synthetic)
    syn_cos = 1.0 - syn_cos_dist[:, 0]

    table = pd.DataFrame(
        [
            {
                "group": "real_to_other_real_train",
                "median_euclidean": float(np.median(real_euc)),
                "mean_euclidean": float(np.mean(real_euc)),
                "median_cosine": float(np.median(real_cos)),
                "mean_cosine": float(np.mean(real_cos)),
            },
            {
                "group": "synthetic_to_real_train",
                "median_euclidean": float(np.median(syn_euc)),
                "mean_euclidean": float(np.mean(syn_euc)),
                "median_cosine": float(np.median(syn_cos)),
                "mean_cosine": float(np.mean(syn_cos)),
            },
        ]
    )

    return table, {
        "real_nn_euclidean": real_euc,
        "synthetic_nn_euclidean": syn_euc,
        "real_nn_cosine": real_cos,
        "synthetic_nn_cosine": syn_cos,
    }


def symmetric_psd_sqrt(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    root = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    return (root + root.T) / 2.0


def covariance_matrix(features):
    cov = np.atleast_2d(np.cov(features, rowvar=False, ddof=1))
    return (cov + cov.T) / 2.0


def pca_frechet_distance(features_a, features_b):
    """Fréchet distance between Gaussian approximations in PCA feature space."""
    A = np.asarray(features_a, dtype=np.float64)
    B = np.asarray(features_b, dtype=np.float64)

    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError("PCA-FD inputs must be 2D with the same feature dimension.")
    if len(A) < 2 or len(B) < 2:
        raise ValueError("At least two observations are required per group.")

    mu_a, mu_b = A.mean(axis=0), B.mean(axis=0)
    cov_a, cov_b = covariance_matrix(A), covariance_matrix(B)

    mean_term = float(np.sum((mu_a - mu_b) ** 2))
    sqrt_a = symmetric_psd_sqrt(cov_a)
    middle = sqrt_a @ cov_b @ sqrt_a
    covariance_term = float(
        np.trace(cov_a + cov_b - 2.0 * symmetric_psd_sqrt(middle))
    )

    if covariance_term < 0 and abs(covariance_term) < 1e-12:
        covariance_term = 0.0

    distance = mean_term + covariance_term
    if distance < 0 and abs(distance) < 1e-12:
        distance = 0.0

    return {
        "pca_fd": float(distance),
        "pca_fd_mean_term": mean_term,
        "pca_fd_covariance_term": covariance_term,
    }


def kth_neighbor_radii(features, k):
    features = np.asarray(features, dtype=np.float64)
    if len(features) <= k:
        raise ValueError(f"Need more than k={k} observations.")

    distances = pairwise_distances(features, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    return np.partition(distances, kth=k - 1, axis=1)[:, k - 1]


def precision_recall_density_coverage(real_features, generated_features, k=3):
    """Generative-model PRDC metrics in one shared Euclidean feature space."""
    real = np.asarray(real_features, dtype=np.float64)
    generated = np.asarray(generated_features, dtype=np.float64)

    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]:
        raise ValueError("PRDC inputs must be 2D with matching feature dimensions.")
    if len(real) <= k or len(generated) <= k:
        raise ValueError(f"Need more than k={k} observations in both groups.")

    real_radii = kth_neighbor_radii(real, k)
    generated_radii = kth_neighbor_radii(generated, k)

    distances_gr = pairwise_distances(generated, real, metric="euclidean")

    generated_inside_real = distances_gr <= real_radii[None, :]
    precision = float(np.mean(np.any(generated_inside_real, axis=1)))
    density = float(np.mean(np.sum(generated_inside_real, axis=1) / float(k)))

    real_inside_generated = distances_gr <= generated_radii[:, None]
    recall = float(np.mean(np.any(real_inside_generated, axis=0)))

    nearest_generated = distances_gr.min(axis=0)
    coverage = float(np.mean(nearest_generated <= real_radii))

    return {
        "precision": precision,
        "recall": recall,
        "density": density,
        "coverage": coverage,
    }


def real_real_prdc_baseline(features_a, features_b, k=3):
    """Symmetric PRDC reference for two real subsets."""
    ab = precision_recall_density_coverage(features_a, features_b, k)
    ba = precision_recall_density_coverage(features_b, features_a, k)
    return {metric: float((ab[metric] + ba[metric]) / 2.0) for metric in ab}
