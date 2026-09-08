# Refactor mapping

The repository keeps the scientific workflow but removes duplicated implementation.

| Original working file | Refactored location |
|---|---|
| `0_new.py` | `scripts/01_prepare_data.py` + `maldi_vae/preprocessing.py` |
| `comparing.py` | `scripts/01_prepare_data.py` + `maldi_vae/plotting.py` |
| `1_new.py` | `scripts/01_prepare_data.py` + `maldi_vae/data.py` |
| `3_dense_beta_vae.py` | `scripts/02_train_baseline.py` + shared model/training modules |
| `4_dense_beta_sweep.py` | `scripts/03_beta_sweep.py` + shared model/training modules |
| `5_analyze_beta_0p01_real_vs_synthetic.py` | `scripts/04_descriptive_evaluation.py` |
| `6_tic_correct_beta_0p01_and_compare.py` | `scripts/04_descriptive_evaluation.py` |
| `7_formal_beta_0p01_distribution_evaluation.py` | `scripts/05_formal_evaluation.py` |

Scientific constants that generated the reported results remain unchanged:
- training seed 42
- architecture 6000 -> 512 -> 128 -> 64 -> 16
- beta sweep 0.001, 0.003, 0.01, 0.03 plus reused beta 0.10 baseline
- selected beta 0.01
- formal PCA dimension 5
- PRDC k=3
- formal evaluation seeds 0-9
- held-out test size 22
- TIC normalization before formal distribution metrics
