# DRIAMS MALDI-TOF Synthetic Spectrum Generation with a Dense β-VAE

This repository contains the code used for a master's course project on
synthetic MALDI-TOF spectrum generation for antimicrobial-resistance research.

## Study cohort

- Dataset: DRIAMS-B (2018)
- Species: *Escherichia coli*
- Antibiotic: ceftriaxone
- Final spectra: 213
- Representation: 6000 bins
- m/z interval: 2000–20000
- Bin width: 3 Da
- Binary labels: S -> 0, I/R -> 1

The VAE itself is unconditional; the resistance labels are used for
stratified splitting and downstream bookkeeping.

## Preprocessing

The MaldiAMRKit pipeline is:

1. square-root variance stabilization
2. Savitzky-Golay smoothing (window 21, polynomial order 3)
3. SNIP baseline correction (half-window 20)
4. TIC normalization
5. trimming to 2000–20000 m/z
6. 3 Da binning to 6000 features

The generated preprocessing and binned spectra are compared with the official
DRIAMS files.

## Model

Dense β-VAE:

`6000 -> 512 -> 128 -> 64 -> latent(16) -> 64 -> 128 -> 512 -> 6000`

- activations: GELU
- decoder output: Softplus
- baseline beta: 0.10
- selected beta after sweep: 0.01
- split seed: 42
- train / validation / test: 169 / 22 / 22

## Reproduction order

Run from the repository root:

```bash
python scripts/01_prepare_data.py
python scripts/02_train_baseline.py
python scripts/03_beta_sweep.py
python scripts/04_descriptive_evaluation.py
python scripts/05_formal_evaluation.py
```

The default data root is `DRIAMS-B/`. To use another location:

```bash
python scripts/01_prepare_data.py --driams-root /path/to/DRIAMS-B
```

## Formal generative evaluation

The final β=0.01 model is frozen. Formal evaluation uses only the held-out
22-spectrum test set.

- PCA is fitted only on real training spectra.
- PCA dimension: 5
- adapted Fréchet metric: PCA Fréchet Distance (PCA-FD), not image FID
- generative precision, recall, density, coverage
- k = 3 for PRDC neighborhoods
- evaluation seeds: 0–9
- 22 synthetic spectra per seed
- real-vs-real baseline: held-out test split into 11 + 11
- generated spectra are TIC-normalized before formal comparison

Expected final values from the completed experiment were approximately:

- PCA-FD real-real, 11 vs 11: `1.37498e-05 ± 4.4002e-06`
- PCA-FD real-synthetic, 11 vs 11: `1.73587e-05 ± 4.08365e-06`
- PCA-FD real test-synthetic, 22 vs 22: `1.36265e-05 ± 1.98544e-06`
- Precision: `0.995455 ± 0.014374`
- Recall: `0.568182 ± 0.117851`
- Density: `1.39545 ± 0.146978`
- Coverage: `0.672727 ± 0.085173`

These results indicate high synthetic fidelity but incomplete coverage of the
real-data diversity.

## Environment

Install the packages in `requirements.txt`. For the exact environment used for
the final report, record the versions from the machine that produced the
results:

```bash
python --version
pip freeze > requirements-lock.txt
```

Commit `requirements-lock.txt` together with the repository.

## Data

The DRIAMS data are not distributed in this repository. See `data/README.md`.

## Repository design

The `maldi_vae/` directory is a small local library containing the reusable
implementation. The `scripts/` directory contains the five scientific
experiments in execution order.
