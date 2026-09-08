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

Final values from the latest completed experiment are:

- PCA-FD real-real baseline, 11 vs 11: `0.000108073 ± 4.76338e-05`
- PCA-FD real-synthetic matched, 11 vs 11: `0.00040362 ± 4.70615e-05`
- PCA-FD real test-synthetic, 22 vs 22: `0.00037688 ± 4.53183e-05`
- Precision: `1.000000 ± 0.000000`
- Recall: `0.290909 ± 0.286840`
- Density: `1.26061 ± 0.0727693`
- Coverage: `0.345455 ± 0.0835397`

The five PCA components explain approximately `48.49%` of the variance in the
real training spectra. The matched real-vs-synthetic PCA-FD is substantially
larger than the real-vs-real reference, while precision is perfect and recall
and coverage are much lower. This supports the main project finding: generated
spectra lie in realistic regions of the real-data feature space, but the frozen
β=0.01 generator reproduces only part of the diversity present in the held-out
real spectra.

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

The DRIAMS dataset is not distributed with this repository.
(you can download it here:  https://datadryad.org/dataset/doi:10.5061/dryad.bzkh1899q )
Download DRIAMS-B separately and preserve the original structure then put it in scripts folder:

```text
DRIAMS-B/
├── raw/
│   └── 2018/
├── preprocessed/
│   └── 2018/
├── binned_6000/
│   └── 2018/
└── id/
    └── 2018/
        └── 2018_clean.csv
```

The repository scripts create additional local folders:

```text
DRIAMS-B/
├── preprocess_MaldiAMRKit/
├── binned_6000_MaldiAMRKit/
├── bin_comparison/
├── vae_data/
└── vae_results/
```

## Repository design

The `maldi_vae/` directory is a small local library containing the reusable
implementation. The `scripts/` directory contains the five scientific
experiments in execution order.
