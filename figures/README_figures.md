# Example Figures

This folder contains a small set of representative figures from the project.  
Complete numerical results are reported in the corresponding CSV files under `results/`.

## PCA: real vs synthetic spectra

![PCA real vs synthetic](pca_real_vs_tic_synthetic.png)

The TIC-normalized synthetic spectra overlap with the real-data distribution in the training-fitted PCA space, but they are much more concentrated in a restricted region. This visually supports the quantitative finding of **high fidelity but reduced diversity**.

## Generative precision, recall, density, and coverage

![PRDC metrics](prdc_real_real_vs_real_synthetic.png)

The synthetic spectra achieve **very high precision (1.00)** and high density, showing that generated samples lie in realistic regions of the real-data manifold. In contrast, the lower **recall (0.291)** and **coverage (0.345)** indicate that the generator does not reproduce the full diversity of the held-out real spectra.

## Example VAE reconstructions

![Reconstruction example 1](reconstruction_01.png)

![Reconstruction example 2](reconstruction_02.png)

These examples show that the VAE reconstructs the main MALDI-TOF peak locations and overall spectral structure well, while some peak intensities and smaller spectral details are smoothed or underestimated.

## Mean real vs synthetic spectrum

![Mean spectrum comparison](tic_mean_spectrum_real_vs_synthetic.png)

The mean synthetic spectrum reproduces many of the characteristic peak locations of the real data, but differences remain in peak amplitudes and background intensity. This is consistent with the distribution-level evaluation showing realistic synthetic spectra with reduced population-level coverage.
