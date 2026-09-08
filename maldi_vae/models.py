"""Neural-network architectures used in the project."""

import torch
import torch.nn as nn


class DenseBetaVAE(nn.Module):
    """Dense beta-VAE for 6000-bin non-negative MALDI spectra."""

    def __init__(
        self,
        input_dim=6000,
        hidden_1=512,
        hidden_2=128,
        hidden_3=64,
        latent_dim=16,
        dropout=0.10,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, hidden_3),
            nn.GELU(),
        )

        self.mu_layer = nn.Linear(hidden_3, latent_dim)
        self.logvar_layer = nn.Linear(hidden_3, latent_dim)

        self.decoder_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_3),
            nn.GELU(),
            nn.Linear(hidden_3, hidden_2),
            nn.GELU(),
            nn.Linear(hidden_2, hidden_1),
            nn.GELU(),
        )
        self.decoder_output = nn.Linear(hidden_1, input_dim)
        self.output_activation = nn.Softplus()

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z):
        return self.output_activation(self.decoder_output(self.decoder_hidden(z)))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def reconstruct_from_mean(self, x):
        """Deterministic reconstruction using z = posterior mean."""
        mu, _ = self.encode(x)
        return self.decode(mu)
