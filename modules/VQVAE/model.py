import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    VQ bottleneck with straight-through gradient estimator.

    Codebook entries are updated by gradient descent on the codebook loss
    ||sg(z_e) - z_q||^2.  The encoder is updated via the commitment loss
    ||z_e - sg(z_q)||^2 and, through the straight-through estimator, via the
    reconstruction loss.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight,
                         -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: (B, D, H, W) continuous encoder output
        Returns:
            z_q_st : (B, D, H, W) quantized, straight-through gradient
            vq_loss: scalar
            indices : (B, H, W) long  — token indices
        """
        B, D, H, W = z_e.shape

        # (B*H*W, D)
        z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)

        # Squared distances: ||z||^2 + ||e||^2 - 2 z·e
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                + self.embedding.weight.pow(2).sum(1)
                - 2.0 * z_flat @ self.embedding.weight.t())   # (B*H*W, A)

        indices_flat = dist.argmin(dim=1)                     # (B*H*W,)
        z_q_flat     = self.embedding(indices_flat)           # (B*H*W, D)
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)

        codebook_loss   = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through: copy z_q value, pass z_e gradient
        z_q_st  = z_e + (z_q - z_e).detach()
        indices = indices_flat.view(B, H, W)
        return z_q_st, vq_loss, indices

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Map token indices → embedding vectors.

        Args:
            indices: (B, H, W) or (B, K) long
        Returns:
            (B, D, H, W) float
        """
        if indices.dim() == 2:
            B, K = indices.shape
            H = W = int(K ** 0.5)
            indices = indices.view(B, H, W)
        B, H, W = indices.shape
        z_q = self.embedding(indices)              # (B, H, W, D)
        return z_q.permute(0, 3, 1, 2)            # (B, D, H, W)


class Encoder(nn.Module):
    """28×28 → 7×7 via two stride-2 convolutions."""

    def __init__(self, in_channels: int, hidden_channels: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, embedding_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """7×7 → 28×28 via two transposed convolutions."""

    def __init__(self, embedding_dim: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_channels * 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VQVAE(nn.Module):
    """
    VQ-VAE for 28×28 grayscale images (MNIST).

    Encodes to a 7×7 grid of discrete tokens from a codebook of size
    num_embeddings.  The full token sequence has K = 49 positions.

    Usage::

        model = VQVAE(num_embeddings=16)

        # Training
        x_hat, loss, info = model(x)          # x: (B,1,28,28)

        # Encoding
        indices = model.encode(x)             # (B, 7, 7)  long

        # Decoding
        x_hat  = model.decode(indices)        # (B, 1, 28, 28)  float
    """

    GRID = 7   # spatial grid size after encoding
    K    = 49  # total tokens per image (GRID ** 2)

    def __init__(self, in_channels: int = 1, hidden_channels: int = 32,
                 embedding_dim: int = 64, num_embeddings: int = 16,
                 commitment_cost: float = 0.25):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, embedding_dim)
        self.vq      = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_channels, in_channels)

        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self._hparams = dict(in_channels=in_channels,
                             hidden_channels=hidden_channels,
                             embedding_dim=embedding_dim,
                             num_embeddings=num_embeddings,
                             commitment_cost=commitment_cost)

    @classmethod
    def load(cls, checkpoint_path: str,
             map_location: str | torch.device = 'cpu') -> 'VQVAE':
        """Load a saved model from a checkpoint file.

        The checkpoint stores both the hyperparameters and the state dict,
        so no separate config is needed::

            model = VQVAE.load('mnist/checkpoints/best.pt')
            model.eval()
        """
        data  = torch.load(checkpoint_path, map_location=map_location,
                           weights_only=False)
        if isinstance(data, dict) and 'hparams' in data:
            model = cls(**data['hparams'])
            model.load_state_dict(data['state_dict'])
        else:
            # Legacy: plain state_dict saved by train_vqvae.py < v2.
            # Reconstruct with default hyperparameters.
            model = cls()
            model.load_state_dict(data)
        return model

    def save(self, path: str) -> None:
        """Save model with hyperparameters so it can be reloaded without config."""
        torch.save({'hparams': self._hparams,
                    'state_dict': self.state_dict()}, path)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            x_hat    : (B, 1, 28, 28)
            loss     : scalar — total loss (reconstruction + VQ)
            info     : dict with 'recon_loss' and 'vq_loss' (float)
        """
        z_e = self.encoder(x)
        z_q, vq_loss, _ = self.vq(z_e)
        x_hat = self.decoder(z_q)
        recon_loss = F.mse_loss(x_hat, x)
        loss = recon_loss + vq_loss
        return x_hat, loss, {'recon_loss': recon_loss.item(),
                              'vq_loss':   vq_loss.item()}

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to token indices.

        Args:
            x: (B, 1, 28, 28)
        Returns:
            (B, 7, 7) long
        """
        z_e = self.encoder(x)
        _, _, indices = self.vq(z_e)
        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices to images.

        Args:
            indices: (B, 7, 7) or (B, 49) long
        Returns:
            (B, 1, 28, 28) float in [0, 1]
        """
        z_q = self.vq.lookup(indices)
        return self.decoder(z_q)
