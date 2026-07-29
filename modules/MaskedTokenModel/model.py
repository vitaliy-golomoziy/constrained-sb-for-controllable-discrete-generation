import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedTokenModel(nn.Module):
    """
    Small bidirectional transformer trained with masked-language-model (MLM)
    objective on discrete token sequences.

    Acts as the reference Gibbs kernel p_R(x_k | x_{-k}): mask token k,
    run the model, read off the output distribution at position k.

    Special tokens
    --------------
    MASK  = vocab_size  (appended above the VQ-VAE codebook)

    The output head predicts over the original vocab_size entries only
    (not the MASK token itself).

    Usage::

        model = MaskedTokenModel(vocab_size=16, seq_len=49)

        # --- training ---
        tokens  = ...                              # (B, K) long in [0, vocab_size)
        logits, loss = model.mlm_loss(tokens)      # standard MLM forward

        # --- Gibbs conditional p(x_k | x_{-k}) ---
        probs = model.conditional(tokens, k=5)     # (B, vocab_size) float

        # --- one full Gibbs sweep (updates every position once in random order) ---
        new_tokens = model.gibbs_sweep(tokens)     # (B, K) long

        # --- conditional entropy estimate H(p_R) for a batch ---
        H = model.conditional_entropy(tokens)      # (B,) float
    """

    def __init__(self, vocab_size: int = 16, seq_len: int = 49,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 dropout: float = 0.1, mask_prob: float = 0.15):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.MASK       = vocab_size          # mask token id
        self.mask_prob  = mask_prob

        self.token_emb = nn.Embedding(vocab_size + 1, d_model)   # +1 for MASK
        self.pos_emb   = nn.Embedding(seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)
        nn.init.normal_(self.head.weight,       std=0.02)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, K) long — may contain MASK tokens
        Returns:
            logits: (B, K, vocab_size) float
        """
        B, K = tokens.shape
        pos  = torch.arange(K, device=tokens.device).unsqueeze(0)
        h    = self.token_emb(tokens) + self.pos_emb(pos)
        h    = self.transformer(h)
        h    = self.norm(h)
        return self.head(h)                         # (B, K, vocab_size)

    # ------------------------------------------------------------------
    def mlm_loss(self, tokens: torch.Tensor,
                 mask_prob: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Random-mask MLM loss.

        Args:
            tokens   : (B, K) long in [0, vocab_size)
            mask_prob: fraction of positions to mask (default: self.mask_prob)
        Returns:
            logits: (B, K, vocab_size)
            loss  : scalar cross-entropy over masked positions
        """
        if mask_prob is None:
            mask_prob = self.mask_prob
        B, K      = tokens.shape
        mask      = torch.rand(B, K, device=tokens.device) < mask_prob
        if not mask.any():
            mask[0, 0] = True                       # guarantee at least one masked

        masked_tokens        = tokens.clone()
        masked_tokens[mask]  = self.MASK

        logits = self.forward(masked_tokens)        # (B, K, vocab_size)
        loss   = F.cross_entropy(
            logits[mask],                           # (M, vocab_size)
            tokens[mask],                           # (M,)
        )
        return logits, loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def conditional(self, tokens: torch.Tensor, k: int) -> torch.Tensor:
        """Compute p(x_k | x_{-k}) for every sequence in the batch.

        Args:
            tokens: (B, K) long
            k     : position to condition on
        Returns:
            probs: (B, vocab_size) float — probability distribution over vocab
        """
        masked        = tokens.clone()
        masked[:, k]  = self.MASK
        logits        = self.forward(masked)        # (B, K, vocab_size)
        return F.softmax(logits[:, k, :], dim=-1)   # (B, vocab_size)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def gibbs_sweep(self, tokens: torch.Tensor,
                    temperature: float = 1.0) -> torch.Tensor:
        """One full Gibbs sweep: update every position once in random order.

        Args:
            tokens     : (B, K) long
            temperature: > 1 flattens distribution (higher entropy),
                         < 1 sharpens it
        Returns:
            updated tokens (B, K) long — in-place modification of a clone
        """
        tokens = tokens.clone()
        B, K   = tokens.shape
        order  = torch.randperm(K, device=tokens.device)
        for k in order:
            k      = k.item()
            masked = tokens.clone()
            masked[:, k] = self.MASK
            logits = self.forward(masked)[:, k, :]   # (B, vocab_size)
            if temperature != 1.0:
                logits = logits / temperature
            probs  = F.softmax(logits, dim=-1)
            tokens[:, k] = torch.multinomial(probs, 1).squeeze(1)
        return tokens

    # ------------------------------------------------------------------
    @torch.no_grad()
    def conditional_entropy(self, tokens: torch.Tensor) -> torch.Tensor:
        """Estimate H(p_R) = (1/K) sum_k H(x_k | x_{-k}) for each sequence.

        Masks each position in turn and reads off the model's entropy at
        that position.  Requires K forward passes — batched over positions
        to stay within memory limits.

        Args:
            tokens: (B, K) long
        Returns:
            H: (B,) float — per-sequence conditional entropy in nats
        """
        B, K    = tokens.shape
        H_total = tokens.new_zeros(B, dtype=torch.float)

        for k in range(K):
            masked       = tokens.clone()
            masked[:, k] = self.MASK
            logits       = self.forward(masked)[:, k, :]      # (B, vocab_size)
            log_probs    = F.log_softmax(logits, dim=-1)       # (B, vocab_size)
            probs        = log_probs.exp()
            H_k          = -(probs * log_probs).sum(dim=-1)   # (B,)
            H_total     += H_k

        return H_total / K
