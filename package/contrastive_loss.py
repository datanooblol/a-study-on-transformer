import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.

    For each sample i, z1[i] and z2[i] are the positive pair.
    All other 2*(N-1) samples in the batch are negatives.

    Args:
        z1: (batch, proj_dim) — L2-normalized embeddings from crop 1.
        z2: (batch, proj_dim) — L2-normalized embeddings from crop 2.
        temperature: Scaling factor for similarity scores.

    Returns:
        Scalar loss.
    """
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2N, proj_dim)

    # Cosine similarity matrix (2N x 2N)
    sim = (z @ z.T) / temperature

    # Mask out self-similarity (diagonal)
    mask = ~torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(~mask, float("-inf"))

    # Positive pairs: (i, i+N) and (i+N, i)
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)], dim=0).to(z.device)

    return F.cross_entropy(sim, labels)
