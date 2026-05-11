import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """NT-Xent (Normalized Temperature-scaled Cross-Entropy) contrastive loss.

    This is the loss function from SimCLR. The idea:
    - z1[i] and z2[i] are a POSITIVE pair (two crops from the same sequence).
    - All other samples in the batch are NEGATIVES.
    - The loss pushes positive pairs closer and negatives apart in embedding space.

    How it works step by step:
    1. Concatenate z1 and z2 into one big batch of 2N vectors.
    2. Compute cosine similarity between ALL pairs (2N x 2N matrix).
    3. For each vector, its positive is the other crop of the same sample.
    4. Treat this as a classification problem: pick the positive out of all others.

    Example:
        >>> z1 = torch.randn(4, 32)  # batch=4, proj_dim=32 (L2-normalized)
        >>> z2 = torch.randn(4, 32)
        >>> z1 = F.normalize(z1, dim=-1)
        >>> z2 = F.normalize(z2, dim=-1)
        >>> loss = nt_xent_loss(z1, z2, temperature=0.07)
        >>> loss.shape
        torch.Size([])  # scalar

    Args:
        z1: L2-normalized embeddings from crop 1. Shape: (batch, proj_dim).
        z2: L2-normalized embeddings from crop 2. Shape: (batch, proj_dim).
        temperature: Scaling factor. Lower = sharper distribution = stricter
            separation between positives and negatives. Typical range: 0.05–0.5.

    Returns:
        Scalar loss value (lower = better alignment of positive pairs).
    """
    # N = batch size (number of original sequences)
    N = z1.shape[0]

    # Concatenate both crops into one tensor
    # z1 occupies indices 0..N-1, z2 occupies indices N..2N-1
    z = torch.cat([z1, z2], dim=0)
    # z shape: (2N, proj_dim)

    # Compute pairwise cosine similarity matrix, scaled by temperature
    # Since z is already L2-normalized, dot product = cosine similarity
    # (2N, proj_dim) @ (proj_dim, 2N) → (2N, 2N)
    sim = (z @ z.T) / temperature
    # sim[i, j] = cosine_similarity(z[i], z[j]) / temperature
    # sim shape: (2N, 2N)

    # Create a mask to exclude self-similarity (diagonal elements)
    # A sample should not be compared to itself
    # eye creates identity matrix (True on diagonal), ~ inverts it
    mask = ~torch.eye(2 * N, dtype=torch.bool, device=z.device)
    # mask shape: (2N, 2N) — False on diagonal, True elsewhere

    # Set diagonal to -inf so it becomes 0 after softmax (ignored)
    sim = sim.masked_fill(~mask, float("-inf"))
    # sim shape: (2N, 2N) — diagonal is -inf, rest are similarity scores

    # Define the correct positive for each sample (the "label" for cross-entropy)
    # For z1[i] (at index i), its positive is z2[i] (at index i+N)
    # For z2[i] (at index i+N), its positive is z1[i] (at index i)
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)], dim=0).to(z.device)
    # labels shape: (2N,)
    # labels = [N, N+1, ..., 2N-1, 0, 1, ..., N-1]

    # Cross-entropy loss: treats sim as logits and labels as the correct class
    # For each row in sim, it should assign highest probability to the positive pair
    return F.cross_entropy(sim, labels)
    # output: scalar — average loss over all 2N samples
