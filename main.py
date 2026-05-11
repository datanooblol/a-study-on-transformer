import random

import torch

from package import (
    ContrastiveTransformerEncoder,
    create_dataloader,
    make_collate_fn,
    nt_xent_loss,
)
from package.pad_and_mask import pad_and_mask

# --- Mock data: 100 sequences (vocab 1-50, length 5-20) ---
random.seed(42)
sequences = [
    [random.randint(1, 50) for _ in range(random.randint(5, 20))]
    for _ in range(100)
]

# --- Config ---
VOCAB_SIZE = 51  # 0 reserved for padding
BATCH_SIZE = 16
EPOCHS = 30
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Setup ---
collate_fn = make_collate_fn(min_len=3, max_len=10)
loader = create_dataloader(sequences, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
model = ContrastiveTransformerEncoder(vocab_size=VOCAB_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Train ---
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for x1, mask1, x2, mask2 in loader:
        x1, mask1 = x1.to(DEVICE), mask1.to(DEVICE)
        x2, mask2 = x2.to(DEVICE), mask2.to(DEVICE)

        z1 = model(x1, mask1)
        z2 = model(x2, mask2)
        loss = nt_xent_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss / len(loader):.4f}")

# --- Embed ---
model.eval()
padded, mask = pad_and_mask(sequences)
padded, mask = padded.to(DEVICE), mask.to(DEVICE)

with torch.no_grad():
    embeddings = model.encode(padded, mask)

print(f"Embeddings shape: {embeddings.shape}")  # (100, 64)
