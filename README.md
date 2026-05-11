# A Study on Transformer

A study of transformer architecture applied to customer interaction sequences.

## Goal

Use a transformer encoder with contrastive learning to produce embeddings for interaction sequences, then cluster them.

## Pipeline

1. **Data Loading** (`package/data_loader.py`) — PyTorch `Dataset` + `DataLoader` for interaction sequences
2. **Contrastive Cropping** (`package/collate.py`) — Crop two random subsequences (positive pair) from each sample while preserving order
3. **Padding & Masking** (`package/pad_and_mask.py`) — Pad variable-length sequences and produce attention masks
4. **Transformer Encoder** — (TBD) Encode padded sequences into fixed-size embeddings
5. **Clustering** — (TBD) Cluster the learned embeddings
