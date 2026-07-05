# CLAUDE.md

## What this repo is

A from-scratch study of transformer architecture, built to understand the mechanics deeply enough to apply them **outside language modeling**. The running example domain is insurance customer data (purchases, claims, payments) instead of text — the point is to prove the same blocks (attention, masking, positional encoding) generalize to any tokenizable sequence.

The full study curriculum lives in **`lessons/index.html`** — open it in a browser to (re)learn this material end to end (datasets → network → train → validate → deploy, plus non-insurance examples). This file is instructions for Claude, not the lesson content.

## Structure

- `package/` — the building blocks, each file mostly self-contained (see Conventions below for why RoPE/FeedForward are copy-pasted across files rather than shared):
  - `encoder_block.py` — bidirectional self-attention (BERT-style)
  - `decoder_block.py` — causal self-attention + cross-attention (original encoder-decoder transformer)
  - `gpt_decoder_block.py` — causal self-attention only, no cross-attention (GPT-style)
  - `insurance_decoder_block.py` — `gpt_decoder_block.py` + multi-feature event embedding + 3 prediction heads
  - `data_loader.py`, `collate.py`, `pad_and_mask.py` — variable-length sequence → padded batch pipeline
  - `contrastive_loss.py`, `transformer.py` — SimCLR-style NT-Xent contrastive pretraining (`ContrastiveTransformerEncoder`)
  - `rope.py` — standalone RoPE used by `transformer.py` only (see mask-convention note below)
- `train_insurance.py` — decoder project: predict a customer's next purchase (policy/age/price) from their history
- `train_insurance_representation_for_claim.py` — encoder project: predict next-year claim amount from full purchase/claim/payment history via CLS-pooled embeddings
- `main.py` — contrastive pretraining entry point + UMAP embedding visualization
- `README.md`, `CLS_and_PAD.md` — existing deep-dive write-ups (kept as reference; don't duplicate their content elsewhere, link to them)
- `lessons/` — the visual study curriculum (HTML, one topic per file, start at `index.html`)

## Conventions and gotchas

- **Two padding-mask conventions coexist — do not mix them:**
  - `package/encoder_block.py`, `decoder_block.py`, `gpt_decoder_block.py`, `insurance_decoder_block.py`: the `mask`/`pad_mask` argument means **`True` = PAD** (masked directly via `masked_fill(mask, -inf)`).
  - `package/transformer.py`'s `RoPEAttention` (used by `ContrastiveTransformerEncoder`, i.e. the `main.py` pipeline): the `mask` argument means **`True` = real token** (inverted internally with `~mask`), matching what `pad_and_mask.py` returns.
  - When wiring a new dataset into either family, check which convention that entry point expects.
- `padding_idx=0` is reserved for PAD across every vocab (`policy_to_id`, `purchase_to_id`, etc. are 1-indexed for this reason).
- RoPE rotates Q and K only, never V.
- `RoPE`/`FeedForward` are intentionally re-implemented per file rather than imported from one shared module — this is a deliberate readability tradeoff for a study repo (each block file is self-contained and diffable against the others), not an oversight. A production codebase would share one module instead.
- `Normalizer` (in both `train_insurance*.py` files) must be `.fit()` once on the training set and reused as-is for `.transform()`/`.inverse_transform()` everywhere else, including at inference — refitting per-batch or per-call will silently corrupt predictions.
- None of the training scripts currently hold out a validation split — see `lessons/08-validation-and-deployment.html` before adding new training code, so validation is added consistently.

## Running things

```bash
python train_insurance.py                              # decoder: next-purchase prediction
python train_insurance_representation_for_claim.py      # encoder: claim-amount regression
python main.py                                          # contrastive pretraining + UMAP plot (needs umap-learn, plotly, pandas)
```
