"""
Training script for HuggingFace TimeSeriesTransformerForPrediction
with PAST DYNAMIC REAL covariates.

Your data shape expectations:
  - target series  : (num_series, series_length)                   float32
  - past covariates: (num_series, series_length, num_covariates)   float32

Usage:
    uv run train_time_series_transformer.py
    uv run train_time_series_transformer.py --epochs 20 --num-covariates 4
"""

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from transformers import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    prediction_length: int = 12
    context_length: int = 48        # recommended >= 2x prediction_length
    input_size: int = 1             # target variates (usually 1)
    num_covariates: int = 3         # ← number of past dynamic real covariates
    num_series: int = 64
    series_length: int = 500

    # Model architecture
    d_model: int = 64
    encoder_layers: int = 2
    decoder_layers: int = 2
    encoder_attention_heads: int = 4
    decoder_attention_heads: int = 4
    encoder_ffn_dim: int = 128
    decoder_ffn_dim: int = 128
    dropout: float = 0.01
    lags_sequence: tuple = (1, 2, 4, 8, 12)

    # Training
    epochs: int = 10
    batch_size: int = 200
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    val_split: float = 0.1
    seed: int = 42

    # Paths
    output_dir: str = "checkpoints"
    best_model_name: str = "best_model.pt"


NUM_TIME_FEATURES = 4   # sin/cos for month + day-of-week


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data_from_csv(csv_path: str = "target.csv"):
    """
    Load a single-series dataset from CSV where:
      - `delta` is the target column
      - every other column is a covariate
    """
    df = pd.read_csv(csv_path)
    if "delta" not in df.columns:
        raise ValueError(f"`delta` column not found in {csv_path}")

    cov_cols = [c for c in df.columns if c != "delta"]
    if not cov_cols:
        raise ValueError("No covariate columns found. Add at least one non-`delta` column.")

    target_series = pd.to_numeric(df["delta"], errors="coerce")
    cov_df = df[cov_cols].apply(pd.to_numeric, errors="coerce")

    valid_rows = target_series.notna() & cov_df.notna().all(axis=1)
    target_series = target_series[valid_rows]
    cov_df = cov_df[valid_rows]

    if len(target_series) == 0:
        raise ValueError("No valid rows after numeric conversion/NaN filtering.")

    target = target_series.to_numpy(dtype=np.float32).reshape(1, -1)             # (1, T)
    covariates = cov_df.to_numpy(dtype=np.float32).reshape(1, -1, len(cov_cols))  # (1, T, C)

    # Normalise covariates.
    mean = covariates.mean(axis=(0, 1), keepdims=True)
    std = covariates.std(axis=(0, 1), keepdims=True) + 1e-8
    covariates = (covariates - mean) / std

    return target, covariates


def make_time_features(length: int) -> np.ndarray:
    """Cyclical encoding → shape (length, NUM_TIME_FEATURES)."""
    t = np.arange(length)
    return np.stack([
        np.sin(2 * math.pi * (t % 12) / 12),
        np.cos(2 * math.pi * (t % 12) / 12),
        np.sin(2 * math.pi * (t % 7)  / 7),
        np.cos(2 * math.pi * (t % 7)  / 7),
    ], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset.

    Keys returned per sample
    ─────────────────────────────────────────────────────────────────────────
    past_values            (context_length, 1)
    past_time_features     (context_length, NUM_TIME_FEATURES)
    past_observed_mask     (context_length, 1)
    past_dynamic_real      (context_length, num_covariates)   ← your covariates
    future_values          (prediction_length, 1)
    future_time_features   (prediction_length, NUM_TIME_FEATURES)
    """

    def __init__(self, target: np.ndarray, covariates: np.ndarray, cfg: TrainConfig):
        self.target     = target        # (N, T)
        self.covariates = covariates    # (N, T, C)
        self.cfg        = cfg
        T = target.shape[1]
        self.time_feats = make_time_features(T)

        total = cfg.context_length + cfg.prediction_length
        self.samples = [
            (i, end)
            for i in range(target.shape[0])
            for end in range(total, T + 1)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        si, end = self.samples[idx]
        cfg   = self.cfg
        start = end - (cfg.context_length + cfg.prediction_length)

        vals          = self.target[si, start:end]
        past_values   = vals[:cfg.context_length]
        future_values = vals[cfg.context_length:]

        feats        = self.time_feats[start:end]
        past_feats   = feats[:cfg.context_length]
        future_feats = feats[cfg.context_length:]

        # Covariates are only available for the past window
        past_cov = self.covariates[si, start : start + cfg.context_length]   # (ctx, C)

        observed = np.ones(cfg.context_length, dtype=np.float32)

        return {
            "past_values":          torch.from_numpy(past_values).unsqueeze(-1),
            "past_time_features":   torch.from_numpy(past_feats),
            "past_observed_mask":   torch.from_numpy(observed).unsqueeze(-1),
            "past_dynamic_real":    torch.from_numpy(past_cov.astype(np.float32)),
            "future_values":        torch.from_numpy(future_values).unsqueeze(-1),
            "future_time_features": torch.from_numpy(future_feats),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(cfg: TrainConfig) -> TimeSeriesTransformerForPrediction:
    model_cfg = TimeSeriesTransformerConfig(
        prediction_length=cfg.prediction_length,
        context_length=cfg.context_length,
        input_size=cfg.input_size,
        num_time_features=NUM_TIME_FEATURES,
        lags_sequence=list(cfg.lags_sequence),
        num_dynamic_real_features=cfg.num_covariates,   # ← key config line
        d_model=cfg.d_model,
        encoder_layers=cfg.encoder_layers,
        decoder_layers=cfg.decoder_layers,
        encoder_attention_heads=cfg.encoder_attention_heads,
        decoder_attention_heads=cfg.decoder_attention_heads,
        encoder_ffn_dim=cfg.encoder_ffn_dim,
        decoder_ffn_dim=cfg.decoder_ffn_dim,
        dropout=cfg.dropout,
        distribution_output="student_t",
    )
    return TimeSeriesTransformerForPrediction(model_cfg)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, optimizer, scheduler, device, grad_clip, train: bool) -> float:
    model.train(train)
    total_loss = 0.0

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                past_values=batch["past_values"],
                past_time_features=batch["past_time_features"],
                past_observed_mask=batch["past_observed_mask"],
                past_dynamic_real=batch["past_dynamic_real"],   # ← pass covariates
                future_values=batch["future_values"],
                future_time_features=batch["future_time_features"],
            )

            loss = outputs.loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler:
                    scheduler.step()

            total_loss += loss.item()

    return total_loss / len(loader)


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = get_device()
    print(f"Device      : {device}")
    # Data
    print("Loading data …")
    target, covariates = load_data_from_csv("target.csv")
    cfg.num_series = target.shape[0]
    cfg.series_length = target.shape[1]
    cfg.num_covariates = covariates.shape[-1]
    print(f"Series      : {cfg.num_series}")
    print(f"Length      : {cfg.series_length}")
    print(f"Covariates  : {cfg.num_covariates} past dynamic real feature(s)")

    if cfg.num_series > 1:
        n_val = max(1, int(cfg.num_series * cfg.val_split))
        train_ds = TimeSeriesDataset(target[n_val:], covariates[n_val:], cfg)
        val_ds = TimeSeriesDataset(target[:n_val], covariates[:n_val], cfg)
    else:
        # Single-series fallback: use the same series for train/val.
        train_ds = TimeSeriesDataset(target, covariates, cfg)
        val_ds = TimeSeriesDataset(target, covariates, cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    # Model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters  : {n_params:,}\n")

    # Optimiser
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=cfg.epochs,
    )

    # Loop
    os.makedirs(cfg.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_path = Path(cfg.output_dir) / cfg.best_model_name

    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'Best':>6}")
    print("-" * 44)

    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, device, cfg.grad_clip, train=True)
        val_loss   = run_epoch(model, val_loader,   None,      None,      device, cfg.grad_clip, train=False)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        print(f"{epoch:>6}  {train_loss:>12.4f}  {val_loss:>10.4f}  {'✓' if is_best else '':>6}")

    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Saved to      : {best_path}")

    # Inference demo
    print("\nRunning inference on one val batch …")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    sample = {k: v.to(device) for k, v in next(iter(val_loader)).items()}

    with torch.no_grad():
        forecasts = model.generate(
            past_values=sample["past_values"],
            past_time_features=sample["past_time_features"],
            past_observed_mask=sample["past_observed_mask"],
            past_dynamic_real=sample["past_dynamic_real"],      # ← covariates at inference
            future_time_features=sample["future_time_features"],
        )

    # forecasts.sequences: (batch, num_samples, prediction_length, input_size)
    median = forecasts.sequences.median(dim=1).values           # (batch, pred_len, 1)
    print(f"Forecast shape (batch, pred_len, variates): {tuple(median.shape)}")
    print(f"Example median forecast:\n  {median[0, :, 0].cpu().numpy()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train TimeSeriesTransformer with past covariates")
    p.add_argument("--epochs",            type=int,   default=10)
    p.add_argument("--batch-size",        type=int,   default=32)
    p.add_argument("--prediction-length", type=int,   default=12)
    p.add_argument("--context-length",    type=int,   default=48)
    p.add_argument("--lr",                type=float, default=1e-3)
    p.add_argument("--num-series",        type=int,   default=64)
    p.add_argument("--series-length",     type=int,   default=500)
    p.add_argument("--num-covariates",    type=int,   default=3,
                   help="Ignored when loading from target.csv; inferred from file columns.")
    p.add_argument("--output-dir",        type=str,   default="checkpoints")
    args = p.parse_args()

    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        learning_rate=args.lr,
        num_series=args.num_series,
        series_length=args.series_length,
        num_covariates=args.num_covariates,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
