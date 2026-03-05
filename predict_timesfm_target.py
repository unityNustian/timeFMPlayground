import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timesfm
import torch


def resolve_target_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested

    lower_map = {c.lower(): c for c in df.columns}
    if requested.lower() in lower_map:
        return lower_map[requested.lower()]

    raise ValueError(
        f"Target column '{requested}' was not found. Available columns: {list(df.columns)}"
    )


def expected_end_value_from_source_close(
    source_csv_path: Path, predicted_delta: np.ndarray, close_col: str = "close"
) -> float:
    source_df = pd.read_csv(source_csv_path)
    if source_df.empty:
        raise ValueError(f"Source CSV is empty: {source_csv_path}")

    close_col = resolve_target_column(source_df, close_col)
    close_values = pd.to_numeric(source_df[close_col], errors="coerce").dropna()
    if close_values.empty:
        raise ValueError(
            f"Close column '{close_col}' has no numeric values in {source_csv_path}"
        )

    last_close = float(close_values.iloc[-1])
    predicted_delta = np.asarray(predicted_delta, dtype=float)
    growth_factor = float(np.prod(1.0 + predicted_delta))
    return last_close * growth_factor


def run_prediction(
    csv_path: Path,
    source_csv_path: Path,
    target_col: str,
    close_col: str,
    context_len: int,
    predict_len: int,
    seed: int,
) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    df = df.select_dtypes(include=[np.number]).copy()
    if df.empty:
        raise ValueError("No numeric columns found.")

    target_col = resolve_target_column(df, target_col)

    if context_len <= 0:
        raise ValueError("--context-len must be > 0")
    if predict_len <= 0:
        raise ValueError("--predict-len must be > 0")
    if len(df) < context_len:
        raise ValueError(f"Need at least {context_len} rows, but found {len(df)} rows.")

    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, len(df) - context_len + 1))
    segment = df.iloc[start : start + context_len].reset_index(drop=True)
    context_target = segment[target_col].astype(float).to_numpy()

    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=max(1024, context_len),
            max_horizon=max(128, predict_len),
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            return_backcast=True,
        )
    )

    point_forecast, _ = model.forecast(
        horizon=predict_len,
        inputs=[context_target],
    )
    pred = np.asarray(point_forecast[0], dtype=float)

    print(f"CSV: {csv_path}")
    print(f"Target column: {target_col}")
    print(f"Random context start index: {start}")
    print(f"Context length: {context_len}")
    print(f"Prediction length: {predict_len}")
    print("\nPredicted Delta values:")
    print(np.array2string(pred, precision=12, separator=", "))
    print(np.sum(pred) * 100)

    expected_end_value = expected_end_value_from_source_close(
        source_csv_path=source_csv_path,
        predicted_delta=pred,
        close_col=close_col,
    )
    print(f"\nExpected end value from {source_csv_path} ({close_col}): {expected_end_value:.12f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict future Delta values from historical context using TimesFM."
    )
    parser.add_argument("--csv", type=Path, default=Path("target.csv"), help="Path to target CSV")
    parser.add_argument(
        "--source-csv", type=Path, default=Path("source.csv"), help="Path to source CSV with close values"
    )
    parser.add_argument("--target-col", type=str, default="delta", help="Target column name")
    parser.add_argument("--close-col", type=str, default="close", help="Close column name in source CSV")
    parser.add_argument("--context-len", type=int, default=500, help="Input context length")
    parser.add_argument("--predict-len", type=int, default=10, help="Number of values to predict")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    run_prediction(
        csv_path=args.csv,
        source_csv_path=args.source_csv,
        target_col=args.target_col,
        close_col=args.close_col,
        context_len=args.context_len,
        predict_len=args.predict_len,
        seed=args.seed,
    )
