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


def run_forecast(
    csv_path: Path,
    source_csv_path: Path,
    target_col: str,
    close_col: str,
    context_len: int,
    horizon: int,
    seed: int,
) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    df = df.select_dtypes(include=[np.number]).copy()
    if df.empty:
        raise ValueError("No numeric columns found. TimesFM covariates must be numeric.")

    target_col = resolve_target_column(df, target_col)

    required_rows = context_len + horizon
    if len(df) < required_rows:
        raise ValueError(
            f"Need at least {required_rows} rows, but found {len(df)} rows."
        )

    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, len(df) - required_rows + 1))
    start = 4200
    segment = df.iloc[start : start + required_rows].reset_index(drop=True)

    context_target = segment[target_col].iloc[:context_len].astype(float).to_numpy()
    true_future = segment[target_col].iloc[context_len : context_len + horizon].to_numpy()

    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=max(1024, context_len),
            max_horizon=max(128, horizon),
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            return_backcast=True,
        )
    )

    point_forecast, _ = model.forecast(
        horizon=horizon,
        inputs=[context_target],  # no covariates passed
    )
    pred = np.asarray(point_forecast[0], dtype=float)

    print(f"CSV: {csv_path}")
    print(f"Target column: {target_col}")
    print(f"Random segment start index: {start}")
    print(f"Context length: {context_len}")
    print(f"Forecast horizon: {horizon}")
    print(f"\nLast {horizon} context values:")
    print(
        np.array2string(
            context_target[-horizon:].astype(float), precision=6, separator=", "
        )
    )
    print(np.sum(context_target[-horizon:]))
    print("\nActual Delta values:")
    print(np.array2string(true_future.astype(float), precision=6, separator=", "))
    print(np.sum(true_future))
    print("\nPredicted Delta values (next 10):")
    print(np.array2string(pred, precision=6, separator=", "))
    print(np.sum(pred))

    expected_end_value = expected_end_value_from_source_close(
        source_csv_path=source_csv_path,
        predicted_delta=pred,
        close_col=close_col,
    )
    print(f"\nExpected end value from {source_csv_path} ({close_col}): {expected_end_value:.12f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Forecast 10 future Delta values from a random 500-row segment using TimesFM, "
            "with all columns as past dynamic covariates (including Delta)."
        )
    )
    parser.add_argument("--csv", type=Path, default=Path("target.csv"), help="Path to target CSV")
    parser.add_argument(
        "--source-csv", type=Path, default=Path("source.csv"), help="Path to source CSV with close values"
    )
    parser.add_argument("--target-col", type=str, default="close", help="Target column name")
    parser.add_argument("--close-col", type=str, default="close", help="Close column name in source CSV")
    parser.add_argument("--context-len", type=int, default=500, help="Input context length")
    parser.add_argument("--horizon", type=int, default=10, help="Forecast horizon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    run_forecast(
        csv_path=args.csv,
        source_csv_path=args.source_csv,
        target_col=args.target_col,
        close_col=args.close_col,
        context_len=args.context_len,
        horizon=args.horizon,
        seed=args.seed,
    )
