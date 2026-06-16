from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


RAW_RESULTS = Path("results/stage3_etth1_h96/raw_results.csv")
SUMMARY_CSV = Path("results/stage3_etth1_h96/summary_mean_std_stderr.csv")
SUMMARY_MD = Path("results/stage3_etth1_h96/summary_mean_std_stderr.md")
IMPROVEMENT_TXT = Path("results/stage3_etth1_h96/baseline_best_improvement.txt")


def pick_column(columns: list[str], candidates: list[str]) -> str | None:
    by_lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        match = by_lower.get(candidate.lower())
        if match is not None:
            return match
    return None


def normalize_adapter(value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "baseline"
    text = str(value).strip()
    if text.lower() == "pca":
        return "PCA"
    return text


def wide_results(df: pd.DataFrame) -> pd.DataFrame:
    adapter_col = pick_column(list(df.columns), ["adapter", "adapter_name"])
    seed_col = pick_column(list(df.columns), ["seed"])
    metric_col = pick_column(list(df.columns), ["metric"])
    value_col = pick_column(list(df.columns), ["value"])

    if adapter_col is None or seed_col is None:
        raise ValueError("Could not identify adapter and seed columns.")

    df = df.copy()
    df["adapter_normalized"] = df[adapter_col].map(normalize_adapter)

    if metric_col and value_col:
        index_candidates = [
            "dataset",
            "dataset_name",
            "forecast_horizon",
            "horizon",
            "context_length",
            "foundational_model",
            "model_name",
            "adapter_normalized",
            seed_col,
        ]
        index_cols = [column for column in index_candidates if column in df.columns]
        wide = (
            df.pivot_table(
                index=index_cols,
                columns=metric_col,
                values=value_col,
                aggfunc="first",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
    else:
        wide = df.copy()

    if seed_col != "seed" and seed_col in wide.columns:
        wide = wide.rename(columns={seed_col: "seed"})
    if "adapter_normalized" in wide.columns:
        wide = wide.rename(columns={"adapter_normalized": "adapter"})
    elif adapter_col in wide.columns:
        wide["adapter"] = wide[adapter_col].map(normalize_adapter)
    return wide


def metric_column(df: pd.DataFrame, candidates: list[str]) -> str:
    column = pick_column(list(df.columns), candidates)
    if column is None:
        raise ValueError(
            f"Could not identify metric column from {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return column


def summarize(wide: pd.DataFrame) -> pd.DataFrame:
    mse_col = metric_column(wide, ["mse", "MSE", "test_mse", "scaled_mse"])
    mae_col = metric_column(wide, ["mae", "MAE", "test_mae", "scaled_mae"])

    rows = []
    order = ["baseline", "PCA", "linearAE", "dropoutLinearAE", "linearVAE", "VAE"]
    for adapter, group in wide.groupby("adapter", dropna=False):
        mse = pd.to_numeric(group[mse_col], errors="coerce").dropna()
        mae = pd.to_numeric(group[mae_col], errors="coerce").dropna()
        mse_std = float(mse.std(ddof=1)) if len(mse) > 1 else 0.0
        mae_std = float(mae.std(ddof=1)) if len(mae) > 1 else 0.0
        rows.append(
            {
                "adapter": adapter,
                "mse_mean": float(mse.mean()) if len(mse) else math.nan,
                "mse_std": mse_std,
                "mse_stderr": mse_std / math.sqrt(len(mse)) if len(mse) else math.nan,
                "mae_mean": float(mae.mean()) if len(mae) else math.nan,
                "mae_std": mae_std,
                "mae_stderr": mae_std / math.sqrt(len(mae)) if len(mae) else math.nan,
                "n": int(min(len(mse), len(mae))),
            }
        )

    summary = pd.DataFrame(rows)
    summary["adapter_order"] = summary["adapter"].map(
        {adapter: idx for idx, adapter in enumerate(order)}
    )
    return (
        summary.sort_values(["adapter_order", "adapter"], na_position="last")
        .drop(columns=["adapter_order"])
        .reset_index(drop=True)
    )


def markdown_table(summary: pd.DataFrame) -> str:
    display = summary.copy()
    for column in display.columns:
        if column not in {"adapter", "n"}:
            display[column] = display[column].map(lambda value: f"{value:.6f}")
    rows = [[str(value) for value in row] for row in display.to_numpy().tolist()]
    headers = list(display.columns)
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[idx]) for idx, value in enumerate(values)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render_row(headers), separator, *(render_row(row) for row in rows)])


def improvement_text(summary: pd.DataFrame) -> str:
    baseline = summary.loc[summary["adapter"] == "baseline"]
    if baseline.empty:
        return "baseline not found"
    baseline_mse = float(baseline["mse_mean"].iloc[0])
    best = summary.sort_values("mse_mean").iloc[0]
    best_adapter = str(best["adapter"])
    best_mse = float(best["mse_mean"])
    improvement = (baseline_mse - best_mse) / baseline_mse * 100
    return (
        f"baseline_mse_mean={baseline_mse:.6f}\n"
        f"best_adapter={best_adapter}\n"
        f"best_mse_mean={best_mse:.6f}\n"
        f"improvement_percent={improvement:.6f}\n"
    )


def main() -> None:
    if not RAW_RESULTS.exists():
        raise FileNotFoundError(RAW_RESULTS)

    df = pd.read_csv(RAW_RESULTS)
    print("Raw columns:", list(df.columns))
    print("Raw head:")
    print(df.head().to_string(index=False))

    wide = wide_results(df)
    print("\nWide columns:", list(wide.columns))
    print("Wide head:")
    print(wide.head().to_string(index=False))

    summary = summarize(wide)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    table = markdown_table(summary)
    SUMMARY_MD.write_text(table + "\n", encoding="utf-8")
    improvement = improvement_text(summary)
    IMPROVEMENT_TXT.write_text(improvement, encoding="utf-8")

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nMarkdown table:")
    print(table)
    print("\nImprovement:")
    print(improvement)
    print(f"Saved: {SUMMARY_CSV}")
    print(f"Saved: {SUMMARY_MD}")
    print(f"Saved: {IMPROVEMENT_TXT}")


if __name__ == "__main__":
    main()
