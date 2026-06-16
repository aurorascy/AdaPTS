from pathlib import Path

import numpy as np
import pandas as pd


RESULT_DIR = Path("results/stage7_remaining_table1_with_baseline_types")
STAGE7_RAW = RESULT_DIR / "raw_results.csv"
OLD_RAW_FILES = [
    ("stage2_illness_h24", Path("results/stage2_illness_h24/raw_results.csv")),
    ("stage3_etth1_h96", Path("results/stage3_etth1_h96/raw_results.csv")),
]

SUMMARY_LONG = RESULT_DIR / "summary_by_dataset_horizon_protocol.csv"
SUMMARY_TABLE = RESULT_DIR / "summary_table1_style.csv"
BEST_BY_TASK = RESULT_DIR / "best_by_task.csv"

NAMED_ADAPTERS = {"linearAE", "dropoutLinearAE", "linearVAE", "VAE"}
TABLE_LABELS = ["baseline", "strong-baseline", "linearAE", "dropoutLinearAE", "linearVAE", "VAE", "PCA"]


def stderr(series: pd.Series) -> float:
    n = series.count()
    if n == 0:
        return np.nan
    return float(series.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0


def normalize_stage7(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["source_stage"] = df.get("source_stage", "stage7")
    return df


def normalize_old(stage: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    rows = []
    for _, row in raw.iterrows():
        adapter = "" if pd.isna(row.get("adapter")) else str(row.get("adapter"))
        use_revin = str(row.get("use_revin")).lower() == "true"
        if adapter == "" and use_revin:
            protocol_type = "strong-baseline"
            adapter_label = "strong-baseline"
        elif adapter in NAMED_ADAPTERS:
            protocol_type = "adapter"
            adapter_label = adapter
        else:
            protocol_type = "pca_or_unsupported"
            adapter_label = adapter or "unknown"
        rows.append(
            {
                "stage": "stage7_imported_previous_results",
                "dataset": row.get("dataset"),
                "horizon": row.get("forecast_horizon"),
                "context_length": row.get("context_length"),
                "seed": row.get("seed"),
                "protocol_type": protocol_type,
                "adapter_label": adapter_label,
                "adapter": adapter,
                "use_revin": use_revin,
                "supervised": row.get("is_fine_tuned"),
                "n_epochs_fine_tuning": np.nan,
                "n_epochs_adapter": np.nan,
                "n_components": row.get("n_components"),
                "metric": row.get("metric"),
                "value": row.get("value"),
                "notes": "imported from previous stage; old empty-adapter baseline relabeled as strong-baseline",
                "source_stage": stage,
                "source": str(path),
                "model_name": row.get("foundational_model"),
                "device": "",
                "n_features": row.get("n_features"),
                "train_size": row.get("train_size"),
                "running_time": row.get("running_time"),
                "pca_in_preprocessing": row.get("pca_in_preprocessing"),
                "is_fine_tuned": row.get("is_fine_tuned"),
            }
        )
    return pd.DataFrame(rows)


def load_all() -> pd.DataFrame:
    parts = [normalize_stage7(STAGE7_RAW)]
    for stage, path in OLD_RAW_FILES:
        parts.append(normalize_old(stage, path))
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True, sort=False)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").astype("Int64")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    return df


def build_long_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = df[df["metric"].isin(["mse", "mae"])].copy()
    grouped = []
    keys = ["dataset", "horizon", "protocol_type", "adapter_label"]
    for key, group in metrics.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        for metric in ["mse", "mae"]:
            vals = group[group["metric"] == metric]["value"].dropna()
            row[f"{metric.upper()} mean"] = vals.mean() if len(vals) else np.nan
            row[f"{metric.upper()} std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0 if len(vals) == 1 else np.nan
            row[f"{metric.upper()} stderr"] = stderr(vals)
        seeds = group["seed"].dropna().unique()
        row["n"] = len(seeds)
        grouped.append(row)
    return pd.DataFrame(grouped).sort_values(["dataset", "horizon", "protocol_type", "adapter_label"])


def build_table1(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, horizon), group in summary.groupby(["dataset", "horizon"], dropna=False):
        row = {"Dataset": dataset, "H": horizon}
        for label in TABLE_LABELS:
            if label == "baseline":
                match = group[(group["protocol_type"] == "baseline") & (group["adapter_label"].str.contains("baseline", na=False))]
            elif label == "strong-baseline":
                match = group[(group["protocol_type"] == "strong-baseline") & (group["adapter_label"] == "strong-baseline")]
            elif label == "PCA":
                row[label] = "unsupported"
                continue
            else:
                match = group[(group["protocol_type"] == "adapter") & (group["adapter_label"] == label)]
            if len(match):
                value = match.iloc[0]["MSE mean"]
                std = match.iloc[0]["MSE std"]
                n = match.iloc[0]["n"]
                row[label] = f"{value:.6f} ± {std:.6f} (n={int(n)})"
            else:
                row[label] = "N/A"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Dataset", "H"])


def build_best(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, horizon), group in summary.groupby(["dataset", "horizon"], dropna=False):
        adapters = group[group["protocol_type"] == "adapter"].copy()
        best_adapter = None
        best_mse = np.nan
        if not adapters.empty:
            best = adapters.sort_values("MSE mean").iloc[0]
            best_adapter = best["adapter_label"]
            best_mse = best["MSE mean"]
        strong = group[(group["protocol_type"] == "strong-baseline") & (group["adapter_label"] == "strong-baseline")]
        baseline = group[group["protocol_type"] == "baseline"]
        strong_mse = strong.iloc[0]["MSE mean"] if len(strong) else np.nan
        baseline_mse = baseline.iloc[0]["MSE mean"] if len(baseline) else np.nan
        improve_vs_strong = (
            (strong_mse - best_mse) / strong_mse * 100 if pd.notna(strong_mse) and pd.notna(best_mse) else np.nan
        )
        improve_vs_baseline = (
            (baseline_mse - best_mse) / baseline_mse * 100 if pd.notna(baseline_mse) and pd.notna(best_mse) else np.nan
        )
        rows.append(
            {
                "Dataset": dataset,
                "Horizon": horizon,
                "best_adapter": best_adapter or "N/A",
                "best_adapter_mse": best_mse,
                "strong_baseline_mse": strong_mse,
                "improvement_vs_strong_baseline_percent": improve_vs_strong,
                "baseline_mse": baseline_mse if pd.notna(baseline_mse) else "baseline unavailable",
                "improvement_vs_baseline_percent": improve_vs_baseline if pd.notna(improve_vs_baseline) else "baseline unavailable",
            }
        )
    return pd.DataFrame(rows).sort_values(["Dataset", "Horizon"])


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all()
    if df.empty:
        raise FileNotFoundError("No stage7 or previous raw results found")
    print("combined rows:", len(df))
    print("sources:", sorted(df["source_stage"].dropna().unique()))
    summary = build_long_summary(df)
    table = build_table1(summary)
    best = build_best(summary)
    summary.to_csv(SUMMARY_LONG, index=False)
    table.to_csv(SUMMARY_TABLE, index=False)
    best.to_csv(BEST_BY_TASK, index=False)
    print("saved:", SUMMARY_LONG)
    print(summary.to_string(index=False))
    print("saved:", SUMMARY_TABLE)
    print(table.to_string(index=False))
    print("saved:", BEST_BY_TASK)
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
