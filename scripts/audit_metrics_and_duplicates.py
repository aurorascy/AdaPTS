from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("results/code_logic_audit")
STAGE7_DIR = Path("results/stage7_remaining_table1_with_baseline_types")
RAW_FILES = [
    ("stage7", STAGE7_DIR / "raw_results.csv"),
    ("stage2_illness_h24", Path("results/stage2_illness_h24/raw_results.csv")),
    ("stage3_etth1_h96", Path("results/stage3_etth1_h96/raw_results.csv")),
]
NAMED_ADAPTERS = {"linearAE", "dropoutLinearAE", "linearVAE", "VAE"}


def to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def clean_adapter(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value)
    return "" if value.lower() in {"nan", "none"} else value


def as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def normalize(stage: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if stage == "stage7":
        df["source_stage_for_audit"] = "stage7"
        return df

    rows = []
    for _, row in df.iterrows():
        adapter = clean_adapter(row.get("adapter", ""))
        use_revin = as_bool(row.get("use_revin", False))
        if adapter == "" and use_revin:
            protocol_type = "strong-baseline"
            adapter_label = "strong-baseline"
        elif adapter in NAMED_ADAPTERS:
            protocol_type = "adapter"
            adapter_label = adapter
        else:
            protocol_type = "unsupported_or_legacy"
            adapter_label = adapter or "unknown"
        rows.append(
            {
                "dataset": row.get("dataset"),
                "horizon": row.get("forecast_horizon"),
                "context_length": row.get("context_length"),
                "seed": row.get("seed"),
                "protocol_type": protocol_type,
                "adapter_label": adapter_label,
                "adapter": adapter,
                "use_revin": use_revin,
                "supervised": row.get("is_fine_tuned"),
                "n_components": row.get("n_components"),
                "metric": row.get("metric"),
                "value": row.get("value"),
                "source_stage_for_audit": stage,
            }
        )
    return pd.DataFrame(rows)


def load_all() -> pd.DataFrame:
    frames = [normalize(stage, path) for stage, path in RAW_FILES]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    return df


def stderr(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) <= 1:
        return 0.0 if len(values) == 1 else np.nan
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def recompute_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "horizon", "protocol_type", "adapter_label"]
    rows = []
    filtered = df[df["metric"].isin(["mse", "mae"])].copy()
    for key, group in filtered.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        for metric in ["mse", "mae"]:
            vals = group[group["metric"] == metric]["value"].dropna()
            row[f"{metric.upper()} mean"] = vals.mean() if len(vals) else np.nan
            row[f"{metric.upper()} std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0 if len(vals) == 1 else np.nan
            row[f"{metric.upper()} stderr"] = stderr(vals)
        row["n"] = group["seed"].dropna().nunique()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all()
    if df.empty:
        raise FileNotFoundError("No raw result files found")

    key_cols = ["source_stage_for_audit", "dataset", "horizon", "seed", "protocol_type", "adapter_label", "metric"]
    dup = df.groupby(key_cols, dropna=False).size().reset_index(name="count")
    dup = dup[dup["count"] > 1].sort_values(key_cols)
    dup.to_csv(OUT_DIR / "metric_duplicate_audit.csv", index=False)

    recomputed = recompute_summary(df)
    current_path = STAGE7_DIR / "summary_by_dataset_horizon_protocol.csv"
    checks = []
    if current_path.exists():
        current = pd.read_csv(current_path)
        merged = recomputed.merge(
            current,
            on=["dataset", "horizon", "protocol_type", "adapter_label"],
            how="outer",
            suffixes=("_recomputed", "_current"),
            indicator=True,
        )
        numeric_cols = ["MSE mean", "MSE std", "MSE stderr", "MAE mean", "MAE std", "MAE stderr", "n"]
        for _, row in merged.iterrows():
            max_abs_diff = 0.0
            diffs = {}
            for col in numeric_cols:
                left = row.get(f"{col}_recomputed")
                right = row.get(f"{col}_current")
                if pd.isna(left) and pd.isna(right):
                    diff = 0.0
                else:
                    diff = abs(float(left) - float(right)) if pd.notna(left) and pd.notna(right) else np.inf
                diffs[col] = diff
                if diff != np.inf:
                    max_abs_diff = max(max_abs_diff, diff)
            checks.append(
                {
                    "dataset": row.get("dataset"),
                    "horizon": row.get("horizon"),
                    "protocol_type": row.get("protocol_type"),
                    "adapter_label": row.get("adapter_label"),
                    "merge_status": row.get("_merge"),
                    "max_abs_diff": max_abs_diff,
                    "matches": row.get("_merge") == "both" and max_abs_diff < 1e-10,
                    "diffs": diffs,
                }
            )
    else:
        checks.append({"merge_status": "missing_current_summary", "matches": False})
    check_df = pd.DataFrame(checks)
    check_df.to_csv(OUT_DIR / "summary_recompute_check.csv", index=False)

    metric_counts = df.groupby(["source_stage_for_audit", "metric"], dropna=False).size().reset_index(name="rows")
    report = [
        "# Metric and Duplicate Audit",
        "",
        "This audit recomputes summaries from raw result rows using only `metric == 'mse'` and `metric == 'mae'`.",
        "",
        f"- Raw rows inspected: {len(df)}",
        f"- Duplicate seed/protocol/metric groups: {len(dup)}",
        f"- Current summary comparison rows: {len(check_df)}",
        f"- All recomputed summary rows match current summary: {bool(check_df['matches'].all()) if 'matches' in check_df else False}",
        "",
        "## Metric Counts",
        "",
        to_md(metric_counts),
        "",
        "## Duplicate Rows",
        "",
        to_md(dup) if not dup.empty else "No duplicate rows at the audited key level.",
    ]
    (OUT_DIR / "metric_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(f"saved: {OUT_DIR / 'metric_duplicate_audit.csv'}")
    print(f"saved: {OUT_DIR / 'summary_recompute_check.csv'}")
    print(f"saved: {OUT_DIR / 'metric_audit.md'}")
    print(f"summary matches: {bool(check_df['matches'].all()) if 'matches' in check_df else False}")


if __name__ == "__main__":
    main()
