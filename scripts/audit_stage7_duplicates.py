from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("results/stage9_targeted_code_verification")
STAGE7_RAW = Path("results/stage7_remaining_table1_with_baseline_types/raw_results.csv")
STAGE7_SUMMARY = Path("results/stage7_remaining_table1_with_baseline_types/summary_by_dataset_horizon_protocol.csv")
OLD_RAW = [
    ("stage2_illness_h24", Path("results/stage2_illness_h24/raw_results.csv")),
    ("stage3_etth1_h96", Path("results/stage3_etth1_h96/raw_results.csv")),
]
NAMED_ADAPTERS = {"linearAE", "dropoutLinearAE", "linearVAE", "VAE"}


def clean_adapter(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value)
    return "" if value.lower() in {"nan", "none"} else value


def old_protocol(row: pd.Series) -> tuple[str, str]:
    adapter = clean_adapter(row.get("adapter", ""))
    use_revin = str(row.get("use_revin", "")).lower() == "true"
    if adapter == "" and use_revin:
        return "strong-baseline", "strong-baseline"
    if adapter in NAMED_ADAPTERS:
        return "adapter", adapter
    return "unsupported_or_legacy", adapter or "unknown"


def normalize_old(stage: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    rows = []
    for _, row in raw.iterrows():
        protocol_type, adapter_label = old_protocol(row)
        rows.append(
            {
                "source_stage": stage,
                "dataset": row.get("dataset"),
                "horizon": row.get("forecast_horizon"),
                "seed": row.get("seed"),
                "protocol_type": protocol_type,
                "adapter_label": adapter_label,
                "metric": row.get("metric"),
                "value": row.get("value"),
            }
        )
    return pd.DataFrame(rows)


def recompute_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["metric"].isin(["mse", "mae"])].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    rows = []
    for key, group in df.groupby(["dataset", "horizon", "protocol_type", "adapter_label"], dropna=False):
        row = dict(zip(["dataset", "horizon", "protocol_type", "adapter_label"], key))
        for metric in ["mse", "mae"]:
            vals = group[group.metric.eq(metric)]["value"].dropna()
            row[f"{metric.upper()} mean"] = vals.mean()
            row[f"{metric.upper()} std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
            row[f"{metric.upper()} stderr"] = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        row["n"] = group["seed"].nunique()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "horizon", "protocol_type", "adapter_label"]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not STAGE7_RAW.exists():
        raise FileNotFoundError(STAGE7_RAW)
    stage7 = pd.read_csv(STAGE7_RAW)
    stage7["source_stage"] = "stage7"
    key = ["dataset", "horizon", "seed", "protocol_type", "adapter_label", "metric"]
    dup = stage7.groupby(key, dropna=False).size().reset_index(name="count")
    dup = dup[dup["count"] > 1]
    dup.to_csv(OUT_DIR / "stage7_duplicate_audit.csv", index=False)

    old = [normalize_old(stage, path) for stage, path in OLD_RAW]
    combined = pd.concat([stage7[["source_stage", "dataset", "horizon", "seed", "protocol_type", "adapter_label", "metric", "value"]]] + old, ignore_index=True, sort=False)
    recomputed = recompute_summary(combined)
    current = pd.read_csv(STAGE7_SUMMARY) if STAGE7_SUMMARY.exists() else pd.DataFrame()
    matches = False
    if not current.empty:
        merged = recomputed.merge(current, on=["dataset", "horizon", "protocol_type", "adapter_label"], suffixes=("_recomputed", "_current"), how="outer", indicator=True)
        diffs = []
        for _, row in merged.iterrows():
            max_diff = 0.0
            for col in ["MSE mean", "MSE std", "MSE stderr", "MAE mean", "MAE std", "MAE stderr", "n"]:
                a = row.get(f"{col}_recomputed")
                b = row.get(f"{col}_current")
                if pd.notna(a) and pd.notna(b):
                    max_diff = max(max_diff, abs(float(a) - float(b)))
                elif not (pd.isna(a) and pd.isna(b)):
                    max_diff = np.inf
            diffs.append(max_diff)
        matches = all(d < 1e-10 for d in diffs) and all(merged["_merge"].eq("both"))

    old_relabel_summary = combined[combined["source_stage"].isin(["stage2_illness_h24", "stage3_etth1_h96"])].groupby(
        ["source_stage", "protocol_type", "adapter_label"], dropna=False
    ).size().reset_index(name="rows")
    lines = [
        "# Stage 7 Duplicate and Summary Audit",
        "",
        f"- Stage 7 raw rows: {len(stage7)}",
        f"- Duplicate rows at dataset/horizon/seed/protocol/adapter/metric key: {len(dup)}",
        f"- Recomputed summary matches current Stage 7 summary: {matches}",
        "",
        "## Old Stage 2/3 Relabel Summary",
        "",
        "| source_stage | protocol_type | adapter_label | rows |",
        "|---|---|---|---:|",
    ]
    for _, row in old_relabel_summary.iterrows():
        lines.append(f"| {row.source_stage} | {row.protocol_type} | {row.adapter_label} | {row.rows} |")
    (OUT_DIR / "stage7_duplicate_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
