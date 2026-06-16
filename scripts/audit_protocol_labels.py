from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path("results/code_logic_audit")
NAMED_ADAPTERS = {"linearAE", "dropoutLinearAE", "linearVAE", "VAE"}
RAW_FILES = [
    ("stage2_illness_h24", Path("results/stage2_illness_h24/raw_results.csv")),
    ("stage3_etth1_h96", Path("results/stage3_etth1_h96/raw_results.csv")),
    ("stage7", Path("results/stage7_remaining_table1_with_baseline_types/raw_results.csv")),
]


def as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def clean_adapter(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value)
    return "" if value.lower() in {"nan", "none"} else value


def get_supervised(row: pd.Series) -> str:
    value = row.get("supervised", "")
    if pd.isna(value) or str(value).lower() == "nan" or str(value) == "":
        value = row.get("is_fine_tuned", "")
    return str(value)


def load_raw() -> pd.DataFrame:
    frames = []
    for source_stage, path in RAW_FILES:
        if not path.exists():
            frames.append(
                pd.DataFrame(
                    [{"source_stage": source_stage, "source_file": str(path), "missing_file": True}]
                )
            )
            continue
        df = pd.read_csv(path)
        df["source_stage"] = source_stage
        df["source_file"] = str(path)
        df["missing_file"] = False
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def expected_protocol(row: pd.Series) -> tuple[str, str]:
    source_stage = row.get("source_stage", "")
    adapter = clean_adapter(row.get("adapter", ""))
    use_revin = as_bool(row.get("use_revin", False))
    supervised = get_supervised(row)
    if source_stage in {"stage2_illness_h24", "stage3_etth1_h96"}:
        if adapter == "" and use_revin and supervised == "ft_then_supervised":
            return "strong-baseline", "strong-baseline"
        if adapter in NAMED_ADAPTERS:
            return "adapter", adapter
        return "unsupported_or_legacy", adapter or "unknown"
    protocol = str(row.get("protocol_type", ""))
    adapter_label = str(row.get("adapter_label", ""))
    return protocol, adapter_label


def audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    anomalies = []
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    for idx, row in df.iterrows():
        if row.get("missing_file", False):
            anomalies.append(
                {"row_index": idx, "source_stage": row["source_stage"], "issue": "missing_raw_file", "detail": row["source_file"]}
            )
            continue
        source_stage = row.get("source_stage", "")
        adapter = clean_adapter(row.get("adapter", ""))
        use_revin = as_bool(row.get("use_revin", False))
        supervised = get_supervised(row)
        protocol = str(row.get("protocol_type", ""))
        adapter_label = str(row.get("adapter_label", ""))

        if source_stage == "stage7":
            if protocol == "baseline":
                if use_revin or supervised != "ft" or adapter not in {"", "baseline"}:
                    anomalies.append(
                        {
                            "row_index": idx,
                            "source_stage": source_stage,
                            "issue": "baseline_protocol_mismatch",
                            "detail": f"use_revin={use_revin}, supervised={supervised}, adapter={adapter}",
                        }
                    )
            elif protocol == "strong-baseline":
                if (not use_revin) or supervised != "ft_then_supervised" or adapter not in {"", "strong-baseline"}:
                    anomalies.append(
                        {
                            "row_index": idx,
                            "source_stage": source_stage,
                            "issue": "strong_baseline_protocol_mismatch",
                            "detail": f"use_revin={use_revin}, supervised={supervised}, adapter={adapter}",
                        }
                    )
            elif protocol == "adapter":
                if adapter_label not in NAMED_ADAPTERS or adapter != adapter_label or not use_revin:
                    anomalies.append(
                        {
                            "row_index": idx,
                            "source_stage": source_stage,
                            "issue": "adapter_protocol_mismatch",
                            "detail": f"adapter={adapter}, adapter_label={adapter_label}, use_revin={use_revin}",
                        }
                    )
            else:
                anomalies.append(
                    {
                        "row_index": idx,
                        "source_stage": source_stage,
                        "issue": "missing_or_unknown_protocol_type",
                        "detail": f"protocol_type={protocol}",
                    }
                )
        else:
            expected_type, expected_label = expected_protocol(row)
            if expected_type == "strong-baseline" and "baseline" in str(row.get("source_file", "")):
                pass
            if expected_type == "unsupported_or_legacy":
                anomalies.append(
                    {
                        "row_index": idx,
                        "source_stage": source_stage,
                        "issue": "legacy_unsupported_protocol",
                        "detail": f"adapter={adapter}, use_revin={use_revin}, supervised={supervised}",
                    }
                )

    key_cols = ["source_stage", "dataset", "horizon", "forecast_horizon", "seed", "protocol_type", "adapter_label", "adapter", "metric"]
    existing_key_cols = [c for c in key_cols if c in df.columns]
    if existing_key_cols:
        dup = df[~df.get("missing_file", False)].groupby(existing_key_cols, dropna=False).size().reset_index(name="count")
        dup = dup[dup["count"] > 1]
        for _, row in dup.iterrows():
            anomalies.append(
                {
                    "row_index": "",
                    "source_stage": row.get("source_stage", ""),
                    "issue": "duplicate_metric_rows",
                    "detail": row.to_dict(),
                }
            )

    summary_cols = ["source_stage", "protocol_type", "adapter_label", "adapter", "use_revin", "supervised", "is_fine_tuned", "metric"]
    for col in summary_cols:
        if col not in df.columns:
            df[col] = ""
    summary = (
        df[~df.get("missing_file", False)]
        .assign(
            protocol_type=df["protocol_type"].fillna(""),
            adapter_label=df["adapter_label"].fillna(""),
            adapter=df["adapter"].fillna(""),
        )
        .groupby(summary_cols, dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(summary_cols)
    )
    anomalies_df = pd.DataFrame(anomalies)
    if anomalies_df.empty:
        anomalies_df = pd.DataFrame(columns=["row_index", "source_stage", "issue", "detail"])
    return anomalies_df, summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    anomalies, summary = audit(df)
    anomalies.to_csv(OUT_DIR / "protocol_label_anomalies.csv", index=False)
    summary.to_csv(OUT_DIR / "protocol_label_summary.csv", index=False)
    print(f"raw rows inspected: {len(df)}")
    print(f"anomalies: {len(anomalies)}")
    print(f"saved: {OUT_DIR / 'protocol_label_anomalies.csv'}")
    print(f"saved: {OUT_DIR / 'protocol_label_summary.csv'}")


if __name__ == "__main__":
    main()
