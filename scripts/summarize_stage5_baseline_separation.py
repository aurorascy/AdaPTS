from pathlib import Path

import numpy as np
import pandas as pd


RAW_RESULTS = Path("results/stage5_baseline_separation/raw_results.csv")
SUMMARY_CSV = Path("results/stage5_baseline_separation/summary_baseline_types.csv")
SUMMARY_MD = Path("results/stage5_baseline_separation/summary_baseline_types.md")


EXPECTED = {
    "B0_pure_moment": "Pure MOMENT zero-shot",
    "B1_head_only": "MOMENT head-only fine-tuning",
    "B2_strong_revin": "Strong ADAPTS/RevIN no-named-adapter baseline",
    "B3_dropoutLinearAE": "dropoutLinearAE named adapter",
}


def format_float(value):
    if pd.isna(value):
        return "missing"
    return f"{float(value):.6f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    if not RAW_RESULTS.exists():
        raise FileNotFoundError(RAW_RESULTS)

    df = pd.read_csv(RAW_RESULTS)
    print("columns:", list(df.columns))
    print("head:")
    print(df.head())

    metric_df = df[df["metric"].isin(["mse", "mae"])].copy()
    metric_df["value"] = pd.to_numeric(metric_df["value"], errors="coerce")

    rows = []
    for (dataset, horizon), group in metric_df.groupby(["dataset", "horizon"], dropna=False):
        pivot = group.pivot_table(
            index="baseline_type",
            columns="metric",
            values="value",
            aggfunc="mean",
        )
        b0_mse = pivot.loc["B0_pure_moment", "mse"] if "B0_pure_moment" in pivot.index and "mse" in pivot.columns else np.nan
        b2_mse = pivot.loc["B2_strong_revin", "mse"] if "B2_strong_revin" in pivot.index and "mse" in pivot.columns else np.nan
        for baseline_type, description in EXPECTED.items():
            mse = pivot.loc[baseline_type, "mse"] if baseline_type in pivot.index and "mse" in pivot.columns else np.nan
            mae = pivot.loc[baseline_type, "mae"] if baseline_type in pivot.index and "mae" in pivot.columns else np.nan
            if pd.notna(mse) and pd.notna(b0_mse):
                improvement_vs_b0 = (b0_mse - mse) / b0_mse * 100.0
            else:
                improvement_vs_b0 = np.nan
            if pd.notna(mse) and pd.notna(b2_mse):
                gap_vs_b2 = (mse - b2_mse) / b2_mse * 100.0
            else:
                gap_vs_b2 = np.nan
            notes = "ok" if pd.notna(mse) or pd.notna(mae) else "missing"
            rows.append(
                {
                    "Dataset": dataset,
                    "Horizon": horizon,
                    "Baseline type": baseline_type,
                    "Description": description,
                    "MSE": mse,
                    "MAE": mae,
                    "Improvement vs B0 (%)": improvement_vs_b0,
                    "Gap vs B2 (%)": gap_vs_b2,
                    "Notes": notes,
                }
            )

    summary = pd.DataFrame(rows)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    md = summary.copy()
    for col in ["MSE", "MAE", "Improvement vs B0 (%)", "Gap vs B2 (%)"]:
        md[col] = md[col].map(format_float)
    markdown = dataframe_to_markdown(md)
    SUMMARY_MD.write_text(markdown + "\n", encoding="utf-8")

    print("summary:")
    print(summary)
    print(f"saved: {SUMMARY_CSV}")
    print(markdown)


if __name__ == "__main__":
    main()
