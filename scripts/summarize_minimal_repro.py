from pathlib import Path

import pandas as pd


INPUT_CSV = Path("results/minimal_repro/illness_h24_minimal.csv")
OUTPUT_CSV = Path("results/minimal_repro/illness_h24_minimal_summary.csv")


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)
    required_columns = {
        "dataset",
        "foundational_model",
        "adapter",
        "context_length",
        "forecast_horizon",
        "seed",
        "metric",
        "value",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    data = df.copy()
    data["adapter"] = data["adapter"].fillna("None").replace("", "None")

    index_columns = [
        "dataset",
        "forecast_horizon",
        "context_length",
        "foundational_model",
        "adapter",
        "seed",
    ]
    summary = (
        data.pivot_table(
            index=index_columns,
            columns="metric",
            values="value",
            aggfunc="last",
            dropna=False,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    preferred_columns = [
        "dataset",
        "forecast_horizon",
        "context_length",
        "foundational_model",
        "adapter",
        "seed",
        "mse",
        "mae",
        "scaled_mse",
        "scaled_mae",
        "ks",
        "ece",
    ]
    existing_columns = [col for col in preferred_columns if col in summary.columns]
    remaining_columns = [col for col in summary.columns if col not in existing_columns]
    summary = summary[existing_columns + remaining_columns]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
