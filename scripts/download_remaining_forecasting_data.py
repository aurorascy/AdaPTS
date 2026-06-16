import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from adapts.utils.main_script import prepare_data


DATASETS = {
    "Weather": {
        "url": "https://huggingface.co/datasets/dunzane/time-series-dataset/resolve/main/LSF/weather/weather.csv",
        "output": Path("external_data/forecasting/Weather/Weather.csv"),
    },
    "ExchangeRate": {
        "url": "https://huggingface.co/datasets/dunzane/time-series-dataset/resolve/main/LSF/exchange_rate/exchange_rate.csv",
        "output": Path("external_data/forecasting/ExchangeRate/ExchangeRate.csv"),
    },
}


def safe_text(value) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def validate_csv(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    numeric_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
    result = {
        "exists": path.exists(),
        "path": str(path),
        "size": path.stat().st_size,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "missing": int(df.isna().sum().sum()),
        "numeric_feature_columns": numeric_cols,
        "numeric_feature_count": len(numeric_cols),
        "head": df.head().to_dict(orient="records"),
    }
    if len(df) == 0:
        raise ValueError(f"{path} has zero rows")
    if "date" not in df.columns:
        raise ValueError(f"{path} does not contain a date column")
    if len(numeric_cols) == 0:
        raise ValueError(f"{path} has no numeric feature columns")
    return result


def check_datareader(dataset: str, horizons: list[int]) -> dict:
    checks = {}
    for horizon in horizons:
        dataset_name = f"{dataset}_pred={horizon}"
        X_train, y_train, X_val, y_val, X_test, y_test, n_features = prepare_data(
            dataset_name, 512, horizon
        )
        checks[str(horizon)] = {
            "n_features": int(n_features),
            "train_X": list(X_train.shape),
            "train_y": list(y_train.shape),
            "val_X": list(X_val.shape),
            "val_y": list(y_val.shape),
            "test_X": list(X_test.shape),
            "test_y": list(y_test.shape),
        }
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Weather and ExchangeRate LTSF data.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dataset", choices=list(DATASETS), action="append")
    args = parser.parse_args()

    selected = args.dataset or list(DATASETS)
    report = {
        "source": "dunzane/time-series-dataset HuggingFace mirror of LTSF benchmark data",
        "datasets": {},
    }

    for dataset in selected:
        meta = DATASETS[dataset]
        output = meta["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and not args.force:
            print(f"{dataset}: exists, skip download: {output}")
        else:
            print(f"{dataset}: downloading {meta['url']} -> {output}")
            urlretrieve(meta["url"], output)

        csv_info = validate_csv(output)
        horizons = [96, 192]
        datareader_info = check_datareader(dataset, horizons)
        report["datasets"][dataset] = {
            "url": meta["url"],
            "csv": csv_info,
            "datareader": datareader_info,
        }
        print("=" * 80)
        print(dataset)
        print("shape:", csv_info["shape"])
        print("columns:", safe_text(csv_info["columns"]))
        print("missing:", csv_info["missing"])
        print("numeric features:", csv_info["numeric_feature_count"])
        print("datareader:", safe_text(datareader_info))

    report_path = Path("DATA_PREP_REMAINING_DATASETS.md")
    lines = [
        "# Remaining Forecasting Data Preparation",
        "",
        "This file records Weather and ExchangeRate data preparation for Stage 7. No synthetic data was generated.",
        "",
        f"- Source: {report['source']}",
        "",
    ]
    for dataset, info in report["datasets"].items():
        csv_info = info["csv"]
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- URL: {info['url']}",
                f"- Local path: `{csv_info['path']}`",
                f"- Exists: {csv_info['exists']}",
                f"- File size: {csv_info['size']}",
                f"- Shape: {tuple(csv_info['shape'])}",
                f"- Columns: {csv_info['columns']}",
                f"- Missing values: {csv_info['missing']}",
                f"- Numeric feature count: {csv_info['numeric_feature_count']}",
                "",
                "### DataReader checks",
                "",
            ]
        )
        for horizon, shapes in info["datareader"].items():
            lines.extend(
                [
                    f"- H={horizon}: train X/y {tuple(shapes['train_X'])} / {tuple(shapes['train_y'])}; "
                    f"val X/y {tuple(shapes['val_X'])} / {tuple(shapes['val_y'])}; "
                    f"test X/y {tuple(shapes['test_X'])} / {tuple(shapes['test_y'])}; "
                    f"n_features={shapes['n_features']}",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## Machine-readable summary",
            "",
            "```json",
            json.dumps(report, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
