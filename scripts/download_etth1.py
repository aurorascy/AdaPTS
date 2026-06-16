from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd


DEFAULT_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)
DEFAULT_OUTPUT = Path("external_data/forecasting/ETTh1/ETTh1.csv")
EXPECTED_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def download_file(url: str, output: Path, force: bool = False) -> bool:
    if output.exists() and not force:
        print(f"File already exists, skipping download: {output}")
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    print(f"Downloading: {url}")
    print(f"Saving to: {output}")

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with tmp_output.open("wb") as handle:
                handle.write(response.read())
        tmp_output.replace(output)
    except Exception:
        if tmp_output.exists():
            tmp_output.unlink()
        raise

    print(f"Downloaded file size: {output.stat().st_size} bytes")
    return True


def validate_csv(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    columns = list(df.columns)
    missing_total = int(df.isna().sum().sum())
    numeric_columns = list(df.drop(columns=["date"], errors="ignore").select_dtypes("number").columns)
    missing_expected = [column for column in EXPECTED_COLUMNS if column not in columns]

    print(f"exists: {path.exists()}")
    print(f"file size: {path.stat().st_size} bytes")
    print(f"shape: {df.shape}")
    print(f"columns: {columns}")
    print("head:")
    print(df.head().to_string(index=False))
    print("tail:")
    print(df.tail().to_string(index=False))
    print(f"missing values: {missing_total}")
    print(f"numeric feature columns ({len(numeric_columns)}): {numeric_columns}")
    print(f"missing expected columns: {missing_expected}")

    errors = []
    if len(df) == 0:
        errors.append("CSV has zero rows.")
    if "date" not in columns:
        errors.append("CSV does not contain a date column.")
    if len(numeric_columns) < 7:
        errors.append("CSV has fewer than 7 numeric feature columns excluding date.")
    if missing_expected:
        errors.append(f"CSV is missing expected columns: {missing_expected}")

    if errors:
        for error in errors:
            print(f"VALIDATION ERROR: {error}", file=sys.stderr)
        raise ValueError("ETTh1 CSV validation failed.")

    print("ETTh1 CSV validation passed.")
    return {
        "exists": True,
        "file_size": path.stat().st_size,
        "shape": df.shape,
        "columns": columns,
        "numeric_columns": numeric_columns,
        "missing_values": missing_total,
        "missing_expected_columns": missing_expected,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and validate ETTh1.csv.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing file.")
    parser.add_argument("--url", default=DEFAULT_URL, help="ETTh1 CSV URL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Local output path for ETTh1.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        download_file(args.url, args.output, force=args.force)
        validate_csv(args.output)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
