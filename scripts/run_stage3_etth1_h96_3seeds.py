from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


DATASET = "ETTh1"
FORECAST_HORIZON = 96
CONTEXT_LENGTH = 512
MODEL_NAME = "AutonLab/MOMENT-1-small"
SUPERVISED = "ft_then_supervised"
SEEDS = [13, 42, 2024]
ADAPTERS = [None, "PCA", "linearAE", "dropoutLinearAE", "linearVAE", "VAE"]
ADAPTER_CLI_NAMES = {
    None: None,
    "PCA": "pca",
    "linearAE": "linearAE",
    "dropoutLinearAE": "dropoutLinearAE",
    "linearVAE": "linearVAE",
    "VAE": "VAE",
}

RESULT_DIR = Path("results/stage3_etth1_h96")
LOG_ROOT = Path("logs/stage3_etth1_h96")
RAW_RESULTS = RESULT_DIR / "raw_results.csv"
FAILED_RUNS = RESULT_DIR / "failed_runs.json"


def adapter_label(adapter: str | None) -> str:
    return "baseline" if adapter is None else adapter


def select_device_and_epochs() -> tuple[str, int, int, str | None]:
    if torch is not None and torch.cuda.is_available():
        return "cuda:0", 5, 30, None
    return "cpu", 1, 5, None


def build_command(
    seed: int,
    adapter: str | None,
    device: str,
    n_epochs_fine_tuning: int,
    n_epochs_adapter: int,
) -> list[str]:
    label = adapter_label(adapter)
    command = [
        sys.executable,
        "scripts/run.py",
        "--forecast-horizon",
        str(FORECAST_HORIZON),
        "--model-name",
        MODEL_NAME,
        "--context-length",
        str(CONTEXT_LENGTH),
        "--seed",
        str(seed),
        "--device",
        device,
        "--dataset-name",
        DATASET,
        "--use-revin",
        "--supervised",
        SUPERVISED,
        "--n-epochs-fine-tuning",
        str(n_epochs_fine_tuning),
        "--n-epochs-adapter",
        str(n_epochs_adapter),
        "--data-path",
        str(RAW_RESULTS),
        "--log-dir",
        str(LOG_ROOT / f"seed_{seed}" / label),
    ]
    adapter_cli = ADAPTER_CLI_NAMES[adapter]
    if adapter_cli is not None:
        command.extend(["--adapter", adapter_cli])
    return command


def load_existing_successes() -> set[tuple[int, str]]:
    if not RAW_RESULTS.exists():
        return set()

    df = pd.read_csv(RAW_RESULTS)
    required_columns = {"seed", "adapter", "metric"}
    if not required_columns.issubset(df.columns):
        return set()

    df = df.copy()
    df["adapter"] = df["adapter"].fillna("baseline").replace({"pca": "PCA"})
    successes: set[tuple[int, str]] = set()
    for (seed, adapter), group in df.groupby(["seed", "adapter"], dropna=False):
        metrics = {str(metric).lower() for metric in group["metric"].dropna()}
        if {"mse", "mae"}.issubset(metrics):
            successes.add((int(seed), str(adapter)))
    return successes


def save_failures(failures: list[dict[str, object]]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_RUNS.write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def append_run_log(record: dict[str, object]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / "run_history.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed",
                "adapter",
                "returncode",
                "start_time",
                "end_time",
                "duration_seconds",
                "command",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 ETTh1 H=96 experiments.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only.")
    parser.add_argument("--only_seed", type=int, choices=SEEDS, help="Run one seed only.")
    parser.add_argument(
        "--only_adapter",
        choices=["baseline", "PCA", "linearAE", "dropoutLinearAE", "linearVAE", "VAE"],
        help="Run one adapter only. Use baseline for no adapter.",
    )
    parser.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip seed+adapter pairs already present in raw_results.csv.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when a seed+adapter result already exists.",
    )
    return parser.parse_args()


def selected_seeds(only_seed: int | None) -> list[int]:
    return [only_seed] if only_seed is not None else SEEDS


def selected_adapters(only_adapter: str | None) -> list[str | None]:
    if only_adapter is None:
        return ADAPTERS
    return [None if only_adapter == "baseline" else only_adapter]


def main() -> int:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    device, n_epochs_fine_tuning, n_epochs_adapter, epoch_note = select_device_and_epochs()
    existing = load_existing_successes() if args.skip_existing and not args.force else set()
    failures: list[dict[str, object]] = []

    print(f"dataset={DATASET}")
    print(f"forecast_horizon={FORECAST_HORIZON}")
    print(f"context_length={CONTEXT_LENGTH}")
    print(f"model_name={MODEL_NAME}")
    print(f"device={device}")
    print(f"n_epochs_fine_tuning={n_epochs_fine_tuning}")
    print(f"n_epochs_adapter={n_epochs_adapter}")
    print(f"raw_results={RAW_RESULTS}")
    print(f"failed_runs={FAILED_RUNS}")
    if epoch_note:
        print(f"epoch_note={epoch_note}")

    for seed in selected_seeds(args.only_seed):
        for adapter in selected_adapters(args.only_adapter):
            label = adapter_label(adapter)
            print("\n" + "=" * 100)
            print(f"seed={seed}, adapter={label}")

            if (seed, label) in existing:
                print(f"SKIP existing successful result: seed={seed}, adapter={label}")
                continue

            command = build_command(
                seed,
                adapter,
                device,
                n_epochs_fine_tuning,
                n_epochs_adapter,
            )
            print(" ".join(command))

            if args.dry_run:
                continue

            start = datetime.now()
            start_monotonic = time.monotonic()
            result = subprocess.run(command)
            end = datetime.now()
            duration = time.monotonic() - start_monotonic
            run_record = {
                "seed": seed,
                "adapter": label,
                "returncode": result.returncode,
                "start_time": start.isoformat(timespec="seconds"),
                "end_time": end.isoformat(timespec="seconds"),
                "duration_seconds": f"{duration:.3f}",
                "command": " ".join(command),
            }
            append_run_log(run_record)
            print(
                f"returncode={result.returncode}, start={run_record['start_time']}, "
                f"end={run_record['end_time']}, duration={duration:.2f}s"
            )

            if result.returncode != 0:
                failure = {
                    "seed": seed,
                    "adapter": label,
                    "adapter_cli": ADAPTER_CLI_NAMES[adapter],
                    "returncode": result.returncode,
                    "start_time": run_record["start_time"],
                    "end_time": run_record["end_time"],
                    "duration_seconds": duration,
                    "command": command,
                }
                failures.append(failure)
                print(f"FAILED: {failure}")
            else:
                print(f"SUCCESS: seed={seed}, adapter={label}")

    if not args.dry_run:
        save_failures(failures)
    print("\nStage 3 batch run finished.")
    print(f"Failures recorded: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
