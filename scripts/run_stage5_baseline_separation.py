import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


RESULT_DIR = Path("results/stage5_baseline_separation")
LOG_DIR = Path("logs/stage5_baseline_separation")
RAW_RESULTS = RESULT_DIR / "raw_results.csv"
FAILED_RUNS = RESULT_DIR / "failed_runs.json"
MODEL_NAME = "AutonLab/MOMENT-1-small"
CONTEXT_LENGTH = 512
SEED = 13


BASELINE_INFO = {
    "B0": {
        "baseline_type": "B0_pure_moment",
        "description": "Pure MOMENT zero-shot; no ADAPTS wrapper, no RevIN, no fine-tuning",
    },
    "B1": {
        "baseline_type": "B1_head_only",
        "description": "MOMENT head-only fine-tuning; no RevIN; no supervised adapter training",
    },
    "B2": {
        "baseline_type": "B2_strong_revin",
        "description": "Current strong no-named-adapter baseline with RevIN/JustRevIn",
    },
    "B3": {
        "baseline_type": "B3_dropoutLinearAE",
        "description": "Named dropoutLinearAE adapter under ft_then_supervised protocol",
    },
}


RAW_COLUMNS = [
    "stage",
    "dataset",
    "horizon",
    "context_length",
    "seed",
    "baseline_type",
    "description",
    "adapter",
    "use_revin",
    "supervised",
    "n_epochs_fine_tuning",
    "n_epochs_adapter",
    "n_components",
    "metric",
    "value",
    "notes",
    "source",
    "model_name",
    "device",
    "n_features",
    "train_size",
    "running_time",
    "pca_in_preprocessing",
    "is_fine_tuned",
]


def dataset_settings(dataset: str) -> tuple[int, str]:
    if dataset == "ETTh1":
        return 96, "ETTh1_h96_seed13"
    if dataset == "Illness":
        return 24, "Illness_h24_seed13"
    raise ValueError(f"Unsupported dataset: {dataset}")


def select_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def quote_cmd(cmd: list[str]) -> str:
    return subprocess.list2cmdline([str(x) for x in cmd])


def existing_baseline_types(dataset: str, horizon: int) -> set[str]:
    if not RAW_RESULTS.exists():
        return set()
    found = set()
    with RAW_RESULTS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row.get("dataset") == dataset
                and str(row.get("horizon")) == str(horizon)
                and str(row.get("seed")) == str(SEED)
            ):
                found.add(row.get("baseline_type", ""))
    return found


def append_runpy_rows(
    temp_csv: Path,
    *,
    dataset: str,
    horizon: int,
    baseline_key: str,
    supervised: str,
    use_revin: bool,
    ft_epochs: int,
    adapter_epochs: int,
    device: str,
    notes: str,
) -> None:
    if not temp_csv.exists():
        raise FileNotFoundError(temp_csv)

    info = BASELINE_INFO[baseline_key]
    RAW_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RAW_RESULTS.exists()
    with temp_csv.open(newline="", encoding="utf-8") as src, RAW_RESULTS.open(
        "a", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=RAW_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in reader:
            writer.writerow(
                {
                    "stage": "stage5_baseline_separation",
                    "dataset": dataset,
                    "horizon": horizon,
                    "context_length": row.get("context_length", CONTEXT_LENGTH),
                    "seed": row.get("seed", SEED),
                    "baseline_type": info["baseline_type"],
                    "description": info["description"],
                    "adapter": row.get("adapter", ""),
                    "use_revin": use_revin,
                    "supervised": supervised,
                    "n_epochs_fine_tuning": ft_epochs,
                    "n_epochs_adapter": adapter_epochs,
                    "n_components": row.get("n_components", ""),
                    "metric": row.get("metric", ""),
                    "value": row.get("value", ""),
                    "notes": notes,
                    "source": "scripts/run.py",
                    "model_name": row.get("foundational_model", MODEL_NAME),
                    "device": device,
                    "n_features": row.get("n_features", ""),
                    "train_size": row.get("train_size", ""),
                    "running_time": row.get("running_time", ""),
                    "pca_in_preprocessing": row.get("pca_in_preprocessing", False),
                    "is_fine_tuned": row.get("is_fine_tuned", ""),
                }
            )


def build_command(
    baseline_key: str,
    *,
    dataset: str,
    horizon: int,
    ft_epochs: int,
    adapter_epochs: int,
    device: str,
    temp_csv: Path,
    log_dir: Path,
) -> list[str]:
    if baseline_key == "B0":
        return [
            sys.executable,
            "scripts/evaluate_pure_moment_baseline.py",
            "--dataset",
            dataset,
            "--horizon",
            str(horizon),
            "--context_length",
            str(CONTEXT_LENGTH),
            "--seed",
            str(SEED),
            "--model_name",
            MODEL_NAME,
            "--device",
            device,
            "--output",
            str(RAW_RESULTS),
        ]

    cmd = [
        sys.executable,
        "scripts/run.py",
        "--forecast-horizon",
        str(horizon),
        "--model-name",
        MODEL_NAME,
        "--context-length",
        str(CONTEXT_LENGTH),
        "--seed",
        str(SEED),
        "--device",
        device,
        "--dataset-name",
        dataset,
        "--data-path",
        str(temp_csv),
        "--log-dir",
        str(log_dir),
        "--number-n-comp-to-try",
        "1",
        "--n-epochs-fine-tuning",
        str(ft_epochs),
        "--n-epochs-adapter",
        str(adapter_epochs),
    ]

    if baseline_key == "B1":
        cmd += ["--no-use-revin", "--supervised", "ft"]
    elif baseline_key == "B2":
        cmd += ["--use-revin", "--supervised", "ft_then_supervised"]
    elif baseline_key == "B3":
        cmd += [
            "--adapter",
            "dropoutLinearAE",
            "--use-revin",
            "--supervised",
            "ft_then_supervised",
        ]
    else:
        raise ValueError(f"Unknown baseline key: {baseline_key}")
    return cmd


def save_failed(failures: list[dict]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_RUNS.write_text(json.dumps(failures, indent=2), encoding="utf-8")


def run_one(
    baseline_key: str,
    *,
    dataset: str,
    horizon: int,
    dataset_tag: str,
    ft_epochs: int,
    adapter_epochs: int,
    device: str,
    dry_run: bool,
) -> dict | None:
    info = BASELINE_INFO[baseline_key]
    run_tag = f"{dataset_tag}_{info['baseline_type']}"
    log_dir = LOG_DIR / run_tag
    temp_csv = RESULT_DIR / "tmp" / f"{run_tag}.csv"

    if temp_csv.exists():
        temp_csv.unlink()
    temp_csv.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(
        baseline_key,
        dataset=dataset,
        horizon=horizon,
        ft_epochs=ft_epochs,
        adapter_epochs=adapter_epochs,
        device=device,
        temp_csv=temp_csv,
        log_dir=log_dir,
    )
    print("=" * 100)
    print(f"{baseline_key} {info['baseline_type']}")
    if baseline_key == "B0":
        print("B0 uses evaluate_pure_moment_baseline.py; it is not represented by scripts/run.py.")
    if baseline_key == "B1":
        print("B1 uses scripts/run.py --supervised ft --no-use-revin; it avoids adapter_supervised_fine_tuning.")
    print(quote_cmd(cmd))
    if dry_run:
        return None

    started = datetime.now().isoformat(timespec="seconds")
    tic = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - tic
    ended = datetime.now().isoformat(timespec="seconds")
    print(f"return_code={result.returncode}, elapsed={elapsed:.2f}s")

    if result.returncode != 0:
        return {
            "baseline": baseline_key,
            "baseline_type": info["baseline_type"],
            "dataset": dataset,
            "horizon": horizon,
            "seed": SEED,
            "return_code": result.returncode,
            "started": started,
            "ended": ended,
            "elapsed_seconds": elapsed,
            "command": quote_cmd(cmd),
        }

    if baseline_key != "B0":
        append_runpy_rows(
            temp_csv,
            dataset=dataset,
            horizon=horizon,
            baseline_key=baseline_key,
            supervised="ft" if baseline_key == "B1" else "ft_then_supervised",
            use_revin=baseline_key in {"B2", "B3"},
            ft_epochs=ft_epochs,
            adapter_epochs=adapter_epochs,
            device=device,
            notes=(
                "run.py does not expose direct --n-components; n_components=7 is obtained "
                "from full-channel setting for this dataset."
            ),
        )
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 5 baseline separation.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dataset", default="ETTh1", choices=["ETTh1", "Illness"])
    parser.add_argument("--full_epochs", action="store_true")
    parser.add_argument("--only", choices=["B0", "B1", "B2", "B3"])
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    horizon, dataset_tag = dataset_settings(args.dataset)
    device = select_device()
    ft_epochs = 50 if args.full_epochs else 5
    adapter_epochs = 300 if args.full_epochs else 30
    baseline_keys = [args.only] if args.only else ["B0", "B1", "B2", "B3"]
    existing = existing_baseline_types(args.dataset, horizon)

    print(f"dataset={args.dataset}, horizon={horizon}, seed={SEED}")
    print(f"device={device}, ft_epochs={ft_epochs}, adapter_epochs={adapter_epochs}")
    print("CLI audit: run.py supports --supervised ft, --use-revin/--no-use-revin, but not direct --n-components.")

    failures = []
    if FAILED_RUNS.exists() and not args.dry_run:
        try:
            failures = json.loads(FAILED_RUNS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            backup = FAILED_RUNS.with_suffix(".json.bak")
            shutil.copy2(FAILED_RUNS, backup)
            failures = []

    for baseline_key in baseline_keys:
        baseline_type = BASELINE_INFO[baseline_key]["baseline_type"]
        if args.skip_existing and baseline_type in existing:
            print(f"Skipping existing {baseline_type} for {args.dataset} H={horizon}")
            continue
        failure = run_one(
            baseline_key,
            dataset=args.dataset,
            horizon=horizon,
            dataset_tag=dataset_tag,
            ft_epochs=ft_epochs,
            adapter_epochs=adapter_epochs,
            device=device,
            dry_run=args.dry_run,
        )
        if failure:
            failures.append(failure)
            save_failed(failures)

    if not args.dry_run:
        save_failed(failures)
        print(f"failed runs saved to {FAILED_RUNS}")


if __name__ == "__main__":
    main()
