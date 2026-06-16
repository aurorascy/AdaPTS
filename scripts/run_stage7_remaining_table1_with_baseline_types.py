import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


RESULT_DIR = Path("results/stage7_remaining_table1_with_baseline_types")
LOG_DIR = Path("logs/stage7_remaining_table1_with_baseline_types")
RAW_RESULTS = RESULT_DIR / "raw_results.csv"
FAILED_RUNS = RESULT_DIR / "failed_runs.json"
MODEL_NAME = "AutonLab/MOMENT-1-small"
CONTEXT_LENGTH = 512
SEEDS = [13, 42, 2024]
ADAPTERS = ["linearAE", "dropoutLinearAE", "linearVAE", "VAE"]

DEFAULT_TASKS = [
    ("ETTh1", 192),
    ("Illness", 60),
    ("Weather", 96),
    ("Weather", 192),
    ("ExchangeRate", 96),
    ("ExchangeRate", 192),
]

N_FEATURES = {
    "ETTh1": 7,
    "Illness": 7,
    "Weather": 21,
    "ExchangeRate": 8,
}

RAW_COLUMNS = [
    "stage",
    "dataset",
    "horizon",
    "context_length",
    "seed",
    "protocol_type",
    "adapter_label",
    "adapter",
    "use_revin",
    "supervised",
    "n_epochs_fine_tuning",
    "n_epochs_adapter",
    "n_components",
    "metric",
    "value",
    "notes",
    "source_stage",
    "source",
    "model_name",
    "device",
    "n_features",
    "train_size",
    "running_time",
    "pca_in_preprocessing",
    "is_fine_tuned",
]


def select_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def epoch_policy(args: argparse.Namespace) -> tuple[int, int, str]:
    if args.full_epochs:
        return 50, 300, "full"
    if args.medium:
        return 20, 100, "medium"
    return 5, 30, "fast"


def quote_cmd(cmd: list[str]) -> str:
    return subprocess.list2cmdline([str(x) for x in cmd])


def task_label(dataset: str, horizon: int, seed: int, protocol_type: str, adapter_label: str) -> str:
    label = f"{dataset}_h{horizon}_seed{seed}_{protocol_type}"
    if protocol_type == "adapter":
        label += f"_{adapter_label}"
    return label


def existing_keys() -> set[tuple[str, str, str, str, str]]:
    if not RAW_RESULTS.exists():
        return set()
    keys = set()
    with RAW_RESULTS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keys.add(
                (
                    row.get("dataset", ""),
                    str(row.get("horizon", "")),
                    str(row.get("seed", "")),
                    row.get("protocol_type", ""),
                    row.get("adapter_label", ""),
                )
            )
    return keys


def append_runpy_rows(
    temp_csv: Path,
    *,
    dataset: str,
    horizon: int,
    seed: int,
    protocol_type: str,
    adapter_label: str,
    adapter: str,
    use_revin: bool,
    supervised: str,
    ft_epochs: int,
    adapter_epochs: int,
    device: str,
    notes: str,
) -> None:
    if not temp_csv.exists():
        raise FileNotFoundError(temp_csv)
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
                    "stage": "stage7_remaining_table1_with_baseline_types",
                    "dataset": dataset,
                    "horizon": horizon,
                    "context_length": row.get("context_length", CONTEXT_LENGTH),
                    "seed": seed,
                    "protocol_type": protocol_type,
                    "adapter_label": adapter_label,
                    "adapter": adapter,
                    "use_revin": use_revin,
                    "supervised": supervised,
                    "n_epochs_fine_tuning": ft_epochs,
                    "n_epochs_adapter": adapter_epochs,
                    "n_components": row.get("n_components", N_FEATURES.get(dataset, "")),
                    "metric": row.get("metric", ""),
                    "value": row.get("value", ""),
                    "notes": notes,
                    "source_stage": "stage7",
                    "source": "scripts/run.py",
                    "model_name": row.get("foundational_model", MODEL_NAME),
                    "device": device,
                    "n_features": row.get("n_features", N_FEATURES.get(dataset, "")),
                    "train_size": row.get("train_size", ""),
                    "running_time": row.get("running_time", ""),
                    "pca_in_preprocessing": row.get("pca_in_preprocessing", False),
                    "is_fine_tuned": row.get("is_fine_tuned", supervised),
                }
            )


def build_run_command(
    *,
    dataset: str,
    horizon: int,
    seed: int,
    protocol_type: str,
    adapter_label: str,
    ft_epochs: int,
    adapter_epochs: int,
    device: str,
    temp_csv: Path,
    log_dir: Path,
) -> tuple[list[str], dict]:
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
        str(seed),
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
    ]
    meta = {}
    if protocol_type == "baseline":
        cmd += ["--n-epochs-adapter", "0", "--no-use-revin", "--supervised", "ft"]
        meta = {
            "adapter": "",
            "use_revin": False,
            "supervised": "ft",
            "n_epochs_adapter": 0,
            "notes": "head-only baseline: no named adapter, no RevIN, no adapter_supervised_fine_tuning",
        }
    elif protocol_type == "strong-baseline":
        cmd += [
            "--n-epochs-adapter",
            str(adapter_epochs),
            "--use-revin",
            "--supervised",
            "ft_then_supervised",
        ]
        meta = {
            "adapter": "",
            "use_revin": True,
            "supervised": "ft_then_supervised",
            "n_epochs_adapter": adapter_epochs,
            "notes": "strong-baseline: no named adapter but trains JustRevIn/RevIN under ft_then_supervised",
        }
    elif protocol_type == "adapter":
        cmd += [
            "--n-epochs-adapter",
            str(adapter_epochs),
            "--adapter",
            adapter_label,
            "--use-revin",
            "--supervised",
            "ft_then_supervised",
        ]
        meta = {
            "adapter": adapter_label,
            "use_revin": True,
            "supervised": "ft_then_supervised",
            "n_epochs_adapter": adapter_epochs,
            "notes": "named adapter under ft_then_supervised; PCA intentionally excluded",
        }
    else:
        raise ValueError(f"Unsupported protocol_type: {protocol_type}")
    return cmd, meta


def save_failed(failures: list[dict]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_RUNS.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")


def run_one(
    *,
    dataset: str,
    horizon: int,
    seed: int,
    protocol_type: str,
    adapter_label: str,
    ft_epochs: int,
    adapter_epochs: int,
    device: str,
    dry_run: bool,
) -> dict | None:
    label = task_label(dataset, horizon, seed, protocol_type, adapter_label)
    log_dir = LOG_DIR / label
    temp_csv = RESULT_DIR / "tmp" / f"{label}.csv"
    if temp_csv.exists():
        temp_csv.unlink()
    temp_csv.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd, meta = build_run_command(
        dataset=dataset,
        horizon=horizon,
        seed=seed,
        protocol_type=protocol_type,
        adapter_label=adapter_label,
        ft_epochs=ft_epochs,
        adapter_epochs=adapter_epochs,
        device=device,
        temp_csv=temp_csv,
        log_dir=log_dir,
    )
    print("=" * 100)
    print(f"{dataset} H={horizon} seed={seed} protocol_type={protocol_type} adapter_label={adapter_label}")
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
            "dataset": dataset,
            "horizon": horizon,
            "seed": seed,
            "protocol_type": protocol_type,
            "adapter_label": adapter_label,
            "return_code": result.returncode,
            "started": started,
            "ended": ended,
            "elapsed_seconds": elapsed,
            "command": quote_cmd(cmd),
        }
    append_runpy_rows(
        temp_csv,
        dataset=dataset,
        horizon=horizon,
        seed=seed,
        protocol_type=protocol_type,
        adapter_label=adapter_label,
        adapter=meta["adapter"],
        use_revin=meta["use_revin"],
        supervised=meta["supervised"],
        ft_epochs=ft_epochs,
        adapter_epochs=meta["n_epochs_adapter"],
        device=device,
        notes=meta["notes"] + "; run.py has no direct --n-components, fixed full-channel via --number-n-comp-to-try 1",
    )
    return None


def filtered_tasks(args: argparse.Namespace) -> list[tuple[str, int]]:
    tasks = DEFAULT_TASKS
    if args.only_dataset:
        tasks = [t for t in tasks if t[0] == args.only_dataset]
    if args.only_horizon:
        tasks = [t for t in tasks if t[1] == args.only_horizon]
    return tasks


def iter_experiments(args: argparse.Namespace):
    protocols = [args.only_protocol] if args.only_protocol else ["baseline", "strong-baseline", "adapter"]
    adapters = [args.only_adapter] if args.only_adapter else ADAPTERS
    for dataset, horizon in filtered_tasks(args):
        for seed in SEEDS:
            for protocol_type in protocols:
                if protocol_type == "adapter":
                    for adapter in adapters:
                        yield dataset, horizon, seed, protocol_type, adapter
                else:
                    yield dataset, horizon, seed, protocol_type, protocol_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 7 with explicit baseline protocol types.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--medium", action="store_true")
    parser.add_argument("--full_epochs", action="store_true")
    parser.add_argument("--only_dataset", choices=["ETTh1", "Illness", "Weather", "ExchangeRate"])
    parser.add_argument("--only_horizon", type=int, choices=[24, 60, 96, 192])
    parser.add_argument("--only_protocol", choices=["baseline", "strong-baseline", "adapter"])
    parser.add_argument("--only_adapter", choices=ADAPTERS)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ft_epochs, adapter_epochs, epoch_label = epoch_policy(args)
    device = select_device()
    existing = existing_keys()
    failures: list[dict] = []
    if FAILED_RUNS.exists() and not args.dry_run:
        try:
            failures = json.loads(FAILED_RUNS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            failures = []

    print(f"device={device}, epoch_policy={epoch_label}, ft_epochs={ft_epochs}, adapter_epochs={adapter_epochs}")
    print("baseline is head-only: --supervised ft --no-use-revin")
    print("strong-baseline is no named adapter + --use-revin + ft_then_supervised")
    print("PCA is intentionally skipped: pca_not_supported_in_ft_then_supervised")
    print("Default tasks exclude old ETTh1 H=96 and Illness H=24; they are imported during summarization.")

    for dataset, horizon, seed, protocol_type, adapter_label in iter_experiments(args):
        key = (dataset, str(horizon), str(seed), protocol_type, adapter_label)
        if args.skip_existing and not args.force and key in existing:
            print(f"Skipping existing {key}")
            continue
        failure = run_one(
            dataset=dataset,
            horizon=horizon,
            seed=seed,
            protocol_type=protocol_type,
            adapter_label=adapter_label,
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
