import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


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

RESULT_DIR = Path("results/stage2_illness_h24")
LOG_ROOT = Path("logs/stage2_illness_h24")
RAW_RESULTS = RESULT_DIR / "raw_results.csv"
FAILED_RUNS = RESULT_DIR / "failed_runs.json"


def select_device_and_epochs() -> tuple[str, int, int]:
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    if cuda_available:
        return "cuda:0", 5, 30
    return "cpu", 1, 5


def adapter_log_name(adapter: str | None) -> str:
    return "baseline" if adapter is None else adapter


def adapter_result_name(adapter: str | None) -> str:
    return "baseline" if adapter is None else ADAPTER_CLI_NAMES[adapter]


def existing_successes() -> set[tuple[int, str]]:
    if not RAW_RESULTS.exists():
        return set()

    try:
        import pandas as pd
    except ImportError:
        return set()

    df = pd.read_csv(RAW_RESULTS)
    required = {"seed", "adapter", "metric"}
    if not required.issubset(df.columns):
        return set()

    data = df.copy()
    data["adapter"] = data["adapter"].fillna("baseline").replace("", "baseline")
    successes: set[tuple[int, str]] = set()
    for (seed, adapter), group in data.groupby(["seed", "adapter"], dropna=False):
        metrics = set(group["metric"].astype(str))
        if {"mse", "mae"}.issubset(metrics):
            seed_value = int(seed) if not isinstance(seed, float) or not math.isnan(seed) else -1
            successes.add((seed_value, str(adapter)))
    return successes


def format_command(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def build_command(
    seed: int,
    adapter: str | None,
    device: str,
    n_epochs_fine_tuning: int,
    n_epochs_adapter: int,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run.py",
        "--forecast-horizon",
        "24",
        "--model-name",
        "AutonLab/MOMENT-1-small",
        "--context-length",
        "512",
        "--seed",
        str(seed),
        "--device",
        device,
        "--dataset-name",
        "Illness",
        "--use-revin",
        "--supervised",
        "ft_then_supervised",
        "--n-epochs-fine-tuning",
        str(n_epochs_fine_tuning),
        "--n-epochs-adapter",
        str(n_epochs_adapter),
        "--data-path",
        str(RAW_RESULTS),
        "--log-dir",
        str(LOG_ROOT / f"seed_{seed}" / adapter_log_name(adapter)),
    ]

    adapter_cli_name = ADAPTER_CLI_NAMES[adapter]
    if adapter_cli_name is not None:
        command.extend(["--adapter", adapter_cli_name])

    return command


def load_failed_runs() -> list[dict]:
    if not FAILED_RUNS.exists():
        return []
    with open(FAILED_RUNS, "r", encoding="utf-8") as f:
        return json.load(f)


def save_failed_runs(failed_runs: list[dict]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(FAILED_RUNS, "w", encoding="utf-8") as f:
        json.dump(failed_runs, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run commands even if raw_results.csv already contains mse/mae for the seed/adapter.",
    )
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    device, n_epochs_fine_tuning, n_epochs_adapter = select_device_and_epochs()
    print(f"device={device}")
    print(f"n_epochs_fine_tuning={n_epochs_fine_tuning}")
    print(f"n_epochs_adapter={n_epochs_adapter}")
    print(f"raw_results={RAW_RESULTS}")
    print(f"failed_runs={FAILED_RUNS}")

    successes = set() if args.force else existing_successes()
    failed_runs = load_failed_runs()

    for seed in SEEDS:
        for adapter in ADAPTERS:
            result_adapter = adapter_result_name(adapter)
            if (seed, result_adapter) in successes:
                print(f"\nSKIP existing successful run: seed={seed}, adapter={result_adapter}")
                continue

            command = build_command(
                seed=seed,
                adapter=adapter,
                device=device,
                n_epochs_fine_tuning=n_epochs_fine_tuning,
                n_epochs_adapter=n_epochs_adapter,
            )
            print("\n" + "=" * 100)
            print(f"seed={seed}, adapter={adapter_log_name(adapter)}")
            print(format_command(command))

            if args.dry_run:
                continue

            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                failure = {
                    "seed": seed,
                    "adapter": adapter_log_name(adapter),
                    "adapter_cli": ADAPTER_CLI_NAMES[adapter],
                    "returncode": completed.returncode,
                    "command": command,
                }
                failed_runs.append(failure)
                save_failed_runs(failed_runs)
                print(f"FAILED: {failure}")
            else:
                print(f"SUCCESS: seed={seed}, adapter={adapter_log_name(adapter)}")

    save_failed_runs(failed_runs)
    print("\nStage 2 batch run finished.")
    print(f"Failures recorded: {len(failed_runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
