import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from adapts.icl.moment import load_moment_model
from adapts.utils.main_script import prepare_data


def append_metric_rows(
    output_path: Path,
    *,
    dataset: str,
    horizon: int,
    context_length: int,
    seed: int,
    model_name: str,
    device: str,
    n_features: int,
    train_size: int,
    running_time: float,
    metrics: dict[str, float],
    notes: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
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
    file_exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        for metric, value in metrics.items():
            writer.writerow(
                {
                    "stage": "stage5_baseline_separation",
                    "dataset": dataset,
                    "horizon": horizon,
                    "context_length": context_length,
                    "seed": seed,
                    "baseline_type": "B0_pure_moment",
                    "description": "Pure MOMENT zero-shot; no ADAPTS wrapper, no RevIN, no fine-tuning",
                    "adapter": "",
                    "use_revin": False,
                    "supervised": "zero_shot",
                    "n_epochs_fine_tuning": 0,
                    "n_epochs_adapter": 0,
                    "n_components": n_features,
                    "metric": metric,
                    "value": value,
                    "notes": notes,
                    "source": "evaluate_pure_moment_baseline.py",
                    "model_name": model_name,
                    "device": device,
                    "n_features": n_features,
                    "train_size": train_size,
                    "running_time": running_time,
                    "pca_in_preprocessing": False,
                    "is_fine_tuned": False,
                }
            )


def predict_univariate_channels(
    model,
    X_test: np.ndarray,
    *,
    device: str,
    batch_size: int,
    verbose: bool,
) -> np.ndarray:
    model.eval()
    predictions = []
    n_samples, n_features, _ = X_test.shape

    with torch.no_grad():
        for feature_idx in range(n_features):
            feature_preds = []
            x_feature = torch.from_numpy(X_test[:, feature_idx, :]).float().unsqueeze(1)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch = x_feature[start:end].to(device)
                output = model(x_enc=batch).forecast
                feature_preds.append(output.detach().cpu().numpy()[:, 0, :])
            predictions.append(np.concatenate(feature_preds, axis=0))
            if verbose:
                print(f"predicted feature {feature_idx + 1}/{n_features}")

    return np.stack(predictions, axis=1)


def evaluate(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    start_time = time.time()
    dataset_name = f"{args.dataset}_pred={args.horizon}"
    X_train, y_train, _X_val, _y_val, X_test, y_test, n_features = prepare_data(
        dataset_name, args.context_length, args.horizon
    )
    train_size = len(X_train)

    model = load_moment_model(args.model_name, args.horizon).to(torch.device(args.device))
    y_pred = predict_univariate_channels(
        model,
        X_test,
        device=args.device,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    mse = float(np.mean((y_pred - y_test) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_test)))
    elapsed = time.time() - start_time

    notes = (
        "Uses AdaPTS DataReader standardized forecasting data; bypasses ADAPTS "
        "wrapper/scaler and does no training."
    )
    append_metric_rows(
        args.output,
        dataset=args.dataset,
        horizon=args.horizon,
        context_length=args.context_length,
        seed=args.seed,
        model_name=args.model_name,
        device=args.device,
        n_features=n_features,
        train_size=train_size,
        running_time=elapsed,
        metrics={"mse": mse, "mae": mae},
        notes=notes,
    )

    print(f"B0 pure MOMENT done: mse={mse:.6f}, mae={mae:.6f}, seconds={elapsed:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pure zero-shot MOMENT baseline.")
    parser.add_argument("--dataset", default="ETTh1", choices=["ETTh1", "Illness"])
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--model_name", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage5_baseline_separation/raw_results.csv"),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.horizon is None:
        args.horizon = 96 if args.dataset == "ETTh1" else 24
    return args


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
