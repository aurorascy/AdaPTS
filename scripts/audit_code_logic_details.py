from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import torch

from adapts import adapters
from adapts.utils.data_readers import DataReader


OUT_DIR = Path("results/code_logic_audit")
DATASETS = [
    ("Illness", 24),
    ("Illness", 60),
    ("ETTh1", 96),
    ("ETTh1", 192),
    ("Weather", 96),
    ("Weather", 192),
    ("ExchangeRate", 96),
    ("ExchangeRate", 192),
]
N_FEATURES = {"Illness": 7, "ETTh1": 7, "Weather": 21, "ExchangeRate": 8}
PROTOCOLS = [
    ("baseline", "baseline", None, False),
    ("strong-baseline", "strong-baseline", "revin", True),
    ("adapter", "linearAE", "linearAE", True),
    ("adapter", "dropoutLinearAE", "dropoutLinearAE", True),
    ("adapter", "linearVAE", "linearVAE", True),
    ("adapter", "VAE", "VAE", True),
    ("unsupported", "PCA", "pca", False),
]


def to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def data_shape_audit() -> pd.DataFrame:
    reader = DataReader(data_path=str(Path("external_data").resolve()) + "/", transform_ts_size=512, univariate=False)
    rows = []
    for dataset, horizon in DATASETS:
        path = Path(f"external_data/forecasting/{dataset}/{dataset}.csv")
        raw_shape = ""
        columns = ""
        missing = ""
        if path.exists():
            raw = pd.read_csv(path)
            raw_shape = str(tuple(raw.shape))
            columns = ", ".join(map(str, raw.columns))
            missing = int(raw.isna().sum().sum())
        try:
            name = f"{dataset}_pred={horizon}"
            train_x, train_y = reader.read_dataset(name, setting="train")
            val_x, val_y = reader.read_dataset(name, setting="val")
            test_x, test_y = reader.read_dataset(name, setting="test")
            notes = "ok"
        except Exception as exc:
            train_x = train_y = val_x = val_y = test_x = test_y = None
            notes = f"error: {type(exc).__name__}: {exc}"
        rows.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "raw_shape": raw_shape,
                "columns": columns,
                "missing_values": missing,
                "train_X_shape": str(tuple(train_x.shape)) if train_x is not None else "",
                "train_y_shape": str(tuple(train_y.shape)) if train_y is not None else "",
                "val_X_shape": str(tuple(val_x.shape)) if val_x is not None else "",
                "val_y_shape": str(tuple(val_y.shape)) if val_y is not None else "",
                "test_X_shape": str(tuple(test_x.shape)) if test_x is not None else "",
                "test_y_shape": str(tuple(test_y.shape)) if test_y is not None else "",
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def trainable_parameter_audit() -> pd.DataFrame:
    rows = []
    for dataset, n_features in N_FEATURES.items():
        for protocol_type, adapter_label, base_projector, use_revin in PROTOCOLS:
            try:
                projector = adapters.MultichannelProjector(
                    num_channels=n_features,
                    new_num_channels=n_features,
                    base_projector=base_projector,
                    device="cpu",
                    use_revin=use_revin,
                    context_length=512,
                    forecast_horizon=96,
                )
                module = projector.base_projector_
                if isinstance(module, torch.nn.Module):
                    params = [(name, p.numel()) for name, p in module.named_parameters() if p.requires_grad]
                    names = "; ".join(f"{name}:{num}" for name, num in params)
                    total = sum(num for _, num in params)
                    notes = "torch.nn.Module; trainable by adapter_supervised_fine_tuning"
                else:
                    names = ""
                    total = 0
                    notes = f"not a torch.nn.Module ({type(module).__name__}); unsupported for supervised adapter training"
                rows.append(
                    {
                        "dataset": dataset,
                        "protocol_type": protocol_type,
                        "adapter_label": adapter_label,
                        "base_projector_class": type(module).__name__,
                        "trainable_parameter_names": names,
                        "num_trainable_parameters": total,
                        "notes": notes,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "dataset": dataset,
                        "protocol_type": protocol_type,
                        "adapter_label": adapter_label,
                        "base_projector_class": "",
                        "trainable_parameter_names": "",
                        "num_trainable_parameters": "",
                        "notes": f"error: {type(exc).__name__}: {exc}",
                    }
                )
    return pd.DataFrame(rows)


def config_audit() -> pd.DataFrame:
    keys = [
        "dataset_name",
        "forecast_horizon",
        "adapter",
        "seed",
        "number_n_comp_to_try",
        "custom_n_comp",
        "use_revin",
        "supervised",
        "n_epochs_fine_tuning",
        "n_epochs_adapter",
        "pca_in_preprocessing",
        "model_name",
    ]
    rows = []
    for path in Path("logs").rglob("config.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"config_path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        row = {"config_path": str(path)}
        row.update({key: data.get(key, "") for key in keys})
        row["n_components"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def log_audit() -> pd.DataFrame:
    rows = []
    pattern_epoch = re.compile(r"Epoch (\d+): Train loss: ([0-9.]+), Val loss: ([0-9.]+)")
    for log_path in Path("logs").rglob("run_*.log"):
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        epochs = []
        for match in pattern_epoch.finditer(text):
            epochs.append((int(match.group(1)), float(match.group(2)), float(match.group(3))))
        adapter_fitted = "adapter fitted" in text
        done_ft = "Done fine tuning, now training adapter" in text
        starting = ""
        for line in text.splitlines():
            if "Starting fitting adapter:" in line:
                starting = line.split("Starting fitting adapter:", 1)[-1].strip()
                break
        early = "Early stopping" in text
        restore_epochs = re.findall(r"Restoring weights from epoch (\d+)", text)
        rows.append(
            {
                "log_path": str(log_path),
                "starting_adapter": starting,
                "fine_tune_epochs_logged": len(epochs),
                "fine_tune_train_loss_first": epochs[0][1] if epochs else "",
                "fine_tune_train_loss_last": epochs[-1][1] if epochs else "",
                "fine_tune_val_loss_first": epochs[0][2] if epochs else "",
                "fine_tune_val_loss_last": epochs[-1][2] if epochs else "",
                "fine_tune_val_loss_decreased": bool(epochs and epochs[-1][2] < epochs[0][2]),
                "done_ft_then_adapter_message": done_ft,
                "adapter_fitted_message": adapter_fitted,
                "early_stopping_message": early,
                "restore_epochs": ",".join(restore_epochs),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(data_shapes: pd.DataFrame, trainable: pd.DataFrame, logs: pd.DataFrame) -> None:
    (OUT_DIR / "data_pipeline_audit.md").write_text(
        "\n".join(
            [
                "# Data Pipeline Audit",
                "",
                "Forecasting CSVs are read from `external_data/forecasting/<Dataset>/<Dataset>.csv`.",
                "`DataReader` fits a `StandardScaler` on `train_df` only, then transforms train/val/test.",
                "Validation and test windows include a `seq_len` overlap before the split boundary to provide context; targets remain after the split boundary.",
                "",
                to_md(data_shapes),
            ]
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "adapter_training_audit.md").write_text(
        "\n".join(
            [
                "# Adapter Training Audit",
                "",
                "`adapter_supervised_fine_tuning` asserts that the base projector is a `torch.nn.Module`.",
                "This explains why sklearn PCA is incompatible with the current `ft_then_supervised` path.",
                "Named adapters and JustRevIn have trainable parameters; the head-only baseline's IdentityTransformer has none.",
                "",
                to_md(trainable),
            ]
        ),
        encoding="utf-8",
    )
    stage_logs = logs[logs["log_path"].str.contains("stage7_remaining_table1", na=False)]
    (OUT_DIR / "training_log_logic_audit.md").write_text(
        "\n".join(
            [
                "# Training Log Logic Audit",
                "",
                f"Log files inspected: {len(logs)}",
                f"Stage 7/8 log files inspected: {len(stage_logs)}",
                "",
                "The log files show head fine-tuning losses via `Epoch ... Train loss ... Val loss ...`.",
                "Adapter training writes TensorBoard scalars; the plain log records restore/early-stop messages and final adapter fitted status.",
                "",
                to_md(stage_logs.head(30)) if not stage_logs.empty else "No Stage 7/8 logs found.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_shapes = data_shape_audit()
    trainable = trainable_parameter_audit()
    configs = config_audit()
    logs = log_audit()
    data_shapes.to_csv(OUT_DIR / "data_shape_audit.csv", index=False)
    trainable.to_csv(OUT_DIR / "trainable_parameter_audit.csv", index=False)
    configs.to_csv(OUT_DIR / "config_audit_table.csv", index=False)
    logs.to_csv(OUT_DIR / "training_log_logic_audit.csv", index=False)
    write_markdown(data_shapes, trainable, logs)
    print(f"saved: {OUT_DIR / 'data_shape_audit.csv'}")
    print(f"saved: {OUT_DIR / 'trainable_parameter_audit.csv'}")
    print(f"saved: {OUT_DIR / 'config_audit_table.csv'}")
    print(f"saved: {OUT_DIR / 'training_log_logic_audit.csv'}")


if __name__ == "__main__":
    main()
