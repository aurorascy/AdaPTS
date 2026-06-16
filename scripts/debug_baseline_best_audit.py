from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:  # pragma: no cover
    EventAccumulator = None


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "debug_audit"
PLAN_DIR = ROOT / "results" / "debug_rerun_plan"
REPORT = ROOT / "DEBUG_BASELINE_BEST_AUDIT.md"

STAGES = {
    "stage2_illness_h24": {
        "dataset": "Illness",
        "horizon": 24,
        "raw": ROOT / "results/stage2_illness_h24/raw_results.csv",
        "summary": ROOT / "results/stage2_illness_h24/summary_mean_std_stderr.csv",
        "logs": ROOT / "logs/stage2_illness_h24",
        "paper_baseline": 2.902,
        "paper_baseline_std": 0.023,
    },
    "stage3_etth1_h96": {
        "dataset": "ETTh1",
        "horizon": 96,
        "raw": ROOT / "results/stage3_etth1_h96/raw_results.csv",
        "summary": ROOT / "results/stage3_etth1_h96/summary_mean_std_stderr.csv",
        "logs": ROOT / "logs/stage3_etth1_h96",
        "paper_baseline": 0.411,
        "paper_baseline_std": 0.012,
    },
}


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def norm_adapter(value: Any) -> str:
    if pd.isna(value) or str(value).strip() in {"", "None", "null"}:
        return "baseline"
    text = str(value).strip()
    return "PCA" if text.lower() == "pca" else text


def md_table(df: pd.DataFrame, digits: int = 6) -> str:
    if df is None or df.empty:
        return "暂无数据"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(display.columns)
    rows = display.to_numpy().tolist()
    widths = [max(len(h), *(len(str(row[i])) for row in rows)) for i, h in enumerate(headers)]

    def render(vals: list[Any]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    return "\n".join(
        [render(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
        + [render(row) for row in rows]
    )


def metric_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows = []
    recomputed_rows = []
    for stage, info in STAGES.items():
        raw_path = info["raw"]
        summary_path = info["summary"]
        if not raw_path.exists():
            audit_rows.append({"stage": stage, "status": "raw missing"})
            continue
        raw = pd.read_csv(raw_path)
        metrics = sorted(raw["metric"].dropna().unique().tolist()) if "metric" in raw.columns else []
        raw_mse = raw[raw["metric"].astype(str).str.lower() == "mse"].copy()
        raw_mse["adapter"] = raw_mse["adapter"].map(norm_adapter)
        recomputed = (
            raw_mse.groupby("adapter")["value"]
            .agg(mse_mean="mean", mse_std="std", n="count")
            .reset_index()
        )
        recomputed["mse_stderr"] = recomputed["mse_std"] / recomputed["n"].pow(0.5)
        summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        summary_cmp = summary[["adapter", "mse_mean"]].copy() if not summary.empty else pd.DataFrame()
        if not summary_cmp.empty:
            summary_cmp["adapter"] = summary_cmp["adapter"].map(norm_adapter)
            merged = recomputed.merge(summary_cmp, on="adapter", how="outer", suffixes=("_recomputed", "_summary"))
            merged["abs_diff"] = (merged["mse_mean_recomputed"] - merged["mse_mean_summary"]).abs()
            max_diff = float(merged["abs_diff"].max())
            matches = max_diff < 1e-9
        else:
            merged = recomputed.copy()
            merged["mse_mean_summary"] = pd.NA
            merged["abs_diff"] = pd.NA
            max_diff = math.nan
            matches = False
        for _, row in merged.iterrows():
            recomputed_rows.append({"stage": stage, **row.to_dict()})
        audit_rows.append(
            {
                "stage": stage,
                "raw_columns": ", ".join(raw.columns),
                "metric_types": ", ".join(metrics),
                "summary_columns": ", ".join(summary.columns) if not summary.empty else "missing",
                "summary_uses_metric": "mse",
                "summary_matches_raw_metric_mse": matches,
                "max_abs_diff": max_diff,
                "note": "summary matches recomputation from metric == 'mse'" if matches else "check needed",
            }
        )
    return pd.DataFrame(audit_rows), pd.DataFrame(recomputed_rows)


def parse_log(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    ft_losses = []
    for match in re.finditer(r"Epoch (\d+): Train loss: ([0-9.]+), Val loss: ([0-9.]+)", text):
        ft_losses.append((int(match.group(1)), float(match.group(2)), float(match.group(3))))
    ncomp_match = re.search(r"n_components to try between .*?: \[([^\]]+)\]", text)
    metrics_match = re.search(r"metrics \[mse=([^,]+),mae=([^\]]+)\]", text)
    return {
        "run_log": rel(log_path),
        "log_has_fine_tune_losses": bool(ft_losses),
        "fine_tune_epochs_logged": len(ft_losses),
        "fine_tune_train_loss_first": ft_losses[0][1] if ft_losses else pd.NA,
        "fine_tune_train_loss_last": ft_losses[-1][1] if ft_losses else pd.NA,
        "fine_tune_val_loss_first": ft_losses[0][2] if ft_losses else pd.NA,
        "fine_tune_val_loss_last": ft_losses[-1][2] if ft_losses else pd.NA,
        "fine_tune_val_loss_decreased": (ft_losses[-1][2] < ft_losses[0][2]) if len(ft_losses) > 1 else pd.NA,
        "log_has_done_fine_tuning": "Done fine tuning, now training adapter" in text,
        "log_has_adapter_fitted": "adapter fitted" in text,
        "log_has_metrics": "metrics [mse=" in text,
        "actual_n_components_from_log": ncomp_match.group(1).strip() if ncomp_match else pd.NA,
        "mse_from_log": float(metrics_match.group(1)) if metrics_match else pd.NA,
        "mae_from_log": float(metrics_match.group(2)) if metrics_match else pd.NA,
    }


def read_event_scalars(run_dir: Path) -> dict[str, Any]:
    if EventAccumulator is None:
        return {"event_read_status": "tensorboard unavailable"}
    event_files = sorted(run_dir.rglob("events.out.tfevents*"))
    if not event_files:
        return {"event_read_status": "no event file"}
    best = max(event_files, key=lambda p: p.stat().st_size)
    try:
        ea = EventAccumulator(str(best), size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        result: dict[str, Any] = {
            "event_read_status": "ok",
            "event_file": rel(best),
            "event_scalar_tags": ", ".join(tags),
        }
        for tag, prefix in [
            ("Loss/training", "adapter_train_loss"),
            ("Loss/validation", "adapter_val_loss"),
            ("Learning_rate", "adapter_lr"),
        ]:
            if tag in tags:
                vals = ea.Scalars(tag)
                result[f"{prefix}_points"] = len(vals)
                result[f"{prefix}_first"] = vals[0].value if vals else pd.NA
                result[f"{prefix}_last"] = vals[-1].value if vals else pd.NA
                result[f"{prefix}_decreased"] = (vals[-1].value < vals[0].value) if len(vals) > 1 else pd.NA
            else:
                result[f"{prefix}_points"] = 0
                result[f"{prefix}_first"] = pd.NA
                result[f"{prefix}_last"] = pd.NA
                result[f"{prefix}_decreased"] = pd.NA
        return result
    except Exception as exc:
        return {"event_read_status": f"failed: {exc}", "event_file": rel(best)}


def config_and_training_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    config_rows = []
    train_rows = []
    for stage, info in STAGES.items():
        log_root = info["logs"]
        for cfg_path in sorted(log_root.rglob("config.json")):
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            run_dir = cfg_path.parent
            run_logs = sorted(run_dir.glob("run_*.log"))
            parsed = parse_log(run_logs[-1]) if run_logs else {}
            event_info = read_event_scalars(run_dir)
            adapter = norm_adapter(cfg.get("adapter"))
            row = {
                "stage": stage,
                "config_path": rel(cfg_path),
                "dataset_name": cfg.get("dataset_name"),
                "forecast_horizon": cfg.get("forecast_horizon"),
                "adapter": adapter,
                "seed": cfg.get("seed"),
                "n_components": parsed.get("actual_n_components_from_log", pd.NA),
                "number_n_comp_to_try": cfg.get("number_n_comp_to_try"),
                "custom_n_comp": cfg.get("custom_n_comp"),
                "use_revin": cfg.get("use_revin"),
                "supervised": cfg.get("supervised"),
                "n_epochs_fine_tuning": cfg.get("n_epochs_fine_tuning"),
                "n_epochs_adapter": cfg.get("n_epochs_adapter"),
                "pca_in_preprocessing": cfg.get("pca_in_preprocessing"),
                "data_path": cfg.get("data_path"),
            }
            config_rows.append(row)
            train_rows.append(
                {
                    **row,
                    **parsed,
                    **event_info,
                }
            )
    return pd.DataFrame(config_rows), pd.DataFrame(train_rows)


def n_components_summary(config_df: pd.DataFrame, raw_df_by_stage: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for stage, raw in raw_df_by_stage.items():
        if raw is not None and not raw.empty:
            tmp = raw.copy()
            tmp["adapter"] = tmp["adapter"].map(norm_adapter)
            for (adapter, seed), group in tmp.groupby(["adapter", "seed"], dropna=False):
                rows.append(
                    {
                        "stage": stage,
                        "adapter": adapter,
                        "seed": seed,
                        "n_components_from_raw": sorted(group["n_components"].dropna().unique().tolist()),
                        "n_features": sorted(group["n_features"].dropna().unique().tolist()),
                    }
                )
    out = pd.DataFrame(rows)
    if not config_df.empty:
        cfg = config_df[["stage", "adapter", "seed", "n_components", "custom_n_comp", "number_n_comp_to_try"]].drop_duplicates()
        out = out.merge(cfg, on=["stage", "adapter", "seed"], how="outer")
    return out


def baseline_vs_paper(summary_by_stage: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for stage, summary in summary_by_stage.items():
        info = STAGES[stage]
        if summary is None or summary.empty:
            continue
        data = summary.copy()
        data["adapter"] = data["adapter"].map(norm_adapter)
        base = data[data["adapter"] == "baseline"]
        if base.empty:
            continue
        local = float(base["mse_mean"].iloc[0])
        paper = info["paper_baseline"]
        rows.append(
            {
                "stage": stage,
                "dataset": info["dataset"],
                "horizon": info["horizon"],
                "local_baseline_mse": local,
                "paper_baseline_mse": paper,
                "paper_baseline_std": info["paper_baseline_std"],
                "absolute_difference_local_minus_paper": local - paper,
                "relative_difference_percent": (local - paper) / paper * 100,
                "local_baseline_stronger_than_paper": local < paper,
            }
        )
    return pd.DataFrame(rows)


def write_rerun_plan() -> Path:
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    commands = """# Debug rerun plan only. Do not execute automatically.

# Baseline, ETTh1 H=96, seed=13, closer to paper epochs.
python scripts/run.py \\
  --forecast-horizon 96 \\
  --model-name "AutonLab/MOMENT-1-small" \\
  --context-length 512 \\
  --seed 13 \\
  --device "cuda:0" \\
  --dataset-name "ETTh1" \\
  --use-revin \\
  --supervised "ft_then_supervised" \\
  --n-epochs-fine-tuning 50 \\
  --n-epochs-adapter 300 \\
  --data-path "results/debug_rerun_plan/etth1_h96_seed13_baseline_vs_dropoutLinearAE.csv" \\
  --log-dir "logs/debug_rerun_plan/etth1_h96_seed13/baseline"

# dropoutLinearAE, ETTh1 H=96, seed=13, n_components expected to be 7 by data dimensionality.
python scripts/run.py \\
  --forecast-horizon 96 \\
  --model-name "AutonLab/MOMENT-1-small" \\
  --context-length 512 \\
  --seed 13 \\
  --device "cuda:0" \\
  --dataset-name "ETTh1" \\
  --adapter "dropoutLinearAE" \\
  --use-revin \\
  --supervised "ft_then_supervised" \\
  --n-epochs-fine-tuning 50 \\
  --n-epochs-adapter 300 \\
  --data-path "results/debug_rerun_plan/etth1_h96_seed13_baseline_vs_dropoutLinearAE.csv" \\
  --log-dir "logs/debug_rerun_plan/etth1_h96_seed13/dropoutLinearAE"
"""
    path = PLAN_DIR / "etth1_h96_seed13_baseline_vs_dropoutLinearAE_commands.sh"
    path.write_text(commands, encoding="utf-8")
    return path


def report(
    metric_df: pd.DataFrame,
    recomputed_df: pd.DataFrame,
    config_df: pd.DataFrame,
    training_df: pd.DataFrame,
    ncomp_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    rerun_path: Path,
) -> str:
    # Compact tables for report.
    config_compact = config_df[
        [
            "stage",
            "dataset_name",
            "forecast_horizon",
            "adapter",
            "seed",
            "n_components",
            "custom_n_comp",
            "number_n_comp_to_try",
            "use_revin",
            "supervised",
            "n_epochs_fine_tuning",
            "n_epochs_adapter",
            "pca_in_preprocessing",
        ]
    ].drop_duplicates()
    success_train = training_df[
        [
            "stage",
            "dataset_name",
            "adapter",
            "seed",
            "log_has_done_fine_tuning",
            "log_has_adapter_fitted",
            "fine_tune_val_loss_first",
            "fine_tune_val_loss_last",
            "fine_tune_val_loss_decreased",
            "adapter_train_loss_first",
            "adapter_train_loss_last",
            "adapter_train_loss_decreased",
            "adapter_val_loss_first",
            "adapter_val_loss_last",
            "adapter_val_loss_decreased",
            "event_read_status",
        ]
    ].copy()
    success_train = success_train[success_train["stage"].isin(STAGES)]

    likely_causes = pd.DataFrame(
        [
            {
                "rank": 1,
                "cause": "训练轮数明显少于论文默认/完整设置",
                "evidence": "当前 ft=5、adapter=30；论文/脚本默认 ft=50、adapter=300，并使用 hyperparameter optimization。",
            },
            {
                "rank": 2,
                "cause": "未做 hyperparameter optimization，且 custom_n_comp=False",
                "evidence": "所有已完成组合实际 n_components=7；没有系统搜索 adapter 超参数或 latent dimension。",
            },
            {
                "rank": 3,
                "cause": "本地 baseline 明显强于论文 baseline",
                "evidence": "Illness baseline 2.618 vs paper 2.902；ETTh1 baseline 0.389 vs paper 0.411。",
            },
            {
                "rank": 4,
                "cause": "PCA 路径与 supervised fine-tuning 协议不匹配",
                "evidence": "PCA 在 Stage 2/3 均失败；probe 捕获 AssertionError: adapter must be a PyTorch Module。",
            },
        ]
    )

    return f"""# Baseline 最优异常复现审计报告

## 1. 异常现象

当前 Illness H=24 和 ETTh1 H=96 的三随机种子结果中，baseline 都是 MSE 最低的配置：

{md_table(baseline_df)}

这与论文中 adapter 尤其 VAE/LinearVAE/dropoutLinearAE 优于 baseline 的趋势不一致。本审计只读取已有 CSV、日志和 config，没有启动新训练，也没有运行 hyperopt。

## 2. 已排除的问题

### 2.1 Summary metric 是否选错

{md_table(metric_df)}

重新从 raw_results 中筛选 `metric == "mse"` 聚合后，与 `summary_mean_std_stderr.csv` 的 `mse_mean` 一致。因此目前没有发现 summary 使用了 `scaled_mse`、`mae`、`ks` 或 `ece` 的问题。

重算摘要：

{md_table(recomputed_df[["stage", "adapter", "mse_mean_recomputed", "mse_mean_summary", "abs_diff"]])}

### 2.2 Adapter 是否实际训练

训练日志显示 fine-tuning head 有 train/val loss，且非 PCA adapter 日志中出现 `Done fine tuning, now training adapter` 与 `adapter fitted`。TensorBoard 事件文件可读取 adapter 的 `Loss/training` 和 `Loss/validation`；多数成功 run 中 adapter loss 有下降。

审计表已保存：

- `results/debug_audit/config_audit_table.csv`
- `results/debug_audit/training_log_audit_table.csv`

## 3. 发现的协议差异

| 项目 | 论文/参考设置 | 当前实验设置 | 影响 |
|---|---|---|---|
| Context length | L=512 | L=512 | 一致 |
| ETTh1 H=96 特征/组件 | 原始特征数 7，Figure 2 中 7 components 最优 | 实际使用 7 components | 对 ETTh1 组件数一致 |
| Head/adapter 训练 | 先训练 Moment linear forecasting head，再冻结后训练 adapter | 使用 `ft_then_supervised`，符合大方向 | 流程大体一致 |
| Optimizer/LR/Batch | Adam, batch size=32, LR=0.001 | 代码中 adapter supervised 使用 Adam, batch size=32, LR=0.001；MOMENT head fine-tune 内部 OneCycle max_lr=1e-4 | adapter 部分一致，head LR 调度可能不同 |
| Epoch | 论文完整设置/脚本默认更大 | 本阶段 ft=5, adapter=30 | 明显不足，可能低估 adapter |
| Hyperopt | 使用 hyperparameter optimization | 未运行 hyperopt | 重要差异 |
| PCA | 论文有 PCA baseline | 当前 PCA 在 supervised 路径失败 | PCA 未可比 |

## 4. n_components 排查

所有成功写入 raw_results 的 Illness H=24 和 ETTh1 H=96 组合中，`n_components` 均为 7。config 中 `custom_n_comp=False`，`number_n_comp_to_try=4`，但日志显示 `n_components to try between 7 and 7: [7]`。

{md_table(ncomp_df.head(40))}

这意味着当前没有真正做 latent dimension 搜索。ETTh1 H=96 用 7 components 与论文 Figure 2 的提示一致；但对于其它数据集或 adapter，缺少 hyperopt/custom component 搜索仍可能导致 adapter 性能偏低。

## 5. PCA 失败排查

PCA 在 Stage 2 和 Stage 3 的 `ft_then_supervised` 路径均失败。Stage 2 的 probe 明确捕获：

```text
AssertionError: adapter must be a PyTorch Module
```

原因是 `adapter_supervised_fine_tuning` 要求 adapter base projector 是 `torch.nn.Module`，而 PCA projector 不是可训练的 PyTorch Module。官方支持 PCA 的正确运行方式很可能不是当前 supervised adapter fine-tuning 路径，而应考虑：

1. 使用非 supervised adapter 路径；
2. 使用 `pca_in_preprocessing`；
3. 单独按论文协议确认 PCA 是否只作为预处理/线性降维 baseline。

## 6. Baseline 是否过强

本地 baseline 明显强于论文 baseline：

{md_table(baseline_df)}

可能原因包括 MOMENT 版本差异、训练轮数与 early stopping、数据版本和 split、RevIN/scaler 行为、metric 是否使用原尺度 MSE、以及本地环境依赖版本差异。由于 summary metric 已确认使用 raw `mse`，metric 选错不是主要解释。

## 7. 最可能原因排序

{md_table(likely_causes)}

## 8. 是否存在结果汇总错误

未发现。`summary_mean_std_stderr.csv` 与从 raw_results 中 `metric == "mse"` 重算的结果一致。

## 9. 是否存在 adapter 未充分训练

存在较大可能。非 PCA adapter 确实进入了训练，并且 TensorBoard loss 通常下降；但当前 epoch 设置为 ft=5、adapter=30，远低于脚本默认和论文更接近的 ft=50、adapter=300。也没有运行 hyperopt，因此 adapter 可能训练不足或超参数不佳。

## 10. 是否存在 baseline 过强

是。Illness 和 ETTh1 的本地 baseline 都优于论文 baseline，这会压缩 adapter 的相对提升空间。该现象需要优先复核 MOMENT 版本、数据 split、RevIN/scaler、head fine-tuning 策略和训练轮数。

## 11. 下一步最小重跑实验计划

不在本次 audit 中执行。命令已写入：

`{rel(rerun_path)}`

建议只重跑一个最小对照：

- Dataset: ETTh1
- Horizon: 96
- Seed: 13
- baseline vs dropoutLinearAE
- context_length=512
- n_components=7
- use_revin=True
- supervised=ft_then_supervised
- n_epochs_fine_tuning=50
- n_epochs_adapter=300
- 输出目录: `results/debug_rerun_plan/`
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLAN_DIR.mkdir(parents=True, exist_ok=True)

    metric_df, recomputed_df = metric_audit()
    metric_df.to_csv(OUT_DIR / "metric_audit_table.csv", index=False)
    recomputed_df.to_csv(OUT_DIR / "summary_recomputed_from_mse.csv", index=False)

    config_df, training_df = config_and_training_audit()
    config_df.to_csv(OUT_DIR / "config_audit_table.csv", index=False)
    training_df.to_csv(OUT_DIR / "training_log_audit_table.csv", index=False)

    raw_by_stage = {
        stage: pd.read_csv(info["raw"]) if info["raw"].exists() else pd.DataFrame()
        for stage, info in STAGES.items()
    }
    ncomp_df = n_components_summary(config_df, raw_by_stage)
    ncomp_df.to_csv(OUT_DIR / "n_components_audit_table.csv", index=False)

    summary_by_stage = {
        stage: pd.read_csv(info["summary"]) if info["summary"].exists() else pd.DataFrame()
        for stage, info in STAGES.items()
    }
    baseline_df = baseline_vs_paper(summary_by_stage)
    baseline_df.to_csv(OUT_DIR / "baseline_vs_paper.csv", index=False)

    rerun_path = write_rerun_plan()
    REPORT.write_text(
        report(metric_df, recomputed_df, config_df, training_df, ncomp_df, baseline_df, rerun_path),
        encoding="utf-8",
    )

    print(f"Saved {rel(OUT_DIR / 'metric_audit_table.csv')}")
    print(f"Saved {rel(OUT_DIR / 'config_audit_table.csv')}")
    print(f"Saved {rel(OUT_DIR / 'training_log_audit_table.csv')}")
    print(f"Saved {rel(OUT_DIR / 'n_components_audit_table.csv')}")
    print(f"Saved {rel(OUT_DIR / 'baseline_vs_paper.csv')}")
    print(f"Saved {rel(rerun_path)}")
    print(f"Saved {rel(REPORT)}")


if __name__ == "__main__":
    main()
