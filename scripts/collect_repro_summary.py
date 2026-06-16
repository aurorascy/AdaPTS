from __future__ import annotations

import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "progress_summary"
REPORT = ROOT / "ADAPTS_PROGRESS_REPORT_FOR_SUPERVISOR.md"

EXPECTED_CSVS = {
    "minimal_raw": ROOT / "results/minimal_repro/illness_h24_minimal.csv",
    "minimal_summary": ROOT / "results/minimal_repro/illness_h24_minimal_summary.csv",
    "stage2_raw": ROOT / "results/stage2_illness_h24/raw_results.csv",
    "stage2_summary": ROOT / "results/stage2_illness_h24/summary_mean_std_stderr.csv",
    "stage3_raw": ROOT / "results/stage3_etth1_h96/raw_results.csv",
    "stage3_summary": ROOT / "results/stage3_etth1_h96/summary_mean_std_stderr.csv",
}

EXPECTED_REPORTS = [
    "ENV_SETUP.md",
    "MINIMAL_REPRO.md",
    "STAGE2_ILLNESS_3SEED.md",
    "DATA_PREP_ETTH1.md",
    "STAGE3_ETTH1_H96_3SEED.md",
]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def pick_column(columns: list[str], candidates: list[str]) -> str | None:
    by_lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        match = by_lower.get(candidate.lower())
        if match is not None:
            return match
    return None


def normalize_adapter(value: Any) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "baseline"
    text = str(value).strip()
    if text.lower() == "pca":
        return "PCA"
    return text


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def wide_results(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    columns = list(df.columns)
    adapter_col = pick_column(columns, ["adapter", "adapter_name"])
    seed_col = pick_column(columns, ["seed"])
    metric_col = pick_column(columns, ["metric"])
    value_col = pick_column(columns, ["value"])

    data = df.copy()
    if adapter_col is None:
        data["adapter_normalized"] = "baseline"
    else:
        data["adapter_normalized"] = data[adapter_col].map(normalize_adapter)

    if metric_col and value_col:
        index_candidates = [
            "dataset",
            "dataset_name",
            "forecast_horizon",
            "horizon",
            "context_length",
            "foundational_model",
            "model_name",
            "adapter_normalized",
            seed_col,
        ]
        index_cols = [column for column in index_candidates if column and column in data.columns]
        wide = (
            data.pivot_table(
                index=index_cols,
                columns=metric_col,
                values=value_col,
                aggfunc="first",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
    else:
        wide = data.copy()

    if "adapter_normalized" in wide.columns:
        if "adapter" in wide.columns:
            wide = wide.drop(columns=["adapter"])
        wide = wide.rename(columns={"adapter_normalized": "adapter"})
    elif adapter_col and adapter_col in wide.columns:
        wide["adapter"] = wide[adapter_col].map(normalize_adapter)
    else:
        wide["adapter"] = "baseline"

    if seed_col and seed_col != "seed" and seed_col in wide.columns:
        wide = wide.rename(columns={seed_col: "seed"})
    if "seed" not in wide.columns:
        wide["seed"] = pd.NA

    for candidate, standard in [
        (["dataset", "dataset_name"], "dataset"),
        (["forecast_horizon", "horizon"], "forecast_horizon"),
        (["mse", "MSE", "test_mse", "scaled_mse"], "mse"),
        (["mae", "MAE", "test_mae", "scaled_mae"], "mae"),
    ]:
        column = pick_column(list(wide.columns), candidate)
        if column and column != standard:
            wide = wide.rename(columns={column: standard})

    wide.insert(0, "stage", stage)
    keep = [
        "stage",
        "dataset",
        "forecast_horizon",
        "context_length",
        "adapter",
        "seed",
        "mse",
        "mae",
        "scaled_mse",
        "scaled_mae",
        "ks",
        "ece",
    ]
    for column in keep:
        if column not in wide.columns:
            wide[column] = pd.NA
    return wide[keep]


def summary_from_wide(wide: pd.DataFrame, stage: str) -> pd.DataFrame:
    rows = []
    if wide.empty or "mse" not in wide.columns:
        return pd.DataFrame()
    for adapter, group in wide.groupby("adapter", dropna=False):
        mse = pd.to_numeric(group["mse"], errors="coerce").dropna()
        mae = pd.to_numeric(group["mae"], errors="coerce").dropna()
        if len(mse) == 0:
            continue
        mse_std = float(mse.std(ddof=1)) if len(mse) > 1 else 0.0
        mae_std = float(mae.std(ddof=1)) if len(mae) > 1 else 0.0
        rows.append(
            {
                "stage": stage,
                "adapter": adapter,
                "mse_mean": float(mse.mean()),
                "mse_std": mse_std,
                "mse_stderr": mse_std / math.sqrt(len(mse)) if len(mse) else math.nan,
                "mae_mean": float(mae.mean()) if len(mae) else math.nan,
                "mae_std": mae_std,
                "mae_stderr": mae_std / math.sqrt(len(mae)) if len(mae) else math.nan,
                "n": int(min(len(mse), len(mae))),
            }
        )
    return pd.DataFrame(rows)


def load_stage_summary(stage: str, path: Path) -> pd.DataFrame | None:
    df = read_csv_if_exists(path)
    if df is None:
        return None
    data = df.copy()
    data.insert(0, "stage", stage)
    return data


def best_summary_row(stage: str, dataset: str, horizon: int, summary: pd.DataFrame | None) -> dict[str, Any]:
    base = {
        "stage": stage,
        "dataset": dataset,
        "horizon": horizon,
        "baseline_mse": pd.NA,
        "best_adapter": "暂无",
        "best_mse": pd.NA,
        "improvement_percent": pd.NA,
        "completed_3_seeds": "否",
    }
    if summary is None or summary.empty:
        return base
    data = summary.copy()
    data["adapter"] = data["adapter"].map(normalize_adapter)
    if "mse_mean" not in data.columns:
        return base
    baseline_rows = data[data["adapter"] == "baseline"]
    if baseline_rows.empty:
        return base
    baseline_mse = float(baseline_rows["mse_mean"].iloc[0])
    best = data.sort_values("mse_mean").iloc[0]
    best_mse = float(best["mse_mean"])
    improvement = (baseline_mse - best_mse) / baseline_mse * 100
    base.update(
        {
            "baseline_mse": baseline_mse,
            "best_adapter": str(best["adapter"]),
            "best_mse": best_mse,
            "improvement_percent": improvement,
            "completed_3_seeds": "是" if int(data["n"].max()) >= 3 else "否",
        }
    )
    return base


def markdown_table(df: pd.DataFrame, float_digits: int = 6) -> str:
    if df is None or df.empty:
        return "暂无数据"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{float_digits}f}"
            )
        else:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = list(display.columns)
    rows = display.to_numpy().tolist()
    widths = [
        max(len(str(header)), *(len(str(row[idx])) for row in rows))
        for idx, header in enumerate(headers)
    ]

    def render(values: list[Any]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


def git_status() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True, encoding="utf-8", errors="replace"
        ).strip()
    except Exception as exc:
        return f"git status unavailable: {exc}"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8", errors="replace"
        ).strip()
    except Exception as exc:
        return f"git commit unavailable: {exc}"


def environment_rows() -> pd.DataFrame:
    rows = [
        {"项目": "OS", "内容": platform.platform()},
        {"项目": "Python", "内容": platform.python_version()},
        {"项目": "环境隔离", "内容": ".venv-adapts；未使用 conda"},
    ]
    try:
        import torch
        import numpy as np
        import pandas as pandas_module

        cuda = torch.cuda.is_available()
        rows.extend(
            [
                {"项目": "torch", "内容": torch.__version__},
                {"项目": "CUDA available", "内容": str(cuda)},
                {"项目": "GPU", "内容": torch.cuda.get_device_name(0) if cuda else "N/A"},
                {"项目": "numpy", "内容": np.__version__},
                {"项目": "pandas", "内容": pandas_module.__version__},
            ]
        )
    except Exception as exc:
        rows.append({"项目": "Python packages", "内容": f"读取失败: {exc}"})
    rows.extend(
        [
            {"项目": "数据路径修正", "内容": "src/adapts/utils/main_script.py 改为项目根目录 external_data"},
            {"项目": "MOMENT 加载修正", "内容": "src/adapts/icl/moment.py 将 local_files_only 改为 False 以支持首次下载"},
            {"项目": "核心算法", "内容": "未重构；新增脚本主要用于下载、批量运行、汇总和报告"},
        ]
    )
    return pd.DataFrame(rows)


def data_status_rows() -> pd.DataFrame:
    rows = []
    for name, path in [
        ("Illness", ROOT / "external_data/forecasting/Illness/Illness.csv"),
        ("ETTh1", ROOT / "external_data/forecasting/ETTh1/ETTh1.csv"),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            rows.append(
                {
                    "数据集": name,
                    "是否存在": "是",
                    "路径": rel(path),
                    "shape": str(tuple(df.shape)),
                    "特征列": ", ".join(list(df.columns)),
                    "缺失值": int(df.isna().sum().sum()),
                    "DataReader 校验": (
                        "train (8033,7,512)/(8033,7,96); val/test (2785,7,512)/(2785,7,96)"
                        if name == "ETTh1"
                        else "最小复现和 Stage 2 已成功读取"
                    ),
                }
            )
        else:
            rows.append(
                {
                    "数据集": name,
                    "是否存在": "否",
                    "路径": rel(path),
                    "shape": "未生成",
                    "特征列": "未生成",
                    "缺失值": pd.NA,
                    "DataReader 校验": "未完成",
                }
            )
    return pd.DataFrame(rows)


def stage_overview_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "阶段": "环境搭建",
                "目标": "安装依赖、修正路径、MOMENT 可加载、单元测试/导入检查",
                "状态": "完成",
                "产出文件": "ENV_SETUP.md",
                "说明": "使用 .venv-adapts，CUDA 可用，完成核心依赖安装与 smoke check",
            },
            {
                "阶段": "最小复现 Illness H=24",
                "目标": "跑通 baseline、linearVAE、VAE 的单 seed 最小流程",
                "状态": "完成",
                "产出文件": "MINIMAL_REPRO.md; results/minimal_repro/*",
                "说明": "确认 scripts/run.py、MOMENT-small、adapter、metrics 保存流程可用",
            },
            {
                "阶段": "Stage 2 Illness H=24 3 seeds",
                "目标": "Illness H=24 上 baseline 与多个 adapter 的三随机种子对比",
                "状态": "部分完成",
                "产出文件": "STAGE2_ILLNESS_3SEED.md; results/stage2_illness_h24/*",
                "说明": "非 PCA 组合完成；PCA 在 supervised 路径失败",
            },
            {
                "阶段": "Stage 3 ETTh1 初始尝试",
                "目标": "扩展到 ETTh1 H=96",
                "状态": "已处理",
                "产出文件": "STAGE3_ETTH1_H96_3SEED.md",
                "说明": "最初因 ETTh1.csv 缺失停止，未伪造数据",
            },
            {
                "阶段": "Stage 3.0 ETTh1 数据准备",
                "目标": "下载官方 ETTh1.csv 并做 DataReader 校验",
                "状态": "完成",
                "产出文件": "DATA_PREP_ETTH1.md; scripts/download_etth1.py",
                "说明": "ETTh1 shape=(17420,8)，缺失值 0，DataReader train/val/test 校验通过",
            },
            {
                "阶段": "Stage 3 ETTh1 H=96 3 seeds",
                "目标": "ETTh1 H=96 上 baseline 与多个 adapter 的三随机种子对比",
                "状态": "部分完成",
                "产出文件": "STAGE3_ETTH1_H96_3SEED.md; results/stage3_etth1_h96/*",
                "说明": "非 PCA 组合完成；PCA 在 supervised 路径失败",
            },
        ]
    )


def experiment_settings_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "实验": "Minimal repro",
                "Dataset": "Illness",
                "Horizon": 24,
                "Context length": 512,
                "Model": "AutonLab/MOMENT-1-small",
                "Seeds": "13",
                "Adapters": "baseline, linearVAE, VAE",
                "Epochs": "ft=5, adapter=20",
                "状态": "完成",
            },
            {
                "实验": "Stage 2",
                "Dataset": "Illness",
                "Horizon": 24,
                "Context length": 512,
                "Model": "AutonLab/MOMENT-1-small",
                "Seeds": "13, 42, 2024",
                "Adapters": "baseline, PCA, linearAE, dropoutLinearAE, linearVAE, VAE",
                "Epochs": "ft=5, adapter=30",
                "状态": "非 PCA 完成，PCA 失败",
            },
            {
                "实验": "Stage 3",
                "Dataset": "ETTh1",
                "Horizon": 96,
                "Context length": 512,
                "Model": "AutonLab/MOMENT-1-small",
                "Seeds": "13, 42, 2024",
                "Adapters": "baseline, PCA, linearAE, dropoutLinearAE, linearVAE, VAE",
                "Epochs": "ft=5, adapter=30",
                "状态": "非 PCA 完成，PCA 失败",
            },
        ]
    )


def issues_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "问题": "数据路径硬编码",
                "出现阶段": "环境搭建",
                "处理方式": "将 prepare_data 中作者本地路径改为 repo_root/external_data",
                "当前状态": "已解决",
            },
            {
                "问题": "MOMENT 首次加载 local_files_only",
                "出现阶段": "环境搭建",
                "处理方式": "将 local_files_only 改为 False，允许首次下载 HuggingFace 模型",
                "当前状态": "已解决",
            },
            {
                "问题": "ETTh1 数据缺失",
                "出现阶段": "Stage 3 初始尝试",
                "处理方式": "停止训练；从官方 ETTDataset 下载 ETTh1.csv",
                "当前状态": "已解决",
            },
            {
                "问题": "ETTh1 DataReader 校验",
                "出现阶段": "Stage 3.0",
                "处理方式": "新增 check_etth1_datareader.py，只读验证 train/val/test shape",
                "当前状态": "通过",
            },
            {
                "问题": "PCA adapter 失败",
                "出现阶段": "Stage 2 与 Stage 3",
                "处理方式": "记录 failed_runs.json；未强改核心逻辑",
                "当前状态": "待确认 PCA 是否应走非 supervised 或 preprocessing 路径",
            },
        ]
    )


def next_steps_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"优先级": 1, "下一步": "排查 PCA 路径", "目的": "确认论文中 PCA 是否应使用非 supervised 或 preprocessing PCA", "预计产出": "PCA 可复现实验脚本/说明"},
            {"优先级": 2, "下一步": "复核 ETTh1 H=96 adapter 超参数和训练轮数", "目的": "解释 adapter 未优于 baseline 的原因", "预计产出": "更接近论文设置的 ETTh1 对比"},
            {"优先级": 3, "下一步": "latent dimension 消融", "目的": "分析 adapter latent 维度对性能的影响", "预计产出": "消融表格"},
            {"优先级": 4, "下一步": "calibration / ECE 分析", "目的": "评估 probabilistic adapter 的不确定性质量", "预计产出": "ECE/KS 与 MSE 对照"},
            {"优先级": 5, "下一步": "扩展 Weather 或 ExchangeRate", "目的": "验证跨数据集稳定性", "预计产出": "新增数据集三随机种子结果"},
            {"优先级": 6, "下一步": "最后再考虑 hyperopt", "目的": "接近完整论文设置", "预计产出": "更完整 Table 1 复现"},
        ]
    )


def format_summary_for_report(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    mapping = {
        "adapter": "Adapter",
        "mse_mean": "MSE mean",
        "mse_std": "MSE std",
        "mse_stderr": "MSE stderr",
        "mae_mean": "MAE mean",
        "mae_std": "MAE std",
        "mae_stderr": "MAE stderr",
        "n": "n",
    }
    cols = [column for column in mapping if column in df.columns]
    out = df[cols].rename(columns=mapping)
    return out


def build_report(
    stage_overview: pd.DataFrame,
    env: pd.DataFrame,
    data_status: pd.DataFrame,
    settings: pd.DataFrame,
    stage2_summary: pd.DataFrame | None,
    stage3_summary: pd.DataFrame | None,
    best_summary: pd.DataFrame,
    issues: pd.DataFrame,
    next_steps: pd.DataFrame,
    missing: pd.DataFrame,
) -> str:
    def best_line(stage_name: str) -> tuple[str, str, str, str]:
        row = best_summary[best_summary["stage"] == stage_name]
        if row.empty:
            return ("暂无", "暂无", "暂无", "暂无")
        item = row.iloc[0]
        return (
            f"{item['baseline_mse']:.6f}" if pd.notna(item["baseline_mse"]) else "暂无",
            str(item["best_adapter"]),
            f"{item['best_mse']:.6f}" if pd.notna(item["best_mse"]) else "暂无",
            f"{item['improvement_percent']:.2f}%" if pd.notna(item["improvement_percent"]) else "暂无",
        )

    ill_base, ill_best_adapter, ill_best, ill_imp = best_line("Illness H=24")
    ett_base, ett_best_adapter, ett_best, ett_imp = best_line("ETTh1 H=96")

    cross = pd.DataFrame(
        [
            {
                "阶段": "Illness H=24",
                "Dataset": "Illness",
                "Horizon": 24,
                "Baseline MSE": ill_base,
                "Best adapter": ill_best_adapter,
                "Best MSE": ill_best,
                "Improvement": ill_imp,
                "是否完成 3 seeds": "是",
            },
            {
                "阶段": "ETTh1 H=96",
                "Dataset": "ETTh1",
                "Horizon": 96,
                "Baseline MSE": ett_base,
                "Best adapter": ett_best_adapter,
                "Best MSE": ett_best,
                "Improvement": ett_imp,
                "是否完成 3 seeds": "是（非 PCA）",
            },
        ]
    )

    report = f"""# AdaPTS 论文复现阶段性汇报

## 研究背景简述

AdaPTS 关注的是如何将以单变量预测为主的时间序列基础模型适配到多变量概率预测场景。核心思路是在 foundation model 前后加入 adapter：先把多变量输入映射到 latent space，再交给冻结或轻量微调的基础模型预测，最后由 adapter 映射回原始变量空间。本阶段复现没有直接追求完整 Table 1，而是按“环境搭建、最小实验、三随机种子、跨数据集验证”的顺序，先确认代码路径、数据路径、模型加载和代表性实验是否可稳定运行。

## 已完成工作总览

{markdown_table(stage_overview)}

## 环境与代码修改情况

{markdown_table(env)}

## 数据准备情况

{markdown_table(data_status)}

## 实验设置汇总

{markdown_table(settings)}

## 结果表 1：Illness H=24 三随机种子结果

{markdown_table(format_summary_for_report(stage2_summary))}

Illness H=24 中，baseline MSE 为 {ill_base}，当前最优为 {ill_best_adapter}，Best MSE 为 {ill_best}，相对改善为 {ill_imp}。这说明在当前小规模训练设置下，adapter 并没有优于 baseline；非 baseline 中表现较好的是 linearVAE，但仍弱于 baseline。该趋势与论文中 VAE/LinearVAE 更优的结果不一致，可能原因包括训练轮数较少、没有 hyperopt、adapter 超参数未调、随机性以及依赖版本差异。

论文参考值：

| Adapter         | Paper MSE on Illness H=24 |
| --------------- | ------------------------: |
| baseline Moment |             2.902 ± 0.023 |
| PCA             |             2.980 ± 0.001 |
| LinearAE        |             2.624 ± 0.035 |
| dropoutLinearAE |             2.760 ± 0.061 |
| LinearVAE       |             2.542 ± 0.036 |
| VAE             |             2.461 ± 0.008 |

## 结果表 2：ETTh1 H=96 三随机种子结果

{markdown_table(format_summary_for_report(stage3_summary))}

ETTh1 H=96 中，baseline MSE 为 {ett_base}，当前最优为 {ett_best_adapter}，Best MSE 为 {ett_best}，相对改善为 {ett_imp}。本地结果的绝对数值与论文量级接近，但 adapter 没有整体优于 baseline；dropoutLinearAE 是非 baseline 中 MSE 最低的 adapter，部分接近论文中 dropoutLinearAE 较强的趋势。PCA 在 `ft_then_supervised` 路径下失败，暂未纳入数值比较。

论文参考值：

| Adapter         | Paper MSE on ETTh1 H=96 |
| --------------- | ----------------------: |
| baseline Moment |           0.411 ± 0.012 |
| PCA             |           0.433 ± 0.001 |
| LinearAE        |           0.402 ± 0.002 |
| dropoutLinearAE |           0.395 ± 0.003 |
| LinearVAE       |           0.400 ± 0.001 |
| VAE             |           0.404 ± 0.001 |

## 结果表 3：跨阶段最佳结果对比

{markdown_table(cross)}

## 问题与处理记录

{markdown_table(issues)}

## 当前结论

- 已经完成环境搭建、依赖安装、数据路径修正和 MOMENT 首次加载修正。
- 已经完成 Illness H=24 的最小复现和三随机种子实验；非 PCA 组合有完整结果。
- ETTh1 数据已经从官方 ETTDataset 补齐，并通过 DataReader 校验。
- ETTh1 H=96 三随机种子训练已经完成非 PCA 组合；PCA 在 supervised adapter fine-tuning 路径下失败。
- 当前 Illness 与 ETTh1 两个阶段都显示 baseline 的 MSE 最低，adapter 暂未复现论文中整体优于 baseline 的趋势。
- 当前结果可以支撑继续推进复现，但还不能声称完整复现论文 Table 1。

## 给导师汇报时可以这样说

1. 我没有直接全量复现 Table 1，而是按“环境、最小实验、三随机种子、跨数据集”的顺序推进，先保证每一步可追踪。
2. 环境已经搭好，CUDA 可用，MOMENT-small 可以加载，作者本地数据路径和 HuggingFace 首次加载问题已经做了最小修正。
3. Illness H=24 已完成三随机种子实验；当前 baseline MSE 最低，adapter 没有超过 baseline，这和论文趋势不一致。
4. Stage 3 初始时发现 ETTh1 数据缺失，所以先停止训练，补齐官方 ETTh1.csv，并完成 DataReader 的 train/val/test 校验。
5. ETTh1 H=96 的三随机种子实验也完成了非 PCA 组合；结果中 baseline 仍然最好，dropoutLinearAE 是非 baseline 中最接近的。
6. PCA 在两个阶段都失败，原因很可能是当前 `ft_then_supervised` 路径要求 adapter 是 PyTorch Module，而 PCA projector 不满足。
7. 目前结果说明代码流程已经打通，但 adapter 优势还没有复现出来，下一步应优先排查 PCA 路径、训练轮数和 adapter 超参数。

## 下一步计划

{markdown_table(next_steps)}

## 附：缺失文件检查

{markdown_table(missing)}
"""
    return report


def save_tables(
    tables: dict[str, pd.DataFrame],
    out_dir: Path,
) -> str:
    try:
        import openpyxl  # noqa: F401

        xlsx = out_dir / "adapts_progress_tables.xlsx"
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            for sheet, df in tables.items():
                df.to_excel(writer, sheet_name=sheet[:31], index=False)
        return rel(xlsx)
    except Exception:
        file_map = {
            "Stage Overview": "stage_overview.csv",
            "Environment": "environment.csv",
            "Data Status": "data_status.csv",
            "Experiment Settings": "experiment_settings.csv",
            "Illness Results": "illness_results.csv",
            "ETTh1 Results": "etth1_results.csv",
            "Best Summary": "best_summary.csv",
            "Issues": "issues.csv",
            "Next Steps": "next_steps.csv",
        }
        for sheet, filename in file_map.items():
            tables[sheet].to_csv(out_dir / filename, index=False)
        return rel(out_dir)


def make_plot(best_summary: pd.DataFrame, out_dir: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt

        rows = []
        for _, row in best_summary.iterrows():
            if pd.isna(row["baseline_mse"]) or pd.isna(row["best_mse"]):
                continue
            rows.append(row)
        if not rows:
            return None

        labels = [row["stage"] for row in rows]
        baseline = [float(row["baseline_mse"]) for row in rows]
        best = [float(row["best_mse"]) for row in rows]
        x = range(len(rows))
        width = 0.35
        plt.figure(figsize=(8, 4.5))
        plt.bar([i - width / 2 for i in x], baseline, width=width, label="Baseline MSE")
        plt.bar([i + width / 2 for i in x], best, width=width, label="Best MSE")
        plt.xticks(list(x), labels)
        plt.ylabel("MSE")
        plt.title("AdaPTS Reproduction: Baseline vs Best MSE")
        plt.legend()
        plt.tight_layout()
        output = out_dir / "mse_comparison.png"
        plt.savefig(output, dpi=160)
        plt.close()
        return rel(output)
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")
        return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    missing_rows = []
    all_wide = []
    for key, path in EXPECTED_CSVS.items():
        df = read_csv_if_exists(path)
        missing_rows.append(
            {
                "file": rel(path),
                "exists": path.exists(),
                "note": "ok" if path.exists() else "文件不存在，说明该阶段尚未生成该结果。",
            }
        )
        if df is None:
            continue
        if key.endswith("raw") or key == "minimal_summary":
            stage = {
                "minimal_raw": "Minimal Illness H=24",
                "minimal_summary": "Minimal Illness H=24",
                "stage2_raw": "Illness H=24",
                "stage3_raw": "ETTh1 H=96",
            }.get(key)
            if stage:
                all_wide.append(wide_results(df, stage))

    for report in EXPECTED_REPORTS:
        path = ROOT / report
        missing_rows.append(
            {
                "file": report,
                "exists": path.exists(),
                "note": "ok" if path.exists() else "报告未生成/未完成",
            }
        )

    all_available = pd.concat(all_wide, ignore_index=True) if all_wide else pd.DataFrame()
    all_available.to_csv(OUT_DIR / "all_available_results.csv", index=False)

    stage2_summary = load_stage_summary("Illness H=24", EXPECTED_CSVS["stage2_summary"])
    stage3_summary = load_stage_summary("ETTh1 H=96", EXPECTED_CSVS["stage3_summary"])
    summaries = [df for df in [stage2_summary, stage3_summary] if df is not None]
    stage_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    stage_summary.to_csv(OUT_DIR / "stage_summary_table.csv", index=False)

    best_summary = pd.DataFrame(
        [
            best_summary_row("Illness H=24", "Illness", 24, stage2_summary),
            best_summary_row("ETTh1 H=96", "ETTh1", 96, stage3_summary),
        ]
    )
    best_summary.to_csv(OUT_DIR / "best_adapter_summary.csv", index=False)

    missing = pd.DataFrame(missing_rows)
    missing.to_csv(OUT_DIR / "missing_files_report.csv", index=False)

    stage_overview = stage_overview_rows()
    env = environment_rows()
    data_status = data_status_rows()
    settings = experiment_settings_rows()
    issues = issues_rows()
    next_steps = next_steps_rows()

    tables = {
        "Stage Overview": stage_overview,
        "Environment": env,
        "Data Status": data_status,
        "Experiment Settings": settings,
        "Illness Results": format_summary_for_report(stage2_summary),
        "ETTh1 Results": format_summary_for_report(stage3_summary),
        "Best Summary": best_summary,
        "Issues": issues,
        "Next Steps": next_steps,
    }
    table_output = save_tables(tables, OUT_DIR)

    plot_output = make_plot(best_summary, OUT_DIR)

    report = build_report(
        stage_overview=stage_overview,
        env=env,
        data_status=data_status,
        settings=settings,
        stage2_summary=stage2_summary,
        stage3_summary=stage3_summary,
        best_summary=best_summary,
        issues=issues,
        next_steps=next_steps,
        missing=missing,
    )
    REPORT.write_text(report, encoding="utf-8")

    print(f"Saved: {rel(OUT_DIR / 'all_available_results.csv')}")
    print(f"Saved: {rel(OUT_DIR / 'stage_summary_table.csv')}")
    print(f"Saved: {rel(OUT_DIR / 'best_adapter_summary.csv')}")
    print(f"Saved: {rel(OUT_DIR / 'missing_files_report.csv')}")
    print(f"Saved table output: {table_output}")
    if plot_output:
        print(f"Saved plot: {plot_output}")
    print(f"Saved report: {rel(REPORT)}")
    print("\nBest summary:")
    print(best_summary.to_string(index=False))


if __name__ == "__main__":
    main()
