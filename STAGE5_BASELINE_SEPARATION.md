# Stage 5：Baseline 分离实验

## 1. 背景

前期 Illness H=24 和 ETTh1 H=96 三随机种子实验中，baseline 都表现最好。随后 `BASELINE_PROTOCOL_AUDIT.md` 确认，前期所谓 baseline 并不是 pure zero-shot MOMENT；在 `adapter=None + --use_revin + supervised=ft_then_supervised` 下，代码会创建 `JustRevIn`，使用 ADAPTS wrapper/scaler，fine-tune MOMENT forecasting head，并进入 `adapter_supervised_fine_tuning` 训练 RevIN affine 参数。因此需要把 baseline 定义拆开，避免把 strong no-named-adapter baseline 误称为纯 MOMENT baseline。

## 2. 本阶段目的

- 区分 pure MOMENT、head-only baseline、strong RevIN baseline 和 named adapter；
- 避免把前期 Stage 2 / Stage 3 baseline 误解释为 pure MOMENT；
- 为后续公平比较 adapter 提供更清楚的协议基线。

## 3. Baseline 定义

| Type | Name | Uses ADAPTS wrapper | Uses RevIN | Fine-tunes head | Trains adapter/RevIN | Description |
|---|---|---|---|---|---|---|
| B0 | pure MOMENT zero-shot | No | No | No | No | 直接加载 MOMENT-small，逐通道预测；使用 DataReader 读入的标准化 forecasting 数据，但不进入 ADAPTS wrapper |
| B1 | head-only fine-tuning | Yes | No | Yes | No | `scripts/run.py --supervised ft --no-use-revin`；fine-tune forecasting head，随后只执行 `fit_adapter` |
| B2 | current strong RevIN baseline | Yes | Yes | Yes | Yes | 前期 baseline 的真实协议：`adapter=None + use_revin=True + ft_then_supervised`，训练 JustRevIn/RevIN affine 参数 |
| B3 | dropoutLinearAE adapter | Yes | Yes | Yes | Yes | named adapter 对照：`adapter=dropoutLinearAE + use_revin=True + ft_then_supervised` |

## 4. 代码路径审查

`scripts/run.py --help` 使用 tyro 短横线参数形式，例如 `--forecast-horizon`、`--model-name`、`--use-revin/--no-use-revin`。本地正确环境为 `.venv-adapts`，其中 `torch 2.12.0+cu126` 可用，GPU 为 NVIDIA GeForce RTX 3060。

CLI 支持情况：

| 检查项 | 结论 |
|---|---|
| `--supervised ft` | 支持，可用于 B1 head-only fine-tuning |
| `--supervised ft_then_supervised` | 支持，用于 B2/B3 |
| 关闭 RevIN | 支持，使用 `--no-use-revin` |
| 直接指定 `n_components=7` | 不支持 `--n-components` 参数；本实验通过 full-channel 设置得到有效 `n_components=7` |
| B0 pure MOMENT | `scripts/run.py` 不能表达，因为它总会创建 ADAPTS wrapper；因此新增独立脚本 |
| B1 head-only | 可以通过 `scripts/run.py --supervised ft --no-use-revin` 表达；不会进入 `adapter_supervised_fine_tuning` |
| B2 strong baseline | 可以通过 `adapter=None + --use-revin + --supervised ft_then_supervised` 表达 |
| B3 dropoutLinearAE | 可以通过 `--adapter dropoutLinearAE + --use-revin + --supervised ft_then_supervised` 表达 |

本阶段新增脚本：

- `scripts/evaluate_pure_moment_baseline.py`
- `scripts/run_stage5_baseline_separation.py`
- `scripts/summarize_stage5_baseline_separation.py`

未修改核心模型、adapter 或 DataReader 算法逻辑。

## 5. 实验设置

| Item | Value |
|---|---|
| Dataset | ETTh1 |
| Horizon | 96 |
| Context length | 512 |
| Seed | 13 |
| Model | AutonLab/MOMENT-1-small |
| Effective n_components | 7 |
| Device | cuda:0 |
| Fast epoch setting | `n_epochs_fine_tuning=5`, `n_epochs_adapter=30` |
| Full epoch setting | 未运行；需显式 `--full_epochs` |
| Illness extension | 本阶段未运行；脚本支持 `--dataset Illness` |

运行命令：

```bash
python scripts/run_stage5_baseline_separation.py --dry_run
python scripts/run_stage5_baseline_separation.py --dataset ETTh1 --skip_existing
python scripts/summarize_stage5_baseline_separation.py
```

## 6. 结果

结果文件：

- Raw results: `results/stage5_baseline_separation/raw_results.csv`
- Summary: `results/stage5_baseline_separation/summary_baseline_types.csv`
- Failed runs: `results/stage5_baseline_separation/failed_runs.json`
- Logs: `logs/stage5_baseline_separation/`

失败记录为空：`[]`。

| Dataset | Horizon | Baseline type | Description | MSE | MAE | Improvement vs B0 (%) | Gap vs B2 (%) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 96 | B0_pure_moment | Pure MOMENT zero-shot | 0.754653 | 0.592376 | 0.000000 | 93.876943 | ok |
| ETTh1 | 96 | B1_head_only | MOMENT head-only fine-tuning | 0.388687 | 0.409648 | 48.494675 | -0.143051 | ok |
| ETTh1 | 96 | B2_strong_revin | Strong ADAPTS/RevIN no-named-adapter baseline | 0.389243 | 0.410324 | 48.420891 | 0.000000 | ok |
| ETTh1 | 96 | B3_dropoutLinearAE | dropoutLinearAE named adapter | 0.398486 | 0.419226 | 47.196132 | 2.374526 | ok |

Runtime 摘要：

| Type | Runtime seconds |
|---|---:|
| B0_pure_moment | 15.97 |
| B1_head_only | 471.91 |
| B2_strong_revin | 2249.33 |
| B3_dropoutLinearAE | 3804.29 |

## 7. 关键发现

1. **B2 strong baseline 没有显著强于 B0，而是 B1/B2 都显著强于 B0。**  
   B0 pure MOMENT zero-shot MSE 为 0.754653；B1 head-only fine-tuning MSE 为 0.388687；B2 strong RevIN baseline MSE 为 0.389243。核心提升主要来自 MOMENT forecasting head fine-tuning，而不是 RevIN adapter stage。

2. **前期所谓 baseline 应重新命名为 strong no-named-adapter baseline。**  
   它不是 pure MOMENT；它包含 ADAPTS wrapper、DataReader/ADAPTS scaling、head fine-tuning，并在 `use_revin=True` 时训练 JustRevIn/RevIN。

3. **dropoutLinearAE 明显优于 pure MOMENT，但没有超过 head-only 或 strong RevIN baseline。**  
   B3 相比 B0 的 MSE 改善约 47.20%，说明 named adapter 并非完全无效；但 B3 MSE 0.398486，仍差于 B1 的 0.388687 和 B2 的 0.389243。

4. **B1 head-only 当前最强。**  
   在 ETTh1 H=96 seed=13 快速档中，B1 比 B2 略好 0.143% 左右；这个差距很小，但说明 RevIN supervised stage 并没有带来额外收益。

5. **B0 已成功实现，但仍使用 DataReader 的 forecasting 标准化数据。**  
   B0 不使用 ADAPTS wrapper/scaler、不使用 RevIN、不训练模型；不过 DataReader 本身会按 forecasting split 使用训练集 StandardScaler。这一点需要在后续报告中明确。

## 8. 对前期结果的重新解释

Stage 2 / Stage 3 中的 baseline 不应再称为 pure MOMENT baseline。更准确的命名是：

> strong no-named-adapter baseline / ADAPTS-RevIN baseline

不过 Stage 5 显示，在 ETTh1 H=96 seed=13 快速档中，真正驱动强 baseline 的主要因素可能是 **forecasting head fine-tuning**。B1 head-only 已经达到 0.388687，略优于 B2 strong RevIN 的 0.389243。因此前期 adapter 输给 baseline，不应简单解释为 adapter 无效，而应解释为：

- pure MOMENT zero-shot 很弱；
- fine-tuned MOMENT head 已经非常强；
- 当前 adapter 快速档没有超过这个强 head-only baseline；
- RevIN/JustRevIn training 并未在本次 ETTh1 seed=13 中带来额外收益；
- dropoutLinearAE 超过 pure MOMENT，但没有超过 fine-tuned head baseline。

## 9. 下一步建议

1. 后续所有汇报必须区分：
   - B0 pure MOMENT zero-shot；
   - B1 MOMENT head-only fine-tuning；
   - B2 strong ADAPTS/RevIN no-named-adapter baseline；
   - B3 named adapter。
2. 如果要判断 adapter 是否有效，优先比较 B3 vs B0 和 B3 vs B1，而不是只比较 B3 vs B2。
3. 下一步建议在 ETTh1 H=96 seed=13 上跑 full epoch 对齐档：
   - B1 head-only；
   - B3 dropoutLinearAE；
   - 可选 B2 strong RevIN。
4. 如果 full epoch 后 B3 仍不超过 B1，则重点排查 adapter 超参数、训练目标、RevIN/scaler 交互。
5. 再扩展 Illness H=24 seed=13 的同样 baseline 分离，以确认这个现象是否跨数据集稳定。
6. PCA 仍需单独排查，因为它不能直接进入 `ft_then_supervised` 的 PyTorch adapter training path。
