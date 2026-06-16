# AdaPTS 阶段二复现：Illness H=24 三随机种子实验

## 1. 阶段目标

本阶段从单次最小复现扩展到 Illness H=24 的 3-seed 小规模可信复现。实验只覆盖 Illness、forecast horizon 24、MOMENT-small，并对 baseline、PCA、linearAE、dropoutLinearAE、linearVAE、VAE 进行小规模比较。

## 2. 实验设置

- Dataset: Illness
- Horizon: 24
- Context length: 512
- Model: AutonLab/MOMENT-1-small
- Supervised mode: ft_then_supervised
- RevIN: True
- Seeds: 13, 42, 2024
- Adapters: baseline, PCA, linearAE, dropoutLinearAE, linearVAE, VAE
- Device: cuda:0
- GPU: NVIDIA GeForce RTX 3060
- Epochs: n_epochs_fine_tuning = 5, n_epochs_adapter = 30
- Python: 3.10.11
- torch: 2.12.0+cu126
- numpy: 1.25.2
- pandas: 2.3.3
- OS: Windows-10-10.0.26200-SP0

## 3. 代码状态

- git commit: 8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278
- git status:

```text
 M .gitignore
 M src/adapts/icl/moment.py
 M src/adapts/utils/main_script.py
?? ENV_SETUP.md
?? MINIMAL_REPRO.md
?? scripts/run_stage2_illness_h24_3seeds.py
?? scripts/summarize_minimal_repro.py
?? scripts/summarize_stage2_illness_h24.py
```

- 本阶段新增文件:
  - scripts/run_stage2_illness_h24_3seeds.py
  - scripts/summarize_stage2_illness_h24.py
  - results/stage2_illness_h24/raw_results.csv
  - results/stage2_illness_h24/summary_mean_std_stderr.csv
  - results/stage2_illness_h24/summary_mean_std_stderr.md
  - results/stage2_illness_h24/failed_runs.json
  - STAGE2_ILLNESS_3SEED.md
- 本阶段是否修改核心代码: 否。本阶段只新增批量运行和汇总脚本；已有的路径和 MOMENT 首次加载修正来自环境搭建阶段。

## 4. 运行命令

使用批量脚本统一运行 3 seeds 与 6 个 adapter 组合。脚本会自动判断 GPU、设置 epoch，并为每个 seed/adapter 使用独立日志目录。

```bash
python scripts/run_stage2_illness_h24_3seeds.py --dry_run
python scripts/run_stage2_illness_h24_3seeds.py
python scripts/summarize_stage2_illness_h24.py
```

单个正式命令示例：

```bash
python scripts/run.py --forecast-horizon 24 --model-name AutonLab/MOMENT-1-small --context-length 512 --seed 13 --device cuda:0 --dataset-name Illness --use-revin --supervised ft_then_supervised --n-epochs-fine-tuning 5 --n-epochs-adapter 30 --data-path results/stage2_illness_h24/raw_results.csv --log-dir logs/stage2_illness_h24/seed_13/linearVAE --adapter linearVAE
```

baseline 运行时不传入 `--adapter` 参数。PCA 在 CLI 中使用项目源码支持的 lowercase 名称 `pca`。

## 5. 原始结果

- 原始结果 CSV: results/stage2_illness_h24/raw_results.csv
- 原始结果 shape: 90 rows x 15 columns

前几行：

```text
dataset      foundational_model adapter  n_features  n_components      is_fine_tuned  pca_in_preprocessing  use_revin  context_length  forecast_horizon  running_time  seed     metric    value  train_size
Illness AutonLab/MOMENT-1-small     NaN           7             7 ft_then_supervised                 False       True             512                24     90.979667    13        mse 2.598519         141
Illness AutonLab/MOMENT-1-small     NaN           7             7 ft_then_supervised                 False       True             512                24     90.979667    13        mae 1.108536         141
Illness AutonLab/MOMENT-1-small     NaN           7             7 ft_then_supervised                 False       True             512                24     90.979667    13 scaled_mse 2.048735         141
Illness AutonLab/MOMENT-1-small     NaN           7             7 ft_then_supervised                 False       True             512                24     90.979667    13 scaled_mae 1.030718         141
Illness AutonLab/MOMENT-1-small     NaN           7             7 ft_then_supervised                 False       True             512                24     90.979667    13         ks 0.720098         141
```

成功运行的 per-seed MSE/MAE：

| adapter | seed | mse | mae |
| --- | ---: | ---: | ---: |
| baseline | 13 | 2.598519 | 1.108536 |
| baseline | 42 | 2.632884 | 1.116006 |
| baseline | 2024 | 2.621596 | 1.112506 |
| linearAE | 13 | 2.989221 | 1.213902 |
| linearAE | 42 | 3.041723 | 1.168894 |
| linearAE | 2024 | 2.906947 | 1.178727 |
| dropoutLinearAE | 13 | 2.935777 | 1.204068 |
| dropoutLinearAE | 42 | 2.967546 | 1.163957 |
| dropoutLinearAE | 2024 | 2.782845 | 1.149646 |
| linearVAE | 13 | 2.891400 | 1.141204 |
| linearVAE | 42 | 2.621444 | 1.115851 |
| linearVAE | 2024 | 2.865908 | 1.142641 |
| VAE | 13 | 2.881393 | 1.102431 |
| VAE | 42 | 2.813911 | 1.090144 |
| VAE | 2024 | 2.850982 | 1.097568 |

## 6. 聚合结果

- 汇总结果 CSV: results/stage2_illness_h24/summary_mean_std_stderr.csv
- 汇总结果 Markdown: results/stage2_illness_h24/summary_mean_std_stderr.md

| adapter         | mse_mean | mse_std  | mse_stderr | mae_mean | mae_std  | mae_stderr | n |
| --------------- | -------- | -------- | ---------- | -------- | -------- | ---------- | - |
| baseline        | 2.617666 | 0.017516 | 0.010113   | 1.112349 | 0.003738 | 0.002158   | 3 |
| linearAE        | 2.979297 | 0.067934 | 0.039222   | 1.187175 | 0.023663 | 0.013662   | 3 |
| dropoutLinearAE | 2.895389 | 0.098752 | 0.057015   | 1.172557 | 0.028212 | 0.016288   | 3 |
| linearVAE       | 2.792917 | 0.149046 | 0.086052   | 1.133232 | 0.015069 | 0.008700   | 3 |
| VAE             | 2.848762 | 0.033796 | 0.019512   | 1.096714 | 0.006188 | 0.003572   | 3 |

## 7. 与论文 Table 1 的初步比较

本阶段没有做 hyperopt，也没有使用完整论文训练轮数，因此不要求数值完全一致。重点是验证 Illness H=24 的 3-seed 流程、结果保存、日志追踪和统计汇总。

论文参考值：

| Adapter | Paper MSE on Illness H=24 |
|---|---:|
| baseline Moment | 2.902 ± 0.023 |
| PCA | 2.980 ± 0.001 |
| LinearAE | 2.624 ± 0.035 |
| dropoutLinearAE | 2.760 ± 0.061 |
| LinearVAE | 2.542 ± 0.036 |
| VAE | 2.461 ± 0.008 |

本地小规模结果中，baseline 的 MSE mean 最低，VAE 的 MAE mean 最低。VAE/linearVAE 在 MSE 上没有超过 baseline，这与论文趋势不一致。可能原因包括：本阶段训练轮数较少、未做 hyperopt、adapter 训练设置未完全对齐论文、MOMENT/torch 依赖版本和随机性差异，以及 PCA 在当前 supervised 路径下无法完成。

## 8. 失败运行记录

- failed_runs.json: results/stage2_illness_h24/failed_runs.json
- 失败组合: PCA seed 13、42、2024
- 失败原因: 使用 `--adapter pca` 时，项目会进入 `adapter_supervised_fine_tuning`；PCA 的 base projector 不是 `torch.nn.Module`，触发断言。

捕获到的关键异常：

```text
AssertionError: adapter must be a PyTorch Module
```

为定位失败，额外运行了一次 PCA probe，日志保存在：

```text
results/stage2_illness_h24/pca_failure_probe.log
```

按照本阶段约束，没有为 PCA 强行修改核心逻辑。

## 9. 初步结论

1. 本地复现没有完整跑完全部 18 个组合，因为 PCA 三个 seed 均失败；其余 15 个组合稳定完成。
2. 在当前小规模设置下，adapter 没有在 MSE 上明显优于 baseline。
3. 按 MSE mean，整体最好是 baseline；非 baseline 中最好是 linearVAE，MSE mean = 2.792917。
4. 按 MAE mean，最好是 VAE，MAE mean = 1.096714。
5. 当前结果与论文趋势不完全一致，主要可能来自训练轮数、超参数搜索、adapter supervised 路径和依赖版本差异。

baseline MSE mean = 2.617666，best non-baseline adapter linearVAE MSE mean = 2.792917。相对 baseline 的 MSE 改善幅度为 -6.69%，即当前小规模设置下 linearVAE 比 baseline 更差。

## 10. 下一步建议

建议下一步进入以下之一：

1. Illness H=24 更接近论文训练轮数；
2. 排查 PCA 是否应在非 supervised adapter 训练路径或 preprocessing PCA 路径运行；
3. Illness H=60；
4. ETTh1 H=96；
5. latent dimension 消融；
6. calibration / ECE 分析。
