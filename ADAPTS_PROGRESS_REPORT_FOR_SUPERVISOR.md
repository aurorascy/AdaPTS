# AdaPTS 论文复现阶段性汇报

## 研究背景简述

AdaPTS 关注的是如何将以单变量预测为主的时间序列基础模型适配到多变量概率预测场景。核心思路是在 foundation model 前后加入 adapter：先把多变量输入映射到 latent space，再交给冻结或轻量微调的基础模型预测，最后由 adapter 映射回原始变量空间。本阶段复现没有直接追求完整 Table 1，而是按“环境搭建、最小实验、三随机种子、跨数据集验证”的顺序，先确认代码路径、数据路径、模型加载和代表性实验是否可稳定运行。

## 已完成工作总览

| 阶段                           | 目标                                           | 状态   | 产出文件                                                  | 说明                                                         |
| ---------------------------- | -------------------------------------------- | ---- | ----------------------------------------------------- | ---------------------------------------------------------- |
| 环境搭建                         | 安装依赖、修正路径、MOMENT 可加载、单元测试/导入检查               | 完成   | ENV_SETUP.md                                          | 使用 .venv-adapts，CUDA 可用，完成核心依赖安装与 smoke check              |
| 最小复现 Illness H=24            | 跑通 baseline、linearVAE、VAE 的单 seed 最小流程       | 完成   | MINIMAL_REPRO.md; results/minimal_repro/*             | 确认 scripts/run.py、MOMENT-small、adapter、metrics 保存流程可用      |
| Stage 2 Illness H=24 3 seeds | Illness H=24 上 baseline 与多个 adapter 的三随机种子对比 | 部分完成 | STAGE2_ILLNESS_3SEED.md; results/stage2_illness_h24/* | 非 PCA 组合完成；PCA 在 supervised 路径失败                           |
| Stage 3 ETTh1 初始尝试           | 扩展到 ETTh1 H=96                               | 已处理  | STAGE3_ETTH1_H96_3SEED.md                             | 最初因 ETTh1.csv 缺失停止，未伪造数据                                   |
| Stage 3.0 ETTh1 数据准备         | 下载官方 ETTh1.csv 并做 DataReader 校验              | 完成   | DATA_PREP_ETTH1.md; scripts/download_etth1.py         | ETTh1 shape=(17420,8)，缺失值 0，DataReader train/val/test 校验通过 |
| Stage 3 ETTh1 H=96 3 seeds   | ETTh1 H=96 上 baseline 与多个 adapter 的三随机种子对比   | 部分完成 | STAGE3_ETTH1_H96_3SEED.md; results/stage3_etth1_h96/* | 非 PCA 组合完成；PCA 在 supervised 路径失败                           |

## 环境与代码修改情况

| 项目             | 内容                                                           |
| -------------- | ------------------------------------------------------------ |
| OS             | Windows-10-10.0.26200-SP0                                    |
| Python         | 3.10.11                                                      |
| 环境隔离           | .venv-adapts；未使用 conda                                       |
| torch          | 2.12.0+cu126                                                 |
| CUDA available | True                                                         |
| GPU            | NVIDIA GeForce RTX 3060                                      |
| numpy          | 1.25.2                                                       |
| pandas         | 2.3.3                                                        |
| 数据路径修正         | src/adapts/utils/main_script.py 改为项目根目录 external_data        |
| MOMENT 加载修正    | src/adapts/icl/moment.py 将 local_files_only 改为 False 以支持首次下载 |
| 核心算法           | 未重构；新增脚本主要用于下载、批量运行、汇总和报告                                    |

## 数据准备情况

| 数据集     | 是否存在 | 路径                                            | shape      | 特征列                                                                                       | 缺失值 | DataReader 校验                                                     |
| ------- | ---- | --------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------- | --- | ----------------------------------------------------------------- |
| Illness | 是    | external_data/forecasting/Illness/Illness.csv | (966, 8)   | date, % WEIGHTED ILI, %UNWEIGHTED ILI, AGE 0-4, AGE 5-24, ILITOTAL, NUM. OF PROVIDERS, OT | 0   | 最小复现和 Stage 2 已成功读取                                               |
| ETTh1   | 是    | external_data/forecasting/ETTh1/ETTh1.csv     | (17420, 8) | date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT                                              | 0   | train (8033,7,512)/(8033,7,96); val/test (2785,7,512)/(2785,7,96) |

## 实验设置汇总

| 实验            | Dataset | Horizon | Context length | Model                   | Seeds        | Adapters                                                 | Epochs           | 状态              |
| ------------- | ------- | ------- | -------------- | ----------------------- | ------------ | -------------------------------------------------------- | ---------------- | --------------- |
| Minimal repro | Illness | 24      | 512            | AutonLab/MOMENT-1-small | 13           | baseline, linearVAE, VAE                                 | ft=5, adapter=20 | 完成              |
| Stage 2       | Illness | 24      | 512            | AutonLab/MOMENT-1-small | 13, 42, 2024 | baseline, PCA, linearAE, dropoutLinearAE, linearVAE, VAE | ft=5, adapter=30 | 非 PCA 完成，PCA 失败 |
| Stage 3       | ETTh1   | 96      | 512            | AutonLab/MOMENT-1-small | 13, 42, 2024 | baseline, PCA, linearAE, dropoutLinearAE, linearVAE, VAE | ft=5, adapter=30 | 非 PCA 完成，PCA 失败 |

## 结果表 1：Illness H=24 三随机种子结果

| Adapter         | MSE mean | MSE std  | MSE stderr | MAE mean | MAE std  | MAE stderr | n |
| --------------- | -------- | -------- | ---------- | -------- | -------- | ---------- | - |
| baseline        | 2.617666 | 0.017516 | 0.010113   | 1.112349 | 0.003738 | 0.002158   | 3 |
| linearAE        | 2.979297 | 0.067934 | 0.039222   | 1.187175 | 0.023663 | 0.013662   | 3 |
| dropoutLinearAE | 2.895389 | 0.098752 | 0.057015   | 1.172557 | 0.028212 | 0.016288   | 3 |
| linearVAE       | 2.792917 | 0.149046 | 0.086052   | 1.133232 | 0.015069 | 0.008700   | 3 |
| VAE             | 2.848762 | 0.033796 | 0.019512   | 1.096714 | 0.006188 | 0.003572   | 3 |

Illness H=24 中，baseline MSE 为 2.617666，当前最优为 baseline，Best MSE 为 2.617666，相对改善为 0.00%。这说明在当前小规模训练设置下，adapter 并没有优于 baseline；非 baseline 中表现较好的是 linearVAE，但仍弱于 baseline。该趋势与论文中 VAE/LinearVAE 更优的结果不一致，可能原因包括训练轮数较少、没有 hyperopt、adapter 超参数未调、随机性以及依赖版本差异。

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

| Adapter         | MSE mean | MSE std  | MSE stderr | MAE mean | MAE std  | MAE stderr | n |
| --------------- | -------- | -------- | ---------- | -------- | -------- | ---------- | - |
| baseline        | 0.388832 | 0.000411 | 0.000237   | 0.409927 | 0.000360 | 0.000208   | 3 |
| linearAE        | 0.397805 | 0.005147 | 0.002971   | 0.418902 | 0.002709 | 0.001564   | 3 |
| dropoutLinearAE | 0.397490 | 0.002565 | 0.001481   | 0.418542 | 0.001915 | 0.001106   | 3 |
| linearVAE       | 0.408220 | 0.002350 | 0.001357   | 0.423988 | 0.002352 | 0.001358   | 3 |
| VAE             | 0.405431 | 0.008243 | 0.004759   | 0.430110 | 0.006786 | 0.003918   | 3 |

ETTh1 H=96 中，baseline MSE 为 0.388832，当前最优为 baseline，Best MSE 为 0.388832，相对改善为 0.00%。本地结果的绝对数值与论文量级接近，但 adapter 没有整体优于 baseline；dropoutLinearAE 是非 baseline 中 MSE 最低的 adapter，部分接近论文中 dropoutLinearAE 较强的趋势。PCA 在 `ft_then_supervised` 路径下失败，暂未纳入数值比较。

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

| 阶段           | Dataset | Horizon | Baseline MSE | Best adapter | Best MSE | Improvement | 是否完成 3 seeds |
| ------------ | ------- | ------- | ------------ | ------------ | -------- | ----------- | ------------ |
| Illness H=24 | Illness | 24      | 2.617666     | baseline     | 2.617666 | 0.00%       | 是            |
| ETTh1 H=96   | ETTh1   | 96      | 0.388832     | baseline     | 0.388832 | 0.00%       | 是（非 PCA）     |

## 问题与处理记录

| 问题                           | 出现阶段              | 处理方式                                                   | 当前状态                                        |
| ---------------------------- | ----------------- | ------------------------------------------------------ | ------------------------------------------- |
| 数据路径硬编码                      | 环境搭建              | 将 prepare_data 中作者本地路径改为 repo_root/external_data       | 已解决                                         |
| MOMENT 首次加载 local_files_only | 环境搭建              | 将 local_files_only 改为 False，允许首次下载 HuggingFace 模型      | 已解决                                         |
| ETTh1 数据缺失                   | Stage 3 初始尝试      | 停止训练；从官方 ETTDataset 下载 ETTh1.csv                       | 已解决                                         |
| ETTh1 DataReader 校验          | Stage 3.0         | 新增 check_etth1_datareader.py，只读验证 train/val/test shape | 通过                                          |
| PCA adapter 失败               | Stage 2 与 Stage 3 | 记录 failed_runs.json；未强改核心逻辑                            | 待确认 PCA 是否应走非 supervised 或 preprocessing 路径 |

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

| 优先级 | 下一步                            | 目的                                              | 预计产出              |
| --- | ------------------------------ | ----------------------------------------------- | ----------------- |
| 1   | 排查 PCA 路径                      | 确认论文中 PCA 是否应使用非 supervised 或 preprocessing PCA | PCA 可复现实验脚本/说明    |
| 2   | 复核 ETTh1 H=96 adapter 超参数和训练轮数 | 解释 adapter 未优于 baseline 的原因                     | 更接近论文设置的 ETTh1 对比 |
| 3   | latent dimension 消融            | 分析 adapter latent 维度对性能的影响                      | 消融表格              |
| 4   | calibration / ECE 分析           | 评估 probabilistic adapter 的不确定性质量                | ECE/KS 与 MSE 对照   |
| 5   | 扩展 Weather 或 ExchangeRate      | 验证跨数据集稳定性                                       | 新增数据集三随机种子结果      |
| 6   | 最后再考虑 hyperopt                 | 接近完整论文设置                                        | 更完整 Table 1 复现    |

## 附：缺失文件检查

| file                                                   | exists | note |
| ------------------------------------------------------ | ------ | ---- |
| results/minimal_repro/illness_h24_minimal.csv          | True   | ok   |
| results/minimal_repro/illness_h24_minimal_summary.csv  | True   | ok   |
| results/stage2_illness_h24/raw_results.csv             | True   | ok   |
| results/stage2_illness_h24/summary_mean_std_stderr.csv | True   | ok   |
| results/stage3_etth1_h96/raw_results.csv               | True   | ok   |
| results/stage3_etth1_h96/summary_mean_std_stderr.csv   | True   | ok   |
| ENV_SETUP.md                                           | True   | ok   |
| MINIMAL_REPRO.md                                       | True   | ok   |
| STAGE2_ILLNESS_3SEED.md                                | True   | ok   |
| DATA_PREP_ETTH1.md                                     | True   | ok   |
| STAGE3_ETTH1_H96_3SEED.md                              | True   | ok   |
