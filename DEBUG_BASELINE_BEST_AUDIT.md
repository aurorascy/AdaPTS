# Baseline 最优异常复现审计报告

## 1. 异常现象

当前 Illness H=24 和 ETTh1 H=96 的三随机种子结果中，baseline 都是 MSE 最低的配置：

| stage              | dataset | horizon | local_baseline_mse | paper_baseline_mse | paper_baseline_std | absolute_difference_local_minus_paper | relative_difference_percent | local_baseline_stronger_than_paper |
| ------------------ | ------- | ------- | ------------------ | ------------------ | ------------------ | ------------------------------------- | --------------------------- | ---------------------------------- |
| stage2_illness_h24 | Illness | 24      | 2.617666           | 2.902000           | 0.023000           | -0.284334                             | -9.797848                   | True                               |
| stage3_etth1_h96   | ETTh1   | 96      | 0.388832           | 0.411000           | 0.012000           | -0.022168                             | -5.393588                   | True                               |

这与论文中 adapter 尤其 VAE/LinearVAE/dropoutLinearAE 优于 baseline 的趋势不一致。本审计只读取已有 CSV、日志和 config，没有启动新训练，也没有运行 hyperopt。

## 2. 已排除的问题

### 2.1 Summary metric 是否选错

| stage              | raw_columns                                                                                                                                                                                     | metric_types                              | summary_columns                                                          | summary_uses_metric | summary_matches_raw_metric_mse | max_abs_diff | note                                               |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------ | ------------------- | ------------------------------ | ------------ | -------------------------------------------------- |
| stage2_illness_h24 | dataset, foundational_model, adapter, n_features, n_components, is_fine_tuned, pca_in_preprocessing, use_revin, context_length, forecast_horizon, running_time, seed, metric, value, train_size | ece, ks, mae, mse, scaled_mae, scaled_mse | adapter, mse_mean, mse_std, mse_stderr, mae_mean, mae_std, mae_stderr, n | mse                 | True                           | 0.000000     | summary matches recomputation from metric == 'mse' |
| stage3_etth1_h96   | dataset, foundational_model, adapter, n_features, n_components, is_fine_tuned, pca_in_preprocessing, use_revin, context_length, forecast_horizon, running_time, seed, metric, value, train_size | ece, ks, mae, mse, scaled_mae, scaled_mse | adapter, mse_mean, mse_std, mse_stderr, mae_mean, mae_std, mae_stderr, n | mse                 | True                           | 0.000000     | summary matches recomputation from metric == 'mse' |

重新从 raw_results 中筛选 `metric == "mse"` 聚合后，与 `summary_mean_std_stderr.csv` 的 `mse_mean` 一致。因此目前没有发现 summary 使用了 `scaled_mse`、`mae`、`ks` 或 `ece` 的问题。

重算摘要：

| stage              | adapter         | mse_mean_recomputed | mse_mean_summary | abs_diff |
| ------------------ | --------------- | ------------------- | ---------------- | -------- |
| stage2_illness_h24 | VAE             | 2.848762            | 2.848762         | 0.000000 |
| stage2_illness_h24 | baseline        | 2.617666            | 2.617666         | 0.000000 |
| stage2_illness_h24 | dropoutLinearAE | 2.895389            | 2.895389         | 0.000000 |
| stage2_illness_h24 | linearAE        | 2.979297            | 2.979297         | 0.000000 |
| stage2_illness_h24 | linearVAE       | 2.792917            | 2.792917         | 0.000000 |
| stage3_etth1_h96   | VAE             | 0.405431            | 0.405431         | 0.000000 |
| stage3_etth1_h96   | baseline        | 0.388832            | 0.388832         | 0.000000 |
| stage3_etth1_h96   | dropoutLinearAE | 0.397490            | 0.397490         | 0.000000 |
| stage3_etth1_h96   | linearAE        | 0.397805            | 0.397805         | 0.000000 |
| stage3_etth1_h96   | linearVAE       | 0.408220            | 0.408220         | 0.000000 |

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

| stage              | adapter         | seed | n_components_from_raw | n_features | n_components | custom_n_comp | number_n_comp_to_try |
| ------------------ | --------------- | ---- | --------------------- | ---------- | ------------ | ------------- | -------------------- |
| stage2_illness_h24 | PCA             | 13   |                       |            | 7            | False         | 4                    |
| stage2_illness_h24 | PCA             | 42   |                       |            | 7            | False         | 4                    |
| stage2_illness_h24 | PCA             | 2024 |                       |            | 7            | False         | 4                    |
| stage2_illness_h24 | VAE             | 13   | [7]                   | [7]        | 7            | False         | 4                    |
| stage2_illness_h24 | VAE             | 42   | [7]                   | [7]        | 7            | False         | 4                    |
| stage2_illness_h24 | VAE             | 2024 | [7]                   | [7]        | 7            | False         | 4                    |
| stage2_illness_h24 | baseline        | 13   | [7]                   | [7]        | 7            | False         | 4                    |
| stage2_illness_h24 | baseline        | 42   | [7]                   | [7]        | 7            | False         | 4                    |
| stage2_illness_h24 | baseline        | 2024 | [7]                   | [7]        | 7            | False         | 4                    |
| stage2_illness_h24 | dropoutLinearAE | 13   | [7]                   | [7]        | 7            | False         | 4                    |
| stage2_illness_h24 | dropoutLinearAE | 42   | [7]                   | [7]        | 7            | False         | 4                    |
| stage2_illness_h24 | dropoutLinearAE | 2024 | [7]                   | [7]        | 7            | False         | 4                    |
| stage2_illness_h24 | linearAE        | 13   | [7]                   | [7]        | 7            | False         | 4                    |
| stage2_illness_h24 | linearAE        | 42   | [7]                   | [7]        | 7            | False         | 4                    |
| stage2_illness_h24 | linearAE        | 2024 | [7]                   | [7]        | 7            | False         | 4                    |
| stage2_illness_h24 | linearVAE       | 13   | [7]                   | [7]        | 7            | False         | 4                    |
| stage2_illness_h24 | linearVAE       | 42   | [7]                   | [7]        | 7            | False         | 4                    |
| stage2_illness_h24 | linearVAE       | 2024 | [7]                   | [7]        | 7            | False         | 4                    |
| stage3_etth1_h96   | PCA             | 13   |                       |            | 7            | False         | 4                    |
| stage3_etth1_h96   | PCA             | 42   |                       |            | 7            | False         | 4                    |
| stage3_etth1_h96   | PCA             | 2024 |                       |            | 7            | False         | 4                    |
| stage3_etth1_h96   | VAE             | 13   | [7]                   | [7]        | 7            | False         | 4                    |
| stage3_etth1_h96   | VAE             | 42   | [7]                   | [7]        | 7            | False         | 4                    |
| stage3_etth1_h96   | VAE             | 2024 | [7]                   | [7]        | 7            | False         | 4                    |
| stage3_etth1_h96   | baseline        | 13   | [7]                   | [7]        | 7            | False         | 4                    |
| stage3_etth1_h96   | baseline        | 42   | [7]                   | [7]        | 7            | False         | 4                    |
| stage3_etth1_h96   | baseline        | 2024 | [7]                   | [7]        | 7            | False         | 4                    |
| stage3_etth1_h96   | dropoutLinearAE | 13   | [7]                   | [7]        | 7            | False         | 4                    |
| stage3_etth1_h96   | dropoutLinearAE | 42   | [7]                   | [7]        | 7            | False         | 4                    |
| stage3_etth1_h96   | dropoutLinearAE | 2024 | [7]                   | [7]        | 7            | False         | 4                    |
| stage3_etth1_h96   | linearAE        | 13   | [7]                   | [7]        | 7            | False         | 4                    |
| stage3_etth1_h96   | linearAE        | 42   | [7]                   | [7]        | 7            | False         | 4                    |
| stage3_etth1_h96   | linearAE        | 2024 | [7]                   | [7]        | 7            | False         | 4                    |
| stage3_etth1_h96   | linearVAE       | 13   | [7]                   | [7]        | 7            | False         | 4                    |
| stage3_etth1_h96   | linearVAE       | 42   | [7]                   | [7]        | 7            | False         | 4                    |
| stage3_etth1_h96   | linearVAE       | 2024 | [7]                   | [7]        | 7            | False         | 4                    |

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

| stage              | dataset | horizon | local_baseline_mse | paper_baseline_mse | paper_baseline_std | absolute_difference_local_minus_paper | relative_difference_percent | local_baseline_stronger_than_paper |
| ------------------ | ------- | ------- | ------------------ | ------------------ | ------------------ | ------------------------------------- | --------------------------- | ---------------------------------- |
| stage2_illness_h24 | Illness | 24      | 2.617666           | 2.902000           | 0.023000           | -0.284334                             | -9.797848                   | True                               |
| stage3_etth1_h96   | ETTh1   | 96      | 0.388832           | 0.411000           | 0.012000           | -0.022168                             | -5.393588                   | True                               |

可能原因包括 MOMENT 版本差异、训练轮数与 early stopping、数据版本和 split、RevIN/scaler 行为、metric 是否使用原尺度 MSE、以及本地环境依赖版本差异。由于 summary metric 已确认使用 raw `mse`，metric 选错不是主要解释。

## 7. 最可能原因排序

| rank | cause                                                | evidence                                                                       |
| ---- | ---------------------------------------------------- | ------------------------------------------------------------------------------ |
| 1    | 训练轮数明显少于论文默认/完整设置                                    | 当前 ft=5、adapter=30；论文/脚本默认 ft=50、adapter=300，并使用 hyperparameter optimization。  |
| 2    | 未做 hyperparameter optimization，且 custom_n_comp=False | 所有已完成组合实际 n_components=7；没有系统搜索 adapter 超参数或 latent dimension。                 |
| 3    | 本地 baseline 明显强于论文 baseline                          | Illness baseline 2.618 vs paper 2.902；ETTh1 baseline 0.389 vs paper 0.411。     |
| 4    | PCA 路径与 supervised fine-tuning 协议不匹配                 | PCA 在 Stage 2/3 均失败；probe 捕获 AssertionError: adapter must be a PyTorch Module。 |

## 8. 是否存在结果汇总错误

未发现。`summary_mean_std_stderr.csv` 与从 raw_results 中 `metric == "mse"` 重算的结果一致。

## 9. 是否存在 adapter 未充分训练

存在较大可能。非 PCA adapter 确实进入了训练，并且 TensorBoard loss 通常下降；但当前 epoch 设置为 ft=5、adapter=30，远低于脚本默认和论文更接近的 ft=50、adapter=300。也没有运行 hyperopt，因此 adapter 可能训练不足或超参数不佳。

## 10. 是否存在 baseline 过强

是。Illness 和 ETTh1 的本地 baseline 都优于论文 baseline，这会压缩 adapter 的相对提升空间。该现象需要优先复核 MOMENT 版本、数据 split、RevIN/scaler、head fine-tuning 策略和训练轮数。

## 11. 下一步最小重跑实验计划

不在本次 audit 中执行。命令已写入：

`results/debug_rerun_plan/etth1_h96_seed13_baseline_vs_dropoutLinearAE_commands.sh`

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
