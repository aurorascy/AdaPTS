# AdaPTS 阶段三复现：ETTh1 H=96 三随机种子实验

## 1. 阶段目标

本阶段在 ETTh1 H=96 上复现 baseline 与 5 类 adapter 的 3-seed 对比实验。目标不是完整 Table 1，也没有运行 hyperopt，而是验证 AdaPTS 从 Illness 扩展到 ETTh1 后的训练流程、日志、失败记录和三随机种子统计结果。

## 2. 数据准备状态

- ETTh1.csv 路径: `external_data/forecasting/ETTh1/ETTh1.csv`
- CSV shape: `(17420, 8)`
- columns: `date`, `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`
- 缺失值: `0`
- DataReader train/val/test shape:
  - train X/y: `(8033, 7, 512)` / `(8033, 7, 96)`
  - val X/y: `(2785, 7, 512)` / `(2785, 7, 96)`
  - test X/y: `(2785, 7, 512)` / `(2785, 7, 96)`
- 数据准备报告: `DATA_PREP_ETTH1.md`

## 3. 实验设置

- Dataset: ETTh1
- Horizon: 96
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

## 4. 代码状态

- git commit: `8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278`
- git status:

```text
 M .gitignore
 M src/adapts/icl/moment.py
 M src/adapts/utils/main_script.py
?? DATA_PREP_ETTH1.md
?? ENV_SETUP.md
?? MINIMAL_REPRO.md
?? STAGE2_ILLNESS_3SEED.md
?? STAGE3_ETTH1_H96_3SEED.md
?? scripts/check_etth1_datareader.py
?? scripts/download_etth1.py
?? scripts/run_stage2_illness_h24_3seeds.py
?? scripts/run_stage3_etth1_h96_3seeds.py
?? scripts/summarize_minimal_repro.py
?? scripts/summarize_stage2_illness_h24.py
?? scripts/summarize_stage3_etth1_h96.py
```

- 本阶段新增或更新文件:
  - `scripts/run_stage3_etth1_h96_3seeds.py`
  - `scripts/summarize_stage3_etth1_h96.py`
  - `results/stage3_etth1_h96/raw_results.csv`
  - `results/stage3_etth1_h96/summary_mean_std_stderr.csv`
  - `results/stage3_etth1_h96/summary_mean_std_stderr.md`
  - `results/stage3_etth1_h96/baseline_best_improvement.txt`
  - `results/stage3_etth1_h96/run_history.csv`
  - `results/stage3_etth1_h96/failed_runs.json`
  - `results/stage3_etth1_h96/failed_runs_before_data_ready.json`
  - `STAGE3_ETTH1_H96_3SEED.md`
- 是否修改核心算法代码: 否

## 5. 运行过程

- dry-run 是否通过: 是。确认了 dataset=ETTh1、horizon=96、context_length=512、model=MOMENT-small；baseline 不含 `--adapter`，其它 adapter 含 `--adapter`。
- seed=13 是否先行完成: 是。第一次 seed=13 因外层超时导致 linearVAE/VAE return code 120；随后使用 `--skip_existing` 成功补跑。
- 3 seeds 是否全部完成: 非 PCA 的 5 个 adapter 均完成 3 seeds，共 15 个成功组合。PCA 的 3 seeds 均失败。
- 失败组合:
  - seed=13, adapter=PCA, returncode=1
  - seed=42, adapter=PCA, returncode=1
  - seed=2024, adapter=PCA, returncode=1
- 运行命令:

```bash
python scripts/run_stage3_etth1_h96_3seeds.py --dry_run
python scripts/run_stage3_etth1_h96_3seeds.py --only_seed 13
python scripts/run_stage3_etth1_h96_3seeds.py --skip_existing
python scripts/summarize_stage3_etth1_h96.py
```

## 6. 原始结果

- raw_results.csv 路径: `results/stage3_etth1_h96/raw_results.csv`
- 原始结果行数: 90
- 原始结果列名: `dataset`, `foundational_model`, `adapter`, `n_features`, `n_components`, `is_fine_tuned`, `pca_in_preprocessing`, `use_revin`, `context_length`, `forecast_horizon`, `running_time`, `seed`, `metric`, `value`, `train_size`

per-seed MSE/MAE:

| adapter | seed | mse | mae |
|---|---:|---:|---:|
| baseline | 13 | 0.389243 | 0.410324 |
| baseline | 42 | 0.388422 | 0.409619 |
| baseline | 2024 | 0.388832 | 0.409838 |
| linearAE | 13 | 0.394896 | 0.417938 |
| linearAE | 42 | 0.403747 | 0.421961 |
| linearAE | 2024 | 0.394772 | 0.416806 |
| dropoutLinearAE | 13 | 0.398486 | 0.419226 |
| dropoutLinearAE | 42 | 0.399407 | 0.420021 |
| dropoutLinearAE | 2024 | 0.394576 | 0.416378 |
| linearVAE | 13 | 0.410008 | 0.424253 |
| linearVAE | 42 | 0.405559 | 0.421515 |
| linearVAE | 2024 | 0.409095 | 0.426196 |
| VAE | 13 | 0.410016 | 0.432161 |
| VAE | 42 | 0.410364 | 0.435633 |
| VAE | 2024 | 0.395915 | 0.422535 |

## 7. 聚合结果

| adapter         | mse_mean | mse_std  | mse_stderr | mae_mean | mae_std  | mae_stderr | n |
| --------------- | -------- | -------- | ---------- | -------- | -------- | ---------- | - |
| baseline        | 0.388832 | 0.000411 | 0.000237   | 0.409927 | 0.000360 | 0.000208   | 3 |
| linearAE        | 0.397805 | 0.005147 | 0.002971   | 0.418902 | 0.002709 | 0.001564   | 3 |
| dropoutLinearAE | 0.397490 | 0.002565 | 0.001481   | 0.418542 | 0.001915 | 0.001106   | 3 |
| linearVAE       | 0.408220 | 0.002350 | 0.001357   | 0.423988 | 0.002352 | 0.001358   | 3 |
| VAE             | 0.405431 | 0.008243 | 0.004759   | 0.430110 | 0.006786 | 0.003918   | 3 |

- baseline MSE mean: `0.388832`
- best adapter by MSE: `baseline`
- best non-baseline adapter by MSE: `dropoutLinearAE`, MSE mean = `0.397490`
- improvement vs baseline: `0.00%` for best overall; `-2.23%` for best non-baseline.

## 8. 与论文 Table 1 的初步比较

论文 ETTh1 H=96 MSE 参考值：

| Adapter | Paper MSE on ETTh1 H=96 |
|---|---:|
| baseline Moment | 0.411 ± 0.012 |
| PCA | 0.433 ± 0.001 |
| LinearAE | 0.402 ± 0.002 |
| dropoutLinearAE | 0.395 ± 0.003 |
| LinearVAE | 0.400 ± 0.001 |
| VAE | 0.404 ± 0.001 |

本地结果中，adapter 没有整体优于 baseline；dropoutLinearAE 是非 baseline 中最优，也接近论文里 dropoutLinearAE 较强的趋势，但仍弱于本地 baseline。PCA 在当前 `ft_then_supervised` 路径下失败，无法比较。probabilistic adapters 中，linearVAE 三个 seed 较稳定但误差偏高；VAE 的 seed 间波动较大。

整体看，本地 ETTh1 H=96 的绝对数值接近论文量级，但 adapter 相对 baseline 的趋势不一致。

## 9. 与阶段二 Illness H=24 的对比

| Dataset | Best adapter | Baseline MSE | Best MSE | Improvement |
|---|---|---:|---:|---:|
| Illness H=24 | baseline | 2.617666 | 2.617666 | 0.00% |
| ETTh1 H=96 | baseline | 0.388832 | 0.388832 | 0.00% |

阶段二 Illness H=24 与本阶段 ETTh1 H=96 都显示 baseline 的 MSE 最低。两个数据集上，当前小规模训练设置都没有复现出 adapter 明显优于 baseline 的论文趋势。

## 10. 失败运行记录

- failed_runs.json: `results/stage3_etth1_h96/failed_runs.json`
- 旧数据缺失失败记录备份: `results/stage3_etth1_h96/failed_runs_before_data_ready.json`

失败摘要：

| seed | adapter | error type | possible reason | rerun needed |
|---:|---|---|---|---|
| 13 | PCA | returncode=1 | `ft_then_supervised` 会调用 adapter supervised fine-tuning；PCA projector 不是 PyTorch Module，和阶段二 PCA 失败模式一致 | 需要先确认 PCA 应走非 supervised 或 preprocessing 路径 |
| 42 | PCA | returncode=1 | 同上 | 同上 |
| 2024 | PCA | returncode=1 | 同上 | 同上 |

说明：PCA 日志文件没有保存完整 stderr traceback；上述原因是基于相同代码路径与阶段二已捕获的 `AssertionError: adapter must be a PyTorch Module` 推断。

## 11. 初步结论

1. ETTh1 H=96 是否稳定跑完 3 seeds: baseline、linearAE、dropoutLinearAE、linearVAE、VAE 均稳定跑完 3 seeds；PCA 未跑通。
2. adapter 是否整体优于 baseline: 否。
3. 哪个 adapter 当前最好: 按 MSE 是 baseline；非 baseline 中是 dropoutLinearAE。
4. 本地结果和论文趋势是否一致: 不完全一致。绝对量级接近，但相对趋势不一致。
5. 与 Illness H=24 相比，adapter 有效性是否稳定: 当前两个数据集都没有显示 adapter 优于 baseline。
6. 可能原因: 未做 hyperopt、训练轮数仍少于完整论文复现、PCA 路径与 supervised fine-tuning 不匹配、adapter 超参数未调、依赖版本与随机性差异。

## 12. 下一步建议

1. 优先排查 PCA：确认论文中的 PCA 是否应使用非 supervised adapter 路径或 `pca_in_preprocessing`。
2. 若继续 ETTh1 H=96，建议先扩大训练轮数或复核 adapter 超参数，而不是直接扩展到更多数据集。
3. dropoutLinearAE 是当前非 baseline 中最强，可优先做 latent dimension 或 dropout 相关消融。
4. 若关注 probabilistic adapter，建议进入 calibration / ECE 分析，因为 linearVAE/VAE 的 MSE 不占优，但 ECE/KS 指标可能有不同信号。
