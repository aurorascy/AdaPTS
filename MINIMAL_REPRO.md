# AdaPTS 最小复现记录

## 1. 实验目标

本次只复现 Illness H=24 上的 MOMENT-small baseline、linearVAE、VAE。

实验组合：

- Dataset: Illness
- Forecast horizon: 24
- Context length: 512
- Foundation model: `AutonLab/MOMENT-1-small`
- Adapters: `None`, `linearVAE`, `VAE`
- Seed: 13
- Device: `cuda:0`

## 2. 环境信息

- OS: Microsoft Windows 11 Pro 10.0.26200, 64-bit
- Python: 3.10.11
- CUDA available: True
- GPU: NVIDIA GeForce RTX 3060
- torch: 2.12.0+cu126
- numpy: 1.25.2
- pandas: 2.3.3

## 3. 数据检查

- Illness.csv 路径: `external_data/forecasting/Illness/Illness.csv`
- 是否存在: True
- 数据 shape: `(966, 8)`
- 数据前 5 行摘要:

```text
date                 % WEIGHTED ILI ... NUM. OF PROVIDERS      OT
2002-01-01 00:00:00  1.22262       ... 754                176569
2002-01-08 00:00:00  1.33344       ... 785                186355
2002-01-15 00:00:00  1.31929       ... 831                192469
2002-01-22 00:00:00  1.49484       ... 863                207512
2002-01-29 00:00:00  1.47195       ... 909                223208
```

## 4. 代码状态

- git commit: `8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278`
- git status: 工作区有未提交改动。
- 本阶段是否修改代码: 是，但未修改核心算法逻辑。
- 本阶段新增文件:
  - `scripts/summarize_minimal_repro.py`
  - `MINIMAL_REPRO.md`
- 本阶段修改原因:
  - 新增汇总脚本，将长格式结果 CSV 整理为每个 adapter 一行的简洁表格。
  - 新增本报告，记录最小复现实验过程、结果、日志和限制。
- 进入本阶段前已有的未提交文件:
  - `.gitignore`
  - `src/adapts/utils/main_script.py`
  - `src/adapts/icl/moment.py`
  - `ENV_SETUP.md`

## 5. 运行命令

本机 `tyro` help 输出使用短横线参数名，例如 `--forecast-horizon`，因此以下命令均使用短横线形式。

### baseline

```powershell
python scripts\run.py `
  --forecast-horizon 24 `
  --model-name "AutonLab/MOMENT-1-small" `
  --context-length 512 `
  --seed 13 `
  --device "cuda:0" `
  --dataset-name "Illness" `
  --use-revin `
  --supervised "ft_then_supervised" `
  --n-epochs-fine-tuning 5 `
  --n-epochs-adapter 20 `
  --data-path "results/minimal_repro/illness_h24_minimal.csv" `
  --log-dir "logs/minimal_repro/formal"
```

### linearVAE

```powershell
python scripts\run.py `
  --forecast-horizon 24 `
  --model-name "AutonLab/MOMENT-1-small" `
  --context-length 512 `
  --seed 13 `
  --device "cuda:0" `
  --dataset-name "Illness" `
  --adapter "linearVAE" `
  --use-revin `
  --supervised "ft_then_supervised" `
  --n-epochs-fine-tuning 5 `
  --n-epochs-adapter 20 `
  --data-path "results/minimal_repro/illness_h24_minimal.csv" `
  --log-dir "logs/minimal_repro/formal"
```

### VAE

```powershell
python scripts\run.py `
  --forecast-horizon 24 `
  --model-name "AutonLab/MOMENT-1-small" `
  --context-length 512 `
  --seed 13 `
  --device "cuda:0" `
  --dataset-name "Illness" `
  --adapter "VAE" `
  --use-revin `
  --supervised "ft_then_supervised" `
  --n-epochs-fine-tuning 5 `
  --n-epochs-adapter 20 `
  --data-path "results/minimal_repro/illness_h24_minimal.csv" `
  --log-dir "logs/minimal_repro/formal"
```

## 6. 结果文件

- 原始结果 CSV: `results/minimal_repro/illness_h24_minimal.csv`
- 汇总结果 CSV: `results/minimal_repro/illness_h24_minimal_summary.csv`
- Smoke test CSV: `results/minimal_repro/illness_h24_smoke.csv`
- 日志目录: `logs/minimal_repro/formal`

正式日志目录包含：

- 3 个 `config.json`
- 3 个 `run_*.log`
- 9 个 `.npy` 指标文件
- 2 个 `adapter.pt` checkpoint，来自 linearVAE 和 VAE
- TensorBoard event 文件

## 7. 最小复现结果

| adapter | MSE | MAE | scaled_MSE | scaled_MAE | KS | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| None | 2.598519 | 1.108536 | 2.048735 | 1.030718 | 0.720098 | 0.325136 |
| linearVAE | 3.507913 | 1.248891 | 2.464350 | 1.114796 | 0.601541 | 0.283761 |
| VAE | 2.882472 | 1.103072 | 2.129563 | 1.009221 | 0.556232 | 0.270967 |

## 8. 与论文结果的关系

本阶段使用较少 epoch，只用于验证流程，不要求数值完全对齐论文 Table 1。论文完整实验需要更多训练轮数、超参数搜索和多随机种子。

## 9. 遇到的问题与解决方案

- `scripts/run.py --help` 显示参数使用短横线形式，例如 `--forecast-horizon`，不是下划线形式。解决方案：所有实际运行命令使用短横线参数。
- baseline 不传 `--adapter`，保存到 CSV 时 `adapter` 列为空值。解决方案：汇总脚本中将空值映射为 `None`，防止 pandas pivot 时丢失 baseline。
- MOMENT import 会出现 `transformers` FutureWarning，和 PyTorch pytree 注册 API 变化有关。该 warning 不影响训练和推理。
- 训练中出现 `torch.utils.checkpoint` 的 `use_reentrant` warning。该 warning 不影响本次最小复现流程。
- 当前工作区包含上一阶段环境搭建留下的未提交修改；本阶段未回退这些修改，也未修改核心算法逻辑。

## 10. 下一步建议

建议下一步进入：

1. Illness H=24 的 3 seeds；
2. 使用更接近论文的训练 epoch；
3. 再扩展到 ETTh1 H=96。
