# ETTh1 数据准备记录

## 1. 任务背景

阶段三 ETTh1 H=96 复现实验因缺少 `external_data/forecasting/ETTh1/ETTh1.csv` 被停止。本阶段仅补齐和校验数据，不启动训练。

## 2. 数据来源

- Source: zhouhaoyi/ETDataset
- Raw URL: `https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv`
- Local path: `external_data/forecasting/ETTh1/ETTh1.csv`

## 3. 下载脚本

- Script: `scripts/download_etth1.py`
- Command:

```bash
python scripts/download_etth1.py
```

脚本使用 Python 标准库 `urllib.request` 下载文件；如果目标文件已存在且未传入 `--force`，会跳过下载并执行校验。

## 4. CSV 校验结果

- Exists: True
- File size: 2589657 bytes
- Shape: `(17420, 8)`
- Columns: `date`, `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`
- Numeric feature columns: `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`
- Missing values: 0

Head:

```text
               date  HUFL  HULL  MUFL  MULL  LUFL  LULL        OT
2016-07-01 00:00:00 5.827 2.009 1.599 0.462 4.203 1.340 30.531000
2016-07-01 01:00:00 5.693 2.076 1.492 0.426 4.142 1.371 27.787001
2016-07-01 02:00:00 5.157 1.741 1.279 0.355 3.777 1.218 27.787001
2016-07-01 03:00:00 5.090 1.942 1.279 0.391 3.807 1.279 25.044001
2016-07-01 04:00:00 5.358 1.942 1.492 0.462 3.868 1.279 21.948000
```

Tail:

```text
               date   HUFL  HULL   MUFL  MULL  LUFL  LULL     OT
2018-06-26 15:00:00 -1.674 3.550 -5.615 2.132 3.472 1.523 10.904
2018-06-26 16:00:00 -5.492 4.287 -9.132 2.274 3.533 1.675 11.044
2018-06-26 17:00:00  2.813 3.818 -0.817 2.097 3.716 1.523 10.271
2018-06-26 18:00:00  9.243 3.818  5.472 2.097 3.655 1.432  9.778
2018-06-26 19:00:00 10.114 3.550  6.183 1.564 3.716 1.462  9.567
```

## 5. AdaPTS DataReader 校验

- Script: `scripts/check_etth1_datareader.py`
- Command:

```bash
python scripts/check_etth1_datareader.py
```

- Result: passed
- Dataset name used for reader check: `ETTh1_pred=96`
- Shapes:

```text
train X shape: (8033, 7, 512)
train y shape: (8033, 7, 96)
val X shape: (2785, 7, 512)
val y shape: (2785, 7, 96)
test X shape: (2785, 7, 512)
test y shape: (2785, 7, 96)
```

`src/adapts/utils/data_readers.py` 中 forecasting 数据按如下模式读取：

```text
external_data/forecasting/<dataset_name>/<dataset_name>.csv
```

本次保存路径符合 AdaPTS 的 DataReader 预期。

## 6. 代码状态

- git status:

```text
 M .gitignore
 M STAGE3_ETTH1_H96_3SEED.md
 M src/adapts/icl/moment.py
 M src/adapts/utils/main_script.py
?? DATA_PREP_ETTH1.md
?? ENV_SETUP.md
?? MINIMAL_REPRO.md
?? STAGE2_ILLNESS_3SEED.md
?? scripts/check_etth1_datareader.py
?? scripts/download_etth1.py
?? scripts/run_stage2_illness_h24_3seeds.py
?? scripts/summarize_minimal_repro.py
?? scripts/summarize_stage2_illness_h24.py
```

- 新增文件:
  - `scripts/download_etth1.py`
  - `scripts/check_etth1_datareader.py`
  - `external_data/forecasting/ETTh1/ETTh1.csv`
  - `DATA_PREP_ETTH1.md`
- 是否修改核心算法代码: 否

## 7. 结论

ETTh1.csv 已从官方 ETTDataset 数据源下载到 AdaPTS 期望路径，CSV 层校验通过，DataReader 对 `ETTh1_pred=96` 的 train/val/test 读取也通过。可以继续执行 Stage 3：ETTh1 H=96 三随机种子实验。

## 8. 下一步建议

如果校验通过，下一步运行：

```bash
python scripts/run_stage3_etth1_h96_3seeds.py --dry_run
python scripts/run_stage3_etth1_h96_3seeds.py --only_seed 13
```
