# Issue Origin Audit：官方代码 vs 本地复现代码

## 1. 审查目的

本次审查用于判断 Stage 9 发现的问题，是官方原始代码中已经存在，还是本地复现过程中修改引入。审查只做代码对比、hash、diff 和 pattern 静态分析；没有训练模型，没有运行实验矩阵，也没有修复核心代码。

## 2. 审查对象

- Official root: `F:\AdaPTS\demo`
- Official project root detected: `F:\AdaPTS\demo\AdaPTS`
- Local root: `F:\AdaPTS`

审查文件：

| Logical file | Official file found | Local file found |
|---|---|---|
| `scripts/run.py` | yes | yes |
| `src/adapts/adapts.py` | yes | yes |
| `src/adapts/adapters.py` | yes | yes |
| `src/adapts/utils/main_script.py` | yes | yes |
| `src/adapts/utils/data_readers.py` | yes | yes |
| `src/adapts/icl/moment.py` | yes | yes |

自动对比脚本：

```text
scripts/audit_issue_origin_against_official.py
```

输出目录：

```text
results/issue_origin_audit/
```

## 3. 文件差异总览

| File | Same hash | Meaning |
|---|---:|---|
| `scripts/run.py` | True | 官方与本地完全一致 |
| `src/adapts/adapts.py` | True | 官方与本地完全一致 |
| `src/adapts/adapters.py` | True | 官方与本地完全一致 |
| `src/adapts/utils/main_script.py` | False | 本地改了数据路径 |
| `src/adapts/utils/data_readers.py` | True | 官方与本地完全一致 |
| `src/adapts/icl/moment.py` | False | 本地改了 `local_files_only` |

生成的 diff 文件：

- `results/issue_origin_audit/diff_run_py.patch`，长度 0
- `results/issue_origin_audit/diff_adapts_py.patch`，长度 0
- `results/issue_origin_audit/diff_adapters_py.patch`，长度 0
- `results/issue_origin_audit/diff_data_readers_py.patch`，长度 0
- `results/issue_origin_audit/diff_main_script_py.patch`
- `results/issue_origin_audit/diff_moment_py.patch`

这说明 Stage 9 的核心问题所涉及的 `run.py`、`adapts.py`、`adapters.py` 均不是本地改出来的。

## 4. 问题来源总表

| Issue | Origin | Severity | Official evidence | Local evidence | Notes |
|---|---|---|---|---|---|
| B01 scaler prediction refit | official-existing | high | `adapts.py:152 fit_transform` | same | 官方已有 prediction/test-context scaler refit |
| R01 prediction train mode | official-existing | medium | `adapts.py:198 base_projector_.train()` | same | 官方预测阶段开启 train mode |
| R02 VAE eval sampling | official-existing | medium | `adapters.py:1434 torch.randn_like` | same | VAE/linearVAE 无条件 sampling |
| B02 LinearAE RevIN shape | official-existing | medium | `adapters.py:402 reshape(-1, self.n_components)` | same | 官方已有 shape 风险 |
| B03 DropoutLinearAE RevIN shape | official-existing | medium | `adapters.py:612 reshape(-1, self.n_components)` | same | 官方已有 shape 风险 |
| P01 strong-baseline JustRevIn | official-existing | high | `run.py:288`, `run.py:356` | same | 官方代码路径就是 strong-baseline |
| P02 MOMENT local_files_only | local-modified | low | official `True` | local `False` | 本地为首次下载改动 |
| P03 data path change | local-modified | low | author absolute path | repo-relative path | 只影响数据位置 |

## 5. B01 Scaler Prediction Refit

结论：`official-existing`。

官方和本地 `src/adapts/adapts.py` 完全同 hash，且都包含：

```text
152: return self.adapter.transform(self.scaler.fit_transform(X))
```

`predict_multi_step` 会调用：

```text
203: X_transformed = self.transform(X[:, :, :-prediction_horizon])
```

因此预测阶段在 test context X 上 refit scaler 的行为是官方原始代码已有，不是本地改出来的。

## 6. Prediction Mode / Stochastic Evaluation

结论：`official-existing`。

官方和本地 `src/adapts/adapts.py` 都包含：

```text
198: self.adapter.base_projector_.train()
```

官方和本地 `src/adapts/adapters.py` 都包含 VAE sampling：

```text
1434: eps = torch.randn_like(std)
1677: eps = torch.randn_like(std)
```

所以 dropout prediction train mode、VAE/linearVAE eval 仍采样，都是官方代码本身行为。

## 7. LinearAE / DropoutLinearAE RevIN Shape Bug

结论：`official-existing`。

官方和本地 `src/adapts/adapters.py` 完全同 hash，且都包含：

```text
402: self.revin(revin_input, mode="norm").reshape(-1, self.n_components)
612: self.revin(revin_input, mode="norm").reshape(-1, self.n_components)
```

这就是 Stage 9 synthetic probe 中 `use_revin=True` 且 `n_components != input_dim` 报 shape error 的来源。它不是本地复现过程引入的。

## 8. Strong-Baseline 行为来源

结论：`official-existing`。

官方和本地 `scripts/run.py` 完全同 hash，且都包含：

```text
288: if not args.adapter and args.use_revin:
289:     adapter = JustRevIn(
356: elif args.supervised == "ft_then_supervised":
371:     adapts_model.adapter_supervised_fine_tuning(
```

因此：

```text
adapter=None
use_revin=True
supervised=ft_then_supervised
```

会创建 `JustRevIn` 并训练其 RevIN affine 参数。这是官方代码路径。此前把它称为 baseline 是本地报告命名/解释问题，不是本地代码引入的新行为。

## 9. MOMENT Loading 改动影响

结论：`local-modified`。

唯一 diff：

```diff
-        local_files_only=True,
+        local_files_only=False,
```

`model_kwargs` 保持一致：

```text
"freeze_encoder": True
"freeze_embedder": True
"freeze_head": False
```

解释：

- 这个改动主要用于首次下载 HuggingFace 模型；
- 如果官方环境已有固定缓存，`local_files_only=True` 会使用缓存；
- 本地 `False` 可能联网下载 checkpoint；
- 因此它不改变代码逻辑，但可能造成 checkpoint 版本/缓存来源差异，进而影响数值。

它不能解释 scaler refit、prediction train mode、shape bug，因为这些文件和官方完全一致。

## 10. 数据路径改动影响

结论：`local-modified`。

官方：

```text
data_path="/mnt/data_2/abenechehab/AdaPTS/external_data/"
```

本地：

```text
repo_root = Path(__file__).resolve().parents[3]
data_path=str(repo_root / "external_data") + "/"
```

`data_readers.py` 与官方同 hash，所以 split、scaler、window 构造逻辑没有被本地修改。该改动只影响数据读取位置，不应解释 baseline/adapter 异常关系，前提是本地 CSV 文件正确。

## 11. 哪些问题是本地写出来的？

本地引入或修改：

| Issue | Type | Impact |
|---|---|---|
| `local_files_only=False` | local-modified | 允许首次下载；可能影响 checkpoint/cache 来源 |
| repo-relative data path | local-modified | 让本地读取 `external_data`；不改变 DataReader 逻辑 |
| baseline/strong-baseline 命名修正脚本 | local-added | 是报告和汇总层面的纠偏，不是核心算法问题 |

没有证据表明 Stage 9 的核心 confirmed bugs 是本地改出来的。

## 12. 哪些问题是官方代码本身就有的？

官方已有：

1. `ADAPTS.transform` prediction/test-context scaler refit；
2. `predict_multi_step` 对 torch adapter 使用 `train()`；
3. VAE/linearVAE reparameterize 无条件 sampling；
4. LinearAE / DropoutLinearAE RevIN branch 在 `n_components != input_dim` 时的 shape bug；
5. `adapter=None + use_revin=True + ft_then_supervised` 实际是 `strong-baseline`。

## 13. 结论

1. Stage 9 的主要 confirmed bugs 是官方已有，不是本地复现代码改出来的。
2. 当前 baseline 强的问题有两层来源：
   - 官方代码协议本身允许 `adapter=None + use_revin=True` 训练 JustRevIn，所以它应叫 `strong-baseline`；
   - 本地早期报告把它误称为 baseline，这是命名/解释问题。
3. 后续修复建议应作为 local patch 明确记录，不要假装是官方原始代码。
4. 当前已有结果应标记为 `official-code pre-fix`，尤其是涉及 scaler prediction refit 和 stochastic prediction mode 的结果。
5. 在修复前，继续扩展 Weather / ExchangeRate 会扩大一个已知 pre-fix 协议矩阵，建议暂停或明确标注。

