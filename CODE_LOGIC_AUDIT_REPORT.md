# AdaPTS 本地代码逻辑审查报告

## 1. 审查目的

当前本地结果中，`baseline` / `strong-baseline` 多次强于 named adapters，与论文中 adapter 优于 baseline 的整体趋势不一致。本次审查暂停继续训练，只检查本地代码、已有 config、已有 logs 和已有 CSV，判断异常是否来自代码 bug、协议差异、结果标记错误、metric 汇总错误或训练不足。

## 2. 审查范围

重点审查文件：

- `scripts/run.py`
- `scripts/run_stage7_remaining_table1_with_baseline_types.py`
- `scripts/summarize_stage7_with_baseline_types.py`
- `src/adapts/adapts.py`
- `src/adapts/adapters.py`
- `src/adapts/utils/main_script.py`
- `src/adapts/utils/data_readers.py`
- `src/adapts/icl/moment.py`

重点结果与日志：

- `results/stage2_illness_h24/raw_results.csv`
- `results/stage3_etth1_h96/raw_results.csv`
- `results/stage7_remaining_table1_with_baseline_types/raw_results.csv`
- `results/stage7_remaining_table1_with_baseline_types/summary_by_dataset_horizon_protocol.csv`
- `logs/**/config.json`
- `logs/**/run_*.log`

输出目录：

- `results/code_logic_audit/`

## 3. 主运行流程

完整流程见 `results/code_logic_audit/run_flow_audit.md`。

简化流程：

```text
scripts/run.py
  -> prepare_data
  -> load MOMENT
  -> instantiate adapter / JustRevIn / IdentityTransformer
  -> MultichannelProjector
  -> ADAPTS
  -> fine_tune_iclearner and/or adapter_supervised_fine_tuning
  -> predict_multi_step
  -> compute_metrics
  -> save_metrics_to_csv
```

关键结论：

- `adapter=None + use_revin=True + supervised=ft_then_supervised` 会创建 `JustRevIn`，并训练其 RevIN affine 参数；这就是 `strong-baseline`。
- `adapter=None + use_revin=False + supervised=ft` 不进入 `adapter_supervised_fine_tuning`，但会调用 `fit_adapter`，此时 base projector 是 `IdentityTransformer`，没有 trainable adapter 参数；这是 Stage 7 的 head-only `baseline`。
- named adapters 走 `fine_tune_iclearner(use_adapter=False)` 后再进入 `adapter_supervised_fine_tuning`。
- 本地 audited runs 的 `n_components` 实际均为原始特征数：ETTh1/Illness=7，Weather=21，ExchangeRate=8。

## 4. Baseline 类型审查

协议标签审查脚本：

- `scripts/audit_protocol_labels.py`
- 输出：`results/code_logic_audit/protocol_label_anomalies.csv`
- 输出：`results/code_logic_audit/protocol_label_summary.csv`

结论：

- 协议标记异常数：`0`。
- 旧 Stage 2 / Stage 3 的 empty adapter + `use_revin=True` 已在 Stage 7 汇总中重标记为 `strong-baseline`。
- Stage 7 ETTh1 H=192 的 true baseline 标记为 `protocol_type=baseline`，`use_revin=False`，`supervised=ft`。
- named adapters 标记为 `protocol_type=adapter`。
- 没有发现把 `strong-baseline` 误写成 `baseline` 的 Stage 7 结果行。

## 5. 数据读取与切分审查

数据审查输出：

- `results/code_logic_audit/data_pipeline_audit.md`
- `results/code_logic_audit/data_shape_audit.csv`

关键 shape：

| Dataset | H | train X/y | val X/y | test X/y |
|---|---:|---|---|---|
| Illness | 24 | (141, 7, 512) / (141, 7, 24) | (74, 7, 512) / (74, 7, 24) | (170, 7, 512) / (170, 7, 24) |
| Illness | 60 | (105, 7, 512) / (105, 7, 60) | (38, 7, 512) / (38, 7, 60) | (134, 7, 512) / (134, 7, 60) |
| ETTh1 | 96 | (8033, 7, 512) / (8033, 7, 96) | (2785, 7, 512) / (2785, 7, 96) | (2785, 7, 512) / (2785, 7, 96) |
| ETTh1 | 192 | (7937, 7, 512) / (7937, 7, 192) | (2689, 7, 512) / (2689, 7, 192) | (2689, 7, 512) / (2689, 7, 192) |

`DataReader` 对 forecasting 数据先按时间划分 train/val/test，再用训练集 fit `StandardScaler`，然后 transform train/val/test。这个部分没有发现 test target 泄漏。

注意：`mse` 不是原始 CSV 物理尺度上的 MSE，而是在 DataReader 已标准化后的时间序列尺度上计算。

## 6. Scaler / RevIN 审查

详见 `results/code_logic_audit/scaler_revin_audit.md`。

最重要发现：`ADAPTS.transform()` 在预测阶段调用 `self.scaler.fit_transform(X)`，而 `predict_multi_step()` 传入的是 test context。

证据：

- `src/adapts/adapts.py:123`：head fine-tune 对训练 X 调用 `fit_transform`
- `src/adapts/adapts.py:442`：adapter training 对 `concat(X_train, y_train)` 重新 `fit`
- `src/adapts/adapts.py:152`：`transform()` 调用 `fit_transform`
- `src/adapts/adapts.py:203`：prediction 调用 `transform(test_context)`

这不是直接使用 future target 的经典泄漏，但它是 **test-context scaler refit**，会让推理时使用测试窗口自身分布归一化，并覆盖训练阶段 scaler 状态。严重程度评估为：`high / uncertain`。

这可能解释 baseline / strong-baseline 异常强或 adapter 关系不稳定，因为所有协议都经过这一套 ADAPTS scaler，但不同协议与 RevIN / adapter 的交互不同。

## 7. MOMENT wrapper 审查

详见 `results/code_logic_audit/moment_wrapper_audit.md`。

关键结论：

- MOMENT 默认逐通道处理多变量序列，每个 feature 以 `[batch, 1, context_length]` 输入。
- `load_moment_model` 请求 `freeze_encoder=True`、`freeze_embedder=True`、`freeze_head=False`。
- `MomentICLTrainer.fine_tune` 的 optimizer 使用 `self.model.parameters()`；实际是否只更新 head 依赖 MOMENT 内部是否正确设置 frozen 参数的 `requires_grad=False`。
- 每个 Stage runner 以 subprocess 调用 `scripts/run.py`，所以不同 seed/protocol/adapter 之间不会复用同一个内存模型。
- 如果未来给 `fine_tune` 传入 `X_val/y_val`，validation 路径没有应用 `direct_transform/inverse_transform`，但当前 audited runs 没有传入验证集给 MOMENT fine-tune，因此不影响当前结果。

## 8. Adapter 训练审查

输出：

- `results/code_logic_audit/adapter_training_audit.md`
- `results/code_logic_audit/trainable_parameter_audit.csv`

关键结论：

- `strong-baseline` 的 `JustRevIn` 确实有 trainable params：7 特征时 14 个参数。
- `linearAE` / `dropoutLinearAE` 7 特征时各 126 个 trainable params。
- `linearVAE` 7 特征时 182 个 trainable params。
- `VAE` 7 特征时 38819 个 trainable params。
- PCA 是 sklearn `PCA`，不是 `torch.nn.Module`，因此在 `adapter_supervised_fine_tuning` 的 assert 下不兼容。这是协议问题，不是训练结果问题。

发现一个未来 n_components 消融风险：

- `LinearAutoEncoder.forward` 和 `DropoutLinearAutoEncoder.forward` 的 RevIN 分支把 normalized input reshape 到 `(-1, self.n_components)` 后再送入 encoder，见 `src/adapts/adapters.py:397-403` 和 `src/adapts/adapters.py:607-613`。
- 当 `n_components != input_dim` 时，这很可能是 shape bug。
- 当前结果中 `n_components == input_dim`，所以这个 bug 被掩盖，不解释当前 ETTh1/Illness 主结果，但会影响后续 latent dimension 消融。

## 9. Metric 与结果写入审查

审查脚本：

- `scripts/audit_metrics_and_duplicates.py`

输出：

- `results/code_logic_audit/metric_duplicate_audit.csv`
- `results/code_logic_audit/summary_recompute_check.csv`
- `results/code_logic_audit/metric_audit.md`

结论：

- 重新从 raw results 聚合 `metric == "mse"` 和 `metric == "mae"`，与现有 Stage 7 summary 完全一致。
- 未发现 duplicate seed/protocol/adapter/metric 行。
- `summary_by_dataset_horizon_protocol.csv` 没有误用 `scaled_mse`、`ks` 或 `ece`。
- `n` 等于实际 seed 数。
- Stage 7 summary 对旧结果的重标记是可信的。

## 10. 训练日志审查

输出：

- `results/code_logic_audit/training_log_logic_audit.csv`
- `results/code_logic_audit/training_log_logic_audit.md`

结论：

- named adapters 和 strong-baseline 日志中出现了 `Done fine tuning, now training adapter` 与 `adapter fitted`，说明确实进入了 adapter/RevIN training。
- Stage 7 baseline 没有进入 `adapter_supervised_fine_tuning`，但因为 `supervised=ft` 分支会调用 `fit_adapter`，日志仍有误导性文本 `now training adapter`。
- head fine-tune 的 train/val loss 多数下降。
- adapter training 的详细 batch/validation loss 主要写在 TensorBoard event 文件中；普通 log 只记录 early stopping/restoring/fitted。

## 11. 发现的问题清单

| ID | 问题 | 类型 | 严重程度 | 证据 | 影响 | 建议 |
|---|---|---|---|---|---|---|
| A01 | ADAPTS prediction 阶段在 test context 上 `fit_transform` scaler | code_bug / protocol_mismatch | high | `src/adapts/adapts.py:152`, `src/adapts/adapts.py:203` | 可能改变 baseline/adapter 表现，造成 test-time normalization | 增加 train-fitted scaler ablation；确认官方意图 |
| A02 | ADAPTS scaler 在 head fine-tune、adapter training、prediction 多次 refit | code_bug / protocol_mismatch | high | `src/adapts/adapts.py:123`, `src/adapts/adapts.py:442`, `src/adapts/adapts.py:152` | head 与 adapter / inference 使用不同 scaler state | 固定 scaler 生命周期或拆分协议 |
| A03 | 旧 baseline 实际为 strong-baseline | naming_issue | high | `scripts/run.py:288-294`, `scripts/run.py:356-383` | 旧报告易误读 | 已通过 `protocol_type` 修正 |
| A04 | 当前 fast epoch 远小于论文完整训练设置 | protocol_mismatch | high | config audit: ft=5, adapter=30 | adapter 可能训练不足 | 单任务 full epoch 复核 |
| A05 | 未运行 hyperopt | protocol_mismatch | high | 本地阶段约束 | adapter 超参可能非最优 | 稳定协议后再 hyperopt |
| A06 | PCA 不兼容 ft_then_supervised | protocol_mismatch | medium | `adapter_supervised_fine_tuning` requires torch module | PCA 缺失，Table 1 不完整 | 单独找 pca_in_preprocessing / non-supervised 路径 |
| A07 | MSE 是 DataReader 标准化尺度，不是原始 CSV 尺度 | expected_behavior | medium | `data_readers.py:343-347` | 与论文数值比较需要谨慎 | 报告中明确 metric scale |
| A08 | LinearAE/DropoutLinearAE RevIN forward 在 n_components != D 时可能 shape bug | code_bug | medium | `adapters.py:397-403`, `adapters.py:607-613` | 后续 latent dimension 消融可能错误 | 消融前修复并加 shape test |
| A09 | `MomentICLTrainer.fine_tune` optimizer 包含 `self.model.parameters()` | uncertain | low/medium | `moment.py:217` | 是否只训练 head 依赖 MOMENT freeze flags | 加只读 requires_grad audit |
| A10 | `supervised=ft` 日志写 "now training adapter" | naming_issue | low | `scripts/run.py:436-439` | 容易误解 baseline 路径 | 改日志或报告中说明 |

## 12. 最可能导致当前结果异常的原因排序

1. **ADAPTS scaler 在 prediction 阶段对 test context 重新 fit。**  
   这是当前最大的实现/协议风险，可能显著改变 baseline 和 adapter 的相对表现。

2. **训练协议比论文轻很多，且未 hyperopt。**  
   当前快速档 ft=5、adapter=30，named adapters 可能没有达到论文训练条件。

3. **baseline 定义此前混淆，strong-baseline 包含 JustRevIn affine training。**  
   已经开始修正命名，但旧结果解释仍要谨慎。

4. **DataReader 标准化 + ADAPTS scaler + RevIN 三层归一化相互作用。**  
   baseline、strong-baseline、adapter 经过的归一化组合不同，可能导致局部最优关系和论文不同。

5. **MOMENT / momentfm 版本与论文版本可能不同。**  
   本地 baseline 明显强或弱时，foundation model 版本差异必须纳入解释。

## 13. 当前哪些结果可信，哪些需要谨慎解释

可信：

- CSV 聚合逻辑可信：summary 由 `mse/mae` 重新计算后匹配。
- Stage 7 protocol labels 可信：未发现 baseline/strong-baseline/adapter 标记错误。
- ETTh1 H=192 的 baseline / strong-baseline / adapters 三随机种子结果可追踪。
- named adapters 确实进入了 adapter supervised training。

需要谨慎解释：

- ETTh1 H=96 和 Illness H=24 旧结果只能解释为 `strong-baseline` vs adapters，不能解释为 pure baseline vs adapters。
- 所有 MSE/MAE 是 DataReader 标准化后尺度，不是原始 CSV 物理尺度。
- 当前 adapter 弱于 baseline 的结论只适用于本地 fast setting 和当前 scaler/RevIN 实现。
- 在确认 ADAPTS scaler prediction-time refit 是否符合论文协议前，不宜声称代码完全复现论文协议。

## 14. 是否需要修代码

当前结论：

1. **发现了需要进一步确认/很可能需要修的代码点：ADAPTS prediction 阶段 scaler `fit_transform`。**  
   这可能是必须修复的逻辑 bug，也可能是作者原始协议的一部分；需要对照官方实现/论文说明确认。

2. **发现了 latent dimension 消融前必须修的 adapter forward shape 风险。**  
   当前 n_components=D，所以不影响已有主结果，但会影响后续 n_components < D。

3. **没有发现 summary metric 选错、重复行、seed 污染或 Stage 7 protocol label 错误。**

4. **不建议现在大改核心算法。**  
   应先做最小 ablation：保持训练不变，只比较 current scaler behavior vs train-fitted scaler behavior。

## 15. 下一步建议

优先级：

1. 先新增一个只读/小规模 debug path，验证 `ADAPTS.transform` 改成 `self.scaler.transform(X)` 后 ETTh1 H=192 seed=13 baseline/linearAE 的 MSE 是否变化。不要直接覆盖原逻辑。
2. 做 `requires_grad` audit，确认 MOMENT head-only fine-tune 是否真的只更新 head。
3. 修复或至少单测 LinearAE/DropoutLinearAE 在 `n_components < input_dim` 且 `use_revin=True` 时的 forward shape。
4. 在上述点确认后，再决定是否重跑 ETTh1 H=192 seed=13 的 baseline vs linearAE/dropoutLinearAE。
5. 暂缓 Weather / ExchangeRate 扩展，避免在协议未确认前扩大错误矩阵。

