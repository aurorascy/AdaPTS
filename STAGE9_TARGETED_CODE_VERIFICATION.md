# Stage 9: Targeted Code Logic Verification

## 1. Background

The previous code logic audit ruled out simple metric aggregation errors and cross-adapter model reuse, but it identified several high-priority risks. Stage 9 verifies those risks with small read-only probes and proposes a minimal fix plan. No training matrix, hyperopt, or Table 1 experiment was run.

Output directory:

```text
results/stage9_targeted_code_verification/
```

## 2. Scaler Fit Behavior

Script:

```text
scripts/verify_scaler_fit_behavior.py
```

Outputs:

```text
results/stage9_targeted_code_verification/scaler_fit_call_trace.csv
results/stage9_targeted_code_verification/scaler_fit_behavior.md
```

Result:

- `ADAPTS.transform(X_context)` calls `scaler.fit_transform`.
- `ADAPTS.predict_multi_step(time_series)` also calls fit-related scaler methods during prediction.
- In the synthetic trace, `predict_multi_step` made 4 fit-related calls for 2 MC samples.
- The prediction path fits on `X[:, :, :-prediction_horizon]`, so it uses test context X, not future test y.

Conclusion:

- Confirmed: scaler is refit during prediction/test-context processing.
- No direct test-y leakage was observed in `predict_multi_step`.
- This is still a high-risk protocol issue because inference does not use only the scaler state learned during training.

## 3. MOMENT Trainable Parameters

Script:

```text
scripts/verify_moment_trainable_params.py
```

Outputs:

```text
results/stage9_targeted_code_verification/moment_trainable_params.csv
results/stage9_targeted_code_verification/moment_trainable_params.md
```

Result for `AutonLab/MOMENT-1-small`:

| Module group | Trainable params |
|---|---:|
| encoder | 0 |
| embedder | 0 |
| head | 3,145,824 |

Conclusion:

- Encoder is frozen.
- Embedder is frozen.
- Forecasting head is trainable.
- Although `MomentICLTrainer.fine_tune` passes `self.model.parameters()` to Adam, frozen encoder/embedder parameters have `requires_grad=False`, so they should not be updated.

## 4. Adapter Prediction Mode

Script:

```text
scripts/verify_adapter_prediction_mode.py
```

Output:

```text
results/stage9_targeted_code_verification/adapter_prediction_mode.md
```

Result:

| Adapter | Mode | Two consecutive outputs identical | Max abs diff |
|---|---|---:|---:|
| dropoutLinearAE | train | False | 4.86289310 |
| dropoutLinearAE | eval | True | 0.00000000 |
| linearVAE | train | False | 10.60854149 |
| linearVAE | eval | False | 11.71002769 |
| VAE | train | False | 6.69177008 |
| VAE | eval | False | 5.36061811 |

Conclusion:

- `ADAPTS.predict_multi_step` currently calls `base_projector_.train()` before prediction.
- dropoutLinearAE is stochastic in train mode and deterministic in eval mode.
- VAE / linearVAE are stochastic even in eval mode because `reparameterize` samples unconditionally.
- Current code mixes deterministic MSE/MAE evaluation and probabilistic sampling in one path.

Recommendation:

- Add explicit `prediction_mode="deterministic"|"stochastic"` and `n_mc_samples`.
- For deterministic VAE metrics, use posterior/latent mean instead of sampling.

## 5. Adapter Shape Consistency

Script:

```text
scripts/verify_adapter_shape_consistency.py
```

Outputs:

```text
results/stage9_targeted_code_verification/adapter_shape_consistency.csv
results/stage9_targeted_code_verification/adapter_shape_consistency.md
```

Synthetic input:

```text
batch=4
seq_len=512
input_dim=7
n_components in [2, 5, 7]
use_revin in [False, True]
```

Failures:

| Adapter | n_components | use_revin | Error |
|---|---:|---:|---|
| linearAE | 2 | True | mat1 and mat2 shapes cannot be multiplied |
| linearAE | 5 | True | invalid reshape |
| dropoutLinearAE | 2 | True | mat1 and mat2 shapes cannot be multiplied |
| dropoutLinearAE | 5 | True | invalid reshape |

Conclusion:

- Confirmed shape bug for LinearAE / DropoutLinearAE when `use_revin=True` and `n_components != input_dim`.
- Current Stage 7/8 runs used `n_components == input_dim`, so this does not explain the existing ETTh1 H=192 results.
- This must be fixed before latent dimension ablations.

## 6. Stage 7 Duplicate / Summary Check

Script:

```text
scripts/audit_stage7_duplicates.py
```

Outputs:

```text
results/stage9_targeted_code_verification/stage7_duplicate_audit.csv
results/stage9_targeted_code_verification/stage7_duplicate_audit.md
```

Result:

- Stage 7 raw rows: 108.
- Duplicate rows at `(dataset, horizon, seed, protocol_type, adapter_label, metric)`: 0.
- Recomputed summary matches current Stage 7 summary: True.
- Old Stage 2/3 empty-adapter results are relabeled as `strong-baseline`.

Conclusion:

- Stage 7 summary is trustworthy.
- No duplicate row pollution was found.

## 7. Confirmed Bugs

| ID | Bug | File | Severity | Evidence | Minimal fix |
|---|---|---|---|---|---|
| B01 | Prediction path refits ADAPTS scaler on test context | `src/adapts/adapts.py` | high | `scaler_fit_call_trace.csv`; `ADAPTS.transform` uses `fit_transform` | Change prediction transform to use already-fitted `scaler.transform`; add regression test |
| B02 | LinearAE RevIN branch fails when `n_components != input_dim` | `src/adapts/adapters.py` | medium | `adapter_shape_consistency.csv` | reshape normalized input to `(-1, input_dim)`, not `(-1, n_components)` |
| B03 | DropoutLinearAE RevIN branch fails when `n_components != input_dim` | `src/adapts/adapters.py` | medium | `adapter_shape_consistency.csv` | same as B02 |

## 8. Protocol Risks, Not Necessarily Bugs

| ID | Risk | Impact | Recommendation |
|---|---|---|---|
| R01 | Prediction uses stochastic adapter mode for MSE/MAE | dropout/VAE metrics may vary and mix uncertainty sampling with deterministic accuracy | Add `prediction_mode` and report deterministic/stochastic metrics separately |
| R02 | VAE/linearVAE sample even in eval mode | deterministic evaluation is not currently available | Add mean-path inference for deterministic VAE metrics |
| R03 | Current fast setting uses ft=5 / adapter=30 and no hyperopt | named adapters may be under-trained | Do not judge final paper trend until protocol bugs are resolved and full/medium runs are tested |
| R04 | DataReader scaling + ADAPTS scaler + RevIN stacking | normalization interactions can dominate results | Add ablation for train-only scaler and RevIN on/off |

## 9. What Should Be Fixed Before Continuing Experiments

Must fix or verify first:

1. `src/adapts/adapts.py`: stop prediction-time scaler refit or explicitly split it into a named protocol.
2. `src/adapts/adapts.py`: add deterministic vs stochastic prediction mode.
3. `src/adapts/adapters.py`: fix LinearAE / DropoutLinearAE RevIN shape for `n_components != input_dim`.

Suggested but can be staged:

1. Add unit tests for scaler call counts in prediction.
2. Add unit tests for adapter shapes with `n_components in [2, 5, 7]`.
3. Add VAE deterministic transform path for metric evaluation.

Patch suggestion, not applied:

```text
results/stage9_targeted_code_verification/minimal_fix_plan.patch
```

## 10. Which Experiments Need Rerun After Fixes

After scaler prediction refit is fixed:

- Rerun a minimal ETTh1 H=192 seed=13 comparison:
  - baseline
  - strong-baseline
  - linearAE
  - dropoutLinearAE

After deterministic/stochastic prediction modes are separated:

- Rerun ETTh1 H=192 seed=13:
  - dropoutLinearAE deterministic vs stochastic
  - linearVAE deterministic vs stochastic
  - VAE deterministic vs stochastic

After LinearAE/DropoutLinearAE shape fix:

- Only then start latent dimension ablation with `n_components != input_dim`.

Weather / ExchangeRate:

- Should remain paused until scaler and prediction-mode behavior are fixed or explicitly documented as protocol choices.

