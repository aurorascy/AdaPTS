# AdaPTS Baseline Protocol Audit

## 1. Audit scope

This audit checks why the local baseline is stronger than the paper baseline and why it beats adapters on both Illness H=24 and ETTh1 H=96. No training, hyperopt, or new experiment was run. The audit only inspected existing code, configs, logs, TensorBoard-derived summaries, and result CSV files.

## 2. Main conclusion

The current local "baseline" is not pure zero-shot MOMENT. It is also not a bare no-adapter inference path.

In the runs audited here, `adapter=None` with `--use_revin=True` enters the full ADAPTS wrapper, creates a `JustRevIn` module, wraps it in `MultichannelProjector`, fine-tunes the MOMENT linear forecasting head, and then enters `adapter_supervised_fine_tuning`. Therefore the current baseline is best described as:

> MOMENT-small + ADAPTS preprocessing scaler + fine-tuned linear forecasting head + trainable RevIN/JustRevIn normalization path.

This is a strong baseline and can reasonably outperform the paper's reported baseline if the paper baseline used a weaker or differently defined protocol.

## 3. Code path for `adapter=None`

### 3.1 `scripts/run.py`

When `args.adapter` is empty and `args.use_revin=True`, `scripts/run.py` replaces the missing adapter with `JustRevIn`:

```python
if not args.adapter and args.use_revin:
    adapter = JustRevIn(...)
else:
    adapter = args.adapter
```

Immediately afterward, all cases are wrapped in:

```python
adapter = adapters.MultichannelProjector(
    num_channels=n_features,
    new_num_channels=n_components,
    base_projector=adapter,
    ...
)
```

So the baseline command does not pass `--adapter`, but the effective `base_projector_` is not `None`; it is a `JustRevIn` `torch.nn.Module`.

For `supervised="ft_then_supervised"`, the code then runs:

```python
adapts_model.fine_tune_iclearner(..., use_adapter=False, ...)
logger.info("Done fine tuning, now training adapter")
adapts_model.adapter_supervised_fine_tuning(...)
```

This confirms that baseline first fine-tunes the MOMENT forecasting head without adapter transforms, then trains the effective adapter path.

### 3.2 `src/adapts/adapts.py`

The `ADAPTS` constructor uses preprocessing by default. With `pca_in_preprocessing=False` and `scaler_in_preprocessing=True`, the scaler is:

```python
AxisScaler(MinMaxScaler(), axis=1)
AxisScaler(StandardScaler(), axis=1)
```

During `fine_tune_iclearner(use_adapter=False)`, the direct and inverse transforms are identity, but the scaler is still fitted and applied:

```python
X = self.scaler.fit_transform(X)
y = self.scaler.transform(y)
```

During `adapter_supervised_fine_tuning`, the code asserts that the effective base projector is a PyTorch module:

```python
assert isinstance(self.adapter.base_projector_, torch.nn.Module)
optimizer = torch.optim.Adam(self.adapter.base_projector_.parameters(), lr=learning_rate)
```

For the current baseline this succeeds because `JustRevIn` is a `torch.nn.Module`.

### 3.3 `src/adapts/adapters.py`

`MultichannelProjector(base_projector=None)` would create an `IdentityTransformer`, but this is not the audited baseline path because `run.py` passes `JustRevIn` when `--use_revin=True`.

`JustRevIn` contains:

```python
self.revin = RevIN(num_features=num_features)
```

`RevIN` uses affine trainable parameters by default:

```python
self.affine_weight = nn.Parameter(torch.ones(self.num_features))
self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
```

Thus the baseline has trainable normalization parameters in the supervised adapter-training phase.

## 4. Baseline config audit

Existing baseline `config.json` files show the same protocol for Illness H=24 and ETTh1 H=96:

| Stage | Dataset | Horizon | Seeds | adapter in config | use_revin | supervised | fine-tune epochs | adapter epochs | pca_in_preprocessing | effective n_components |
|---|---|---:|---|---|---|---|---:|---:|---|---:|
| Stage 2 | Illness | 24 | 13, 42, 2024 | `null` | True | ft_then_supervised | 5 | 30 | False | 7 |
| Stage 3 | ETTh1 | 96 | 13, 42, 2024 | `null` | True | ft_then_supervised | 5 | 30 | False | 7 |

Notes:

- The raw `config.json` stores `adapter: null` and `n_components: null`.
- The effective selected component count is visible in the log path and audit table as `n_comp_7` / `n_components=7`.
- Model checkpoint in config is `AutonLab/MOMENT-1-small`.
- The config files do not record a precise installed MOMENT package version; that should be added in future runs for traceability.

Audit source: `results/debug_audit/config_audit_table.csv`.

## 5. Baseline log audit

All baseline logs inspected contain:

```text
Starting fitting adapter: None
Done fine tuning, now training adapter
adapter fitted (supervised:ft_then_supervised)
```

TensorBoard-derived audit also found adapter-training scalar events for the baseline:

| Stage | Dataset | Seed | adapter train loss points | train loss decreased | val loss decreased |
|---|---|---:|---:|---|---|
| Stage 2 | Illness | 13 | 30 | False | True |
| Stage 2 | Illness | 42 | 30 | True | True |
| Stage 2 | Illness | 2024 | 25 | True | True |
| Stage 3 | ETTh1 | 13 | 16 | False | False |
| Stage 3 | ETTh1 | 42 | 17 | False | True |
| Stage 3 | ETTh1 | 2024 | 17 | False | False |

This means the baseline does enter an adapter-training stage, but the trained object is the `JustRevIn` / RevIN path, not linearAE, VAE, or another named adapter.

No `adapter.pt` files were found under baseline log directories. This is expected from the save condition:

```python
if args.adapter and args.adapter not in ["pca"]:
    torch.save(...)
```

Since `args.adapter` is `None`, the baseline trained RevIN module is not saved as `adapter.pt`, even though supervised adapter-stage training occurred.

Audit source: `results/debug_audit/training_log_audit_table.csv`.

## 6. Three baseline concepts

| Baseline concept | Meaning | Is this the current local baseline? |
|---|---|---|
| A. Pure zero-shot MOMENT | Load MOMENT and evaluate without fine-tuning, adapter, RevIN, or ADAPTS scaler | No |
| B. MOMENT + linear head fine-tuning | Fine-tune forecasting head, but no adapter or RevIN training | No, only partially |
| C. MOMENT + linear head fine-tuning + RevIN/scaler | Fine-tuned MOMENT with normalization/scaling assistance | Closest |

The current baseline is closest to C, but slightly stronger/more specific:

> MOMENT + linear head fine-tuning + ADAPTS scaler + supervised training of `JustRevIn` affine parameters.

## 7. Is the current baseline an AdaPTS adapter result?

Yes and no:

- It is an ADAPTS pipeline result because it creates an `ADAPTS` object, uses the ADAPTS scaler, and calls `adapter_supervised_fine_tuning`.
- It is a no named-adapter result because `args.adapter` is `None`, and no linearAE/VAE/PCA adapter is selected.
- It is not pure no-adapter MOMENT because `--use_revin=True` creates a trainable `JustRevIn` base projector.
- It contains RevIN.
- It contains fine-tuned MOMENT forecasting head.
- It enters the supervised adapter-training phase, where the trainable object is `JustRevIn` / RevIN affine parameters.

## 8. Why the local baseline may be stronger than the paper baseline

The local baseline is stronger than the paper baseline in both audited datasets:

| Dataset | Horizon | Local baseline MSE | Paper baseline MSE | Relative difference |
|---|---:|---:|---:|---:|
| Illness | 24 | 2.6177 | 2.902 | -9.80% |
| ETTh1 | 96 | 0.3888 | 0.411 | -5.39% |

Possible reasons, ranked:

1. **Baseline protocol mismatch.** The local baseline includes fine-tuned MOMENT head, ADAPTS scaler, RevIN, and supervised training of `JustRevIn`. If the paper baseline is plain MOMENT or only head fine-tuning without this RevIN path, the local baseline is stronger.
2. **RevIN/scaler advantage.** Both datasets use `use_revin=True` and ADAPTS internal scaling. This may substantially help baseline while the paper's baseline definition may not include the same normalization.
3. **MOMENT package/checkpoint/version differences.** The config records `AutonLab/MOMENT-1-small`, but not the exact installed package commit/version. A newer MOMENT implementation or checkpoint behavior can shift results.
4. **Metric/split/preprocessing differences.** Existing summary uses unscaled MSE/MAE columns, but data split, scaler fitting scope, or inverse-transform details can still differ from the paper protocol.
5. **Adapter under-training relative to paper.** Current stage uses 5 head fine-tuning epochs and 30 adapter epochs, while closer paper-style reruns proposed 50 and 300. Named adapters may not have converged enough to beat the strengthened baseline.
6. **Hyperopt not used.** The paper uses hyperparameter optimization; current adapter runs use fixed/default settings.

## 9. Relation to PCA failure

PCA fails in the current `ft_then_supervised` protocol because `adapter_supervised_fine_tuning` requires:

```python
self.adapter.base_projector_ isinstance torch.nn.Module
```

PCA creates an sklearn `PCA` object, not a PyTorch module. Therefore it cannot be trained with the supervised adapter fine-tuning loop. The likely correct PCA protocol is one of:

- use PCA as a preprocessing/fixed transform path, not supervised adapter fine-tuning;
- use `pca_in_preprocessing`;
- use a non-supervised fitting path such as `supervised=False` or `ft`, depending on the intended paper protocol.

This does not explain the strong baseline directly, but it confirms that adapter protocols are not all equivalent under `ft_then_supervised`.

## 10. Is there a result summary error?

No clear summary-metric selection error was found from the previous audit.

The summary files report the unscaled `mse`/`mae` metrics, not `scaled_mse`, `ks`, or `ece`:

- `results/stage2_illness_h24/summary_mean_std_stderr.csv`
- `results/stage3_etth1_h96/summary_mean_std_stderr.csv`

The anomaly is therefore more likely protocol-related than a simple metric aggregation mistake.

## 11. Recommended fair baselines for next checks

Do not compare only one baseline. The next audit reruns should explicitly separate:

1. **Pure zero-shot MOMENT**
   - no adapter;
   - no head fine-tuning;
   - no RevIN;
   - ideally no ADAPTS scaler unless paper baseline includes it.

2. **MOMENT + head fine-tuning only**
   - no named adapter;
   - `use_revin=False`;
   - fine-tune forecasting head only;
   - avoid `adapter_supervised_fine_tuning` if the effective adapter is identity.

3. **MOMENT + head fine-tuning + ADAPTS scaler**
   - no named adapter;
   - no RevIN;
   - keep ADAPTS scaler if this matches the paper's preprocessing.

4. **Current strong local baseline**
   - no named adapter;
   - `use_revin=True`;
   - `supervised=ft_then_supervised`;
   - train `JustRevIn` after head fine-tuning.

5. **Named adapters under the same preprocessing**
   - compare dropoutLinearAE / linearVAE / VAE against the matching baseline;
   - use paper-like epochs and fixed `n_components=7` before drawing conclusions.

## 12. Minimal next rerun plan, not executed

The most useful next small rerun is to isolate ETTh1 H=96 seed=13 with the paper-like component count and longer training:

```bash
# Baseline variant 1: current strong baseline, documented explicitly
python scripts/run.py \
  --forecast_horizon 96 \
  --model_name "AutonLab/MOMENT-1-small" \
  --context_length 512 \
  --seed 13 \
  --device "cuda:0" \
  --dataset_name "ETTh1" \
  --use_revin \
  --supervised "ft_then_supervised" \
  --n_epochs_fine_tuning 50 \
  --n_epochs_adapter 300 \
  --n_components 7 \
  --data_path "results/debug_rerun_plan/etth1_h96_seed13_protocol_audit.csv" \
  --log_dir "logs/debug_rerun_plan/etth1_h96_seed13/current_strong_baseline"

# Adapter rerun: dropoutLinearAE under same protocol
python scripts/run.py \
  --forecast_horizon 96 \
  --model_name "AutonLab/MOMENT-1-small" \
  --context_length 512 \
  --seed 13 \
  --device "cuda:0" \
  --dataset_name "ETTh1" \
  --adapter "dropoutLinearAE" \
  --use_revin \
  --supervised "ft_then_supervised" \
  --n_epochs_fine_tuning 50 \
  --n_epochs_adapter 300 \
  --n_components 7 \
  --data_path "results/debug_rerun_plan/etth1_h96_seed13_protocol_audit.csv" \
  --log_dir "logs/debug_rerun_plan/etth1_h96_seed13/dropoutLinearAE"
```

Additionally, one should define a truly weaker pure/head-only baseline before claiming adapter failure:

```bash
# Head-only baseline candidate; verify CLI support and protocol before running.
python scripts/run.py \
  --forecast_horizon 96 \
  --model_name "AutonLab/MOMENT-1-small" \
  --context_length 512 \
  --seed 13 \
  --device "cuda:0" \
  --dataset_name "ETTh1" \
  --supervised "ft" \
  --n_epochs_fine_tuning 50 \
  --n_epochs_adapter 0 \
  --n_components 7 \
  --data_path "results/debug_rerun_plan/etth1_h96_seed13_protocol_audit.csv" \
  --log_dir "logs/debug_rerun_plan/etth1_h96_seed13/head_only_no_revin"
```

Before running this candidate, confirm how the CLI represents disabling `use_revin` and whether `n_epochs_adapter=0` is accepted.

## 13. Bottom line

The current local baseline should be treated as a strong ADAPTS-normalized, RevIN-assisted, fine-tuned baseline, not as pure MOMENT. This is the most likely reason it beats adapters and is stronger than the paper baseline. The next fair comparison should first establish multiple baseline definitions, then rerun a minimal ETTh1 H=96 seed=13 pair with `n_components=7` and paper-like training epochs.
