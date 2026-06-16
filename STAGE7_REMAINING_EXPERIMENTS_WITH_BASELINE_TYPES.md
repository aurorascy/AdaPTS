# Stage 7: Remaining Experiments With Baseline Types

## 1. Background

The baseline protocol audit confirmed that the old empty-adapter results from Stage 2 and Stage 3 are not pure MOMENT baselines. They use:

```text
adapter=None
use_revin=True
supervised=ft_then_supervised
ADAPTS scaler
fine-tuned MOMENT forecasting head
trainable JustRevIn / RevIN affine parameters
```

Those results are now treated as `strong-baseline`, not `baseline`.

## 2. Baseline Type Definitions

| Protocol type | Definition | Uses RevIN | Fine-tunes head | Trains JustRevIn / adapter | Notes |
|---|---|---|---|---|---|
| baseline | `adapter=None`, `use_revin=False`, `supervised=ft` | No | Yes | No | Head-only baseline supported by current CLI |
| strong-baseline | `adapter=None`, `use_revin=True`, `supervised=ft_then_supervised` | Yes | Yes | Yes | Old Stage 2 / Stage 3 empty-adapter protocol |
| adapter | named adapter + `use_revin=True` + `ft_then_supervised` | Yes | Yes | Yes | linearAE, dropoutLinearAE, linearVAE, VAE |
| PCA missing | PCA projector is not a `torch.nn.Module` in `ft_then_supervised` | N/A | N/A | N/A | Skipped and marked unsupported |

## 3. Data Preparation Status

| Dataset | Path | Shape | Feature count | Missing | DataReader status |
|---|---|---:|---:|---:|---|
| Illness | `external_data/forecasting/Illness/Illness.csv` | (966, 8) | 7 | 0 | Previously validated |
| ETTh1 | `external_data/forecasting/ETTh1/ETTh1.csv` | (17420, 8) | 7 | 0 | Previously validated |
| Weather | `external_data/forecasting/Weather/Weather.csv` | (52696, 22) | 21 | 0 | H=96 and H=192 validated |
| ExchangeRate | `external_data/forecasting/ExchangeRate/ExchangeRate.csv` | (7588, 9) | 8 | 0 | H=96 and H=192 validated |

Weather / ExchangeRate preparation is documented in `DATA_PREP_REMAINING_DATASETS.md`.

## 4. Experiment Matrix Status

| Dataset | H | baseline | strong-baseline | adapters | Status |
|---|---:|---|---|---|---|
| ETTh1 | 96 | N/A in old Stage 3 | Completed from old Stage 3 | linearAE/dropoutLinearAE/linearVAE/VAE completed from old Stage 3 | Imported and relabeled |
| Illness | 24 | N/A in old Stage 2 | Completed from old Stage 2 | linearAE/dropoutLinearAE/linearVAE/VAE completed from old Stage 2 | Imported and relabeled |
| ETTh1 | 192 | Completed, 3 seeds | Completed, 3 seeds | linearAE/dropoutLinearAE/linearVAE/VAE completed, 3 seeds | Completed in Stage 8 |
| Illness | 60 | Not yet run | Not yet run | Not yet run | Pending |
| Weather | 96 | Not yet run | Not yet run | Not yet run | Data ready, pending |
| Weather | 192 | Not yet run | Not yet run | Not yet run | Data ready, pending |
| ExchangeRate | 96 | Not yet run | Not yet run | Not yet run | Data ready, pending |
| ExchangeRate | 192 | Not yet run | Not yet run | Not yet run | Data ready, pending |

PCA was not run and is reported as `unsupported`.

## 5. Result Files

- Raw standardized results: `results/stage7_remaining_table1_with_baseline_types/raw_results.csv`
- Summary by protocol: `results/stage7_remaining_table1_with_baseline_types/summary_by_dataset_horizon_protocol.csv`
- Table-1-style summary: `results/stage7_remaining_table1_with_baseline_types/summary_table1_style.csv`
- Best adapter summary: `results/stage7_remaining_table1_with_baseline_types/best_by_task.csv`
- Failed runs: `results/stage7_remaining_table1_with_baseline_types/failed_runs.json`

## 6. Summary Results

| Dataset | Horizon | Protocol type | Adapter label | MSE mean | MSE std | MSE stderr | MAE mean | MAE std | MAE stderr | n |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | adapter | VAE | 0.405431 | 0.008243 | 0.004759 | 0.430110 | 0.006786 | 0.003918 | 3 |
| ETTh1 | 96 | adapter | dropoutLinearAE | 0.397490 | 0.002565 | 0.001481 | 0.418542 | 0.001915 | 0.001106 | 3 |
| ETTh1 | 96 | adapter | linearAE | 0.397805 | 0.005147 | 0.002971 | 0.418902 | 0.002709 | 0.001564 | 3 |
| ETTh1 | 96 | adapter | linearVAE | 0.408220 | 0.002350 | 0.001357 | 0.423988 | 0.002352 | 0.001358 | 3 |
| ETTh1 | 96 | strong-baseline | strong-baseline | 0.388832 | 0.000411 | 0.000237 | 0.409927 | 0.000360 | 0.000208 | 3 |
| ETTh1 | 192 | adapter | VAE | 0.434089 | 0.004169 | 0.002407 | 0.452990 | 0.003785 | 0.002185 | 3 |
| ETTh1 | 192 | adapter | dropoutLinearAE | 0.426939 | 0.003479 | 0.002009 | 0.437795 | 0.002678 | 0.001546 | 3 |
| ETTh1 | 192 | adapter | linearAE | 0.425188 | 0.003969 | 0.002291 | 0.437814 | 0.000859 | 0.000496 | 3 |
| ETTh1 | 192 | adapter | linearVAE | 0.438605 | 0.004037 | 0.002331 | 0.445862 | 0.001928 | 0.001113 | 3 |
| ETTh1 | 192 | baseline | baseline | 0.411224 | 0.000186 | 0.000107 | 0.423447 | 0.000173 | 0.000100 | 3 |
| ETTh1 | 192 | strong-baseline | strong-baseline | 0.413644 | 0.002213 | 0.001277 | 0.426067 | 0.002511 | 0.001450 | 3 |
| Illness | 24 | adapter | VAE | 2.848762 | 0.033796 | 0.019512 | 1.096714 | 0.006188 | 0.003572 | 3 |
| Illness | 24 | adapter | dropoutLinearAE | 2.895389 | 0.098752 | 0.057015 | 1.172557 | 0.028212 | 0.016288 | 3 |
| Illness | 24 | adapter | linearAE | 2.979297 | 0.067934 | 0.039222 | 1.187175 | 0.023663 | 0.013662 | 3 |
| Illness | 24 | adapter | linearVAE | 2.792917 | 0.149046 | 0.086052 | 1.133232 | 0.015069 | 0.008700 | 3 |
| Illness | 24 | strong-baseline | strong-baseline | 2.617666 | 0.017516 | 0.010113 | 1.112349 | 0.003738 | 0.002158 | 3 |

## 7. Table-1-Style Summary

| Dataset | H | baseline | strong-baseline | linearAE | dropoutLinearAE | linearVAE | VAE | PCA |
|---|---:|---|---|---|---|---|---|---|
| ETTh1 | 96 | N/A | 0.388832 +/- 0.000411 (n=3) | 0.397805 +/- 0.005147 (n=3) | 0.397490 +/- 0.002565 (n=3) | 0.408220 +/- 0.002350 (n=3) | 0.405431 +/- 0.008243 (n=3) | unsupported |
| ETTh1 | 192 | 0.411224 +/- 0.000186 (n=3) | 0.413644 +/- 0.002213 (n=3) | 0.425188 +/- 0.003969 (n=3) | 0.426939 +/- 0.003479 (n=3) | 0.438605 +/- 0.004037 (n=3) | 0.434089 +/- 0.004169 (n=3) | unsupported |
| Illness | 24 | N/A | 2.617666 +/- 0.017516 (n=3) | 2.979297 +/- 0.067934 (n=3) | 2.895389 +/- 0.098752 (n=3) | 2.792917 +/- 0.149046 (n=3) | 2.848762 +/- 0.033796 (n=3) | unsupported |

## 8. ETTh1 H=192 Completion Update

Stage 8 completed the ETTh1 H=192 matrix on top of the Stage 7 head-only baseline.

Experiment setting:

- Dataset: ETTh1
- Horizon: 192
- Seeds: 13, 42, 2024
- Context length: 512
- n_components: 7
- Epochs: fine-tuning=5, adapter=30
- PCA: not run, marked unsupported

Key comparison:

1. The true head-only baseline is strongest in this local fast run: MSE 0.411224, MAE 0.423447.
2. The strong-baseline is slightly weaker than the true baseline: MSE 0.413644, MAE 0.426067.
3. The best named adapter is linearAE: MSE 0.425188, MAE 0.437814.
4. linearAE is worse than baseline by about 3.40% MSE and worse than strong-baseline by about 2.79% MSE.
5. Under the current fast setting, ETTh1 H=192 does not support the simple claim that named adapters improve over the local baseline.

Paper ETTh1 H=192 reference MSE:

| Method | Paper MSE |
|---|---:|
| Moment baseline | 0.431 +/- 0.001 |
| PCA | 0.440 +/- 0.000 |
| LinearAE | 0.452 +/- 0.002 |
| dropoutLinearAE | 0.446 +/- 0.001 |
| LinearVAE | 0.448 +/- 0.002 |
| VAE | 0.431 +/- 0.001 |

The local baseline here is the head-only baseline, while strong-baseline is RevIN-assisted. These must remain separated in all future tables.

## 9. Relation to Paper Results

This is still not the full paper protocol because:

- Hyperopt was not run.
- The current runs use the fast epoch setting: `n_epochs_fine_tuning=5`, `n_epochs_adapter=30`.
- PCA is not included because the current supervised path does not support it cleanly.
- Baseline definitions have been split into `baseline` and `strong-baseline`.

## 10. Next Steps

1. Run Illness H=60 with the same separated protocols: baseline, strong-baseline, linearAE, dropoutLinearAE, linearVAE, VAE.
2. If Illness H=60 adapters remain weaker than baseline, audit adapter training epochs and hyperparameters before expanding to Weather / ExchangeRate.
3. Keep PCA as a separate protocol investigation instead of forcing it into `ft_then_supervised`.
4. After a few fast-matrix tasks are stable, consider medium/full epochs or hyperparameter optimization as a later phase.
