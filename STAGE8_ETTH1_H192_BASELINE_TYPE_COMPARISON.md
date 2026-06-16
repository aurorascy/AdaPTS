# Stage 8: ETTh1 H=192 Baseline Type Comparison

## 1. Background

Stage 7 established that `baseline` and `strong-baseline` must be kept separate. The old empty-adapter runs with `use_revin=True` and `supervised=ft_then_supervised` are not true baselines; they are RevIN-assisted strong-baselines. This stage completed ETTh1 H=192 by adding the strong-baseline and four named adapters on top of the existing Stage 7 head-only baseline.

## 2. Baseline Type Definitions

| Type | Definition |
|---|---|
| baseline | head-only no RevIN baseline: `adapter=None`, `use_revin=False`, `supervised=ft` |
| strong-baseline | `adapter=None`, `use_revin=True`, `supervised=ft_then_supervised`; trains JustRevIn / RevIN affine parameters |
| adapter | named adapter + `use_revin=True` + `supervised=ft_then_supervised` |

PCA was not run because the PCA projector is not a `torch.nn.Module` under the current `ft_then_supervised` path.

## 3. Experiment Setting

| Item | Value |
|---|---|
| Dataset | ETTh1 |
| Horizon | 192 |
| Context length | 512 |
| Model | AutonLab/MOMENT-1-small |
| Seeds | 13, 42, 2024 |
| n_components | 7 |
| Epochs | fine-tuning=5, adapter=30 |
| Device | cuda:0 |
| Results path | `results/stage7_remaining_table1_with_baseline_types/raw_results.csv` |

## 4. Results

| Protocol type | Adapter label | MSE mean | MSE std | MSE stderr | MAE mean | MAE std | MAE stderr | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | baseline | 0.411224 | 0.000186 | 0.000107 | 0.423447 | 0.000173 | 0.000100 | 3 |
| strong-baseline | strong-baseline | 0.413644 | 0.002213 | 0.001277 | 0.426067 | 0.002511 | 0.001450 | 3 |
| adapter | linearAE | 0.425188 | 0.003969 | 0.002291 | 0.437814 | 0.000859 | 0.000496 | 3 |
| adapter | dropoutLinearAE | 0.426939 | 0.003479 | 0.002009 | 0.437795 | 0.002678 | 0.001546 | 3 |
| adapter | linearVAE | 0.438605 | 0.004037 | 0.002331 | 0.445862 | 0.001928 | 0.001113 | 3 |
| adapter | VAE | 0.434089 | 0.004169 | 0.002407 | 0.452990 | 0.003785 | 0.002185 | 3 |

Per-seed MSE:

| Protocol | Adapter | Seed 13 | Seed 42 | Seed 2024 |
|---|---|---:|---:|---:|
| baseline | baseline | 0.411176 | 0.411067 | 0.411429 |
| strong-baseline | strong-baseline | 0.412449 | 0.412286 | 0.416197 |
| adapter | linearAE | 0.428690 | 0.425995 | 0.420878 |
| adapter | dropoutLinearAE | 0.430847 | 0.425792 | 0.424178 |
| adapter | linearVAE | 0.439640 | 0.442023 | 0.434151 |
| adapter | VAE | 0.430607 | 0.438709 | 0.432953 |

## 5. Key Conclusions

1. The true baseline MSE is **0.411224** and MAE is **0.423447**.
2. The strong-baseline MSE is **0.413644** and MAE is **0.426067**.
3. The best named adapter is **linearAE**, with MSE **0.425188** and MAE **0.437814**.
4. The best named adapter does not beat the true baseline. It is about **3.40% worse** in MSE than baseline.
5. The best named adapter also does not beat the strong-baseline. It is about **2.79% worse** in MSE than strong-baseline.
6. Under the fast setting, ETTh1 H=192 does not support the trend that adapters improve over the local baseline. The strongest local protocol is the head-only baseline.
7. This stage supports continuing to Illness H=60, but future reports must keep baseline and strong-baseline separate.

## 6. Relation to Paper Reference

Paper ETTh1 H=192 reference MSE:

| Method | Paper MSE |
|---|---:|
| Moment baseline | 0.431 +/- 0.001 |
| PCA | 0.440 +/- 0.000 |
| LinearAE | 0.452 +/- 0.002 |
| dropoutLinearAE | 0.446 +/- 0.001 |
| LinearVAE | 0.448 +/- 0.002 |
| VAE | 0.431 +/- 0.001 |

The local fast-run values are not directly comparable to the full paper protocol because hyperopt was not run, PCA is not included, and training uses the fast epoch setting.

## 7. Next Steps

1. Run Illness H=60 with the same separated protocols: baseline, strong-baseline, linearAE, dropoutLinearAE, linearVAE, VAE.
2. If Illness H=60 adapters remain weaker than baseline, audit adapter training length and hyperparameters before expanding to Weather / ExchangeRate.
3. Keep PCA as a separate protocol investigation rather than forcing it into `ft_then_supervised`.
