# AdaPTS Environment Setup

Date: 2026-06-10

## System Check

- OS: Microsoft Windows 11 Pro 10.0.26200, 64-bit
- Shell: PowerShell
- Repository: `F:\AdaPTS`
- Git: available, `origin` points to `https://github.com/abenechehab/AdaPTS.git`
- Initial git status: clean on `main`, up to date with `origin/main`
- Python on host: 3.10.11
- Conda: not available in this PowerShell session
- Environment used: Python venv at `.venv-adapts`
- GPU: NVIDIA GeForce RTX 3060, 12 GB VRAM
- NVIDIA driver: 591.86; `nvidia-smi` reports CUDA Version 13.1
- PyTorch CUDA in this environment: available after replacing CPU torch with CUDA torch

## Environment Creation

Conda was preferred but `conda` was not found, so a venv was created:

```powershell
python -m venv .venv-adapts
. .\.venv-adapts\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Direct PyPI access hit TLS EOF errors. Re-running with the Tsinghua PyPI mirror succeeded:

```powershell
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip setuptools wheel
```

Environment verification:

- Python: 3.10.11
- Python path: `F:\AdaPTS\.venv-adapts\Scripts\python.exe`
- pip: 26.1.2

## Dependency Installation

Project development dependencies installed successfully:

```powershell
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[dev]"
```

The requested GitHub install for MOMENT failed because this machine could not connect to `github.com:443`:

```powershell
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple git+https://github.com/moment-timeseries-foundation-model/moment.git
```

Fallback used:

```powershell
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple momentfm
```

This installed `momentfm 0.1.4`, the package imported by the AdaPTS code. `pip check` reported no broken requirements.

Key installed package versions:

- AdaPTS: `0.1.dev370+g8bf57c7ee`, editable from `F:\AdaPTS`
- momentfm: 0.1.4
- torch: 2.12.0+cu126
- numpy: 1.25.2
- pandas: 2.3.3
- scikit-learn: 1.7.2
- transformers: 4.33.3
- huggingface-hub: 0.24.0
- pytest: 9.0.3

CUDA check after CUDA torch installation:

```text
torch: 2.12.0+cu126
torch cuda: 12.6
cuda available: True
gpu: NVIDIA GeForce RTX 3060
device count: 1
```

CUDA torch installation command:

```powershell
python -m pip install --force-reinstall --no-deps torch==2.12.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

This replaced only the existing CPU torch wheel and left the rest of the dependency set unchanged. `pip check` still reported no broken requirements.

## Code Changes

Changed `src/adapts/utils/main_script.py`:

- Replaced the author's hard-coded absolute data path `/mnt/data_2/abenechehab/AdaPTS/external_data/`.
- New path is derived from the repository root:

```python
repo_root = Path(__file__).resolve().parents[3]
data_path=str(repo_root / "external_data") + "/"
```

Changed `src/adapts/icl/moment.py`:

- Set `local_files_only=False` in `MOMENTPipeline.from_pretrained(...)` so the MOMENT model can be downloaded on first use.

Changed `.gitignore`:

- Added `.venv-adapts/` so the local virtual environment is not accidentally committed.

Syntax checks passed:

```powershell
python -m py_compile src\adapts\utils\main_script.py
python -m py_compile src\adapts\icl\moment.py
```

## Data Check

Illness data exists:

```text
external_data/forecasting/Illness/Illness.csv
shape: (966, 8)
```

First rows loaded successfully with pandas.

## Unit Tests

Command:

```powershell
pytest tests/ -v
```

Result:

```text
46 passed in 4.19s
```

## Import Checks

AdaPTS import:

```text
adapts import ok
adapters import ok
```

MOMENT import:

```text
MOMENT import ok
```

The MOMENT import emitted a `transformers` FutureWarning about `torch.utils._pytree`, but it did not block execution.

## CPU Smoke Test

The first smoke test used CPU before CUDA torch was installed.

Command:

```powershell
python scripts\run.py `
  --forecast-horizon 24 `
  --model-name "AutonLab/MOMENT-1-small" `
  --context-length 512 `
  --seed 13 `
  --device "cpu" `
  --dataset-name "Illness" `
  --adapter "linearVAE" `
  --use-revin `
  --supervised "ft_then_supervised" `
  --n-epochs-fine-tuning 1 `
  --n-epochs-adapter 1 `
  --data-path "results/smoke_test_illness.csv" `
  --log-dir "logs/smoke_test"
```

Result: success.

Notes:

- MOMENT-small downloaded and loaded.
- Fine-tuning ran for 1 epoch.
- Adapter training ran for 1 epoch.
- Multi-step prediction and metrics completed.
- Runtime was about 381 seconds on CPU.
- Windows HuggingFace cache emitted a symlink warning; caching still worked.
- Torch AMP emitted warnings because CUDA is unavailable; execution continued on CPU.

Result CSV:

```text
results/smoke_test_illness.csv
```

Metrics written:

| metric | value |
| --- | ---: |
| mse | 5.062546 |
| mae | 1.547317 |
| scaled_mse | 3.834413 |
| scaled_mae | 1.415393 |
| ks | 0.653655 |
| ece | 0.304911 |

## CUDA Smoke Test

After installing `torch 2.12.0+cu126`, a CUDA smoke test was run with the same Illness + MOMENT-small setup and `--device "cuda:0"`.

Command:

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
  --n-epochs-fine-tuning 1 `
  --n-epochs-adapter 1 `
  --data-path "results/smoke_test_illness_cuda.csv" `
  --log-dir "logs/smoke_test_cuda"
```

Result: success.

Notes:

- MOMENT-small loaded from the HuggingFace cache.
- Fine-tuning ran for 1 epoch on GPU.
- Adapter training ran for 1 epoch on GPU.
- Multi-step prediction and metrics completed on GPU.
- Runtime was about 27.7 seconds.
- Log reported GPU 0 memory stats: reserved 35.06%, allocated 1.36%.

CUDA result CSV:

```text
results/smoke_test_illness_cuda.csv
```

CUDA metrics written:

| metric | value |
| --- | ---: |
| mse | 5.062006 |
| mae | 1.547299 |
| scaled_mse | 3.834197 |
| scaled_mae | 1.415390 |
| ks | 0.654867 |
| ece | 0.304962 |

CUDA log directory:

```text
logs/smoke_test_cuda/Illness_pred=24/20260610_190445_Illness_pred=24_linearVAE_MOMENT-1-small/
```

Log directory:

```text
logs/smoke_test/Illness_pred=24/20260610_183939_Illness_pred=24_linearVAE_MOMENT-1-small/
```

Important generated files:

- `config.json`
- `run_20260610_183939.log`
- `n_comp_7/adapter.pt`
- `n_comp_7/ks.npy`
- `n_comp_7/ece.npy`
- TensorBoard event file

## Current Issues

- `conda` is not available in this PowerShell session, so venv was used.
- Direct PyPI access had TLS EOF failures; the Tsinghua PyPI mirror worked.
- GitHub clone for MOMENT failed due inability to connect to `github.com:443`; `momentfm 0.1.4` from PyPI was used instead.
- CUDA is now available through `torch 2.12.0+cu126`.
- The earlier CPU smoke test succeeded but was slow; the CUDA smoke test also succeeded and completed much faster.

## Next Steps

1. Keep the current CUDA smoke test as a sanity check before attempting larger experiments.
2. Use `--device "cuda:0"` for future MOMENT-small runs on this machine.
3. Avoid hyperopt and full paper reproduction until a slightly larger single-dataset run is confirmed stable.
