from __future__ import annotations

from pathlib import Path

import torch

from adapts.adapters import DropoutLinearAutoEncoder, betaLinearVAE, betaVAE


OUT_DIR = Path("results/stage9_targeted_code_verification")


def compare_twice(module, x, transform: bool = True) -> dict:
    with torch.no_grad():
        y1 = module.transform_torch(x) if transform else module(x)
        y2 = module.transform_torch(x) if transform else module(x)
    return {
        "same": bool(torch.allclose(y1, y2)),
        "max_abs_diff": float((y1 - y2).abs().max().item()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(13)
    input_dim = 7
    n_components = 7
    context_length = 512
    horizon = 96
    x = torch.randn(4 * context_length, input_dim)
    rows = []
    modules = [
        ("dropoutLinearAE", DropoutLinearAutoEncoder(input_dim, n_components, context_length, horizon, p=0.5, use_revin=False)),
        ("linearVAE", betaLinearVAE(input_dim, n_components, context_length, horizon, use_revin=False)),
        ("VAE", betaVAE(input_dim, n_components, context_length, horizon, use_revin=False)),
    ]
    for name, module in modules:
        module.train()
        train_result = compare_twice(module, x)
        module.eval()
        eval_result = compare_twice(module, x)
        rows.append((name, "train", train_result["same"], train_result["max_abs_diff"]))
        rows.append((name, "eval", eval_result["same"], eval_result["max_abs_diff"]))

    lines = [
        "# Adapter Prediction Mode Verification",
        "",
        "The current `ADAPTS.predict_multi_step` calls `self.adapter.base_projector_.train()` before prediction.",
        "This intentionally enables stochasticity for dropout, but it also means deterministic MSE/MAE evaluation and probabilistic sampling are mixed in one path.",
        "",
        "| Adapter | Mode | Two consecutive outputs identical | Max abs diff |",
        "|---|---|---:|---:|",
    ]
    for name, mode, same, diff in rows:
        lines.append(f"| {name} | {mode} | {same} | {diff:.8f} |")
    lines += [
        "",
        "Interpretation:",
        "- dropoutLinearAE is stochastic in train mode and deterministic in eval mode.",
        "- VAE / linearVAE remain stochastic in both train and eval because `reparameterize` samples unconditionally.",
        "- Current code uses stochastic prediction for MSE/MAE when `n_samples > 1`, then averages predictions.",
        "- Recommended design: add explicit `prediction_mode='deterministic'|'stochastic'` and `n_mc_samples`.",
    ]
    (OUT_DIR / "adapter_prediction_mode.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
