from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from adapts.adapters import DropoutLinearAutoEncoder, LinearAutoEncoder, betaLinearVAE, betaVAE


OUT_DIR = Path("results/stage9_targeted_code_verification")
ADAPTERS = {
    "linearAE": LinearAutoEncoder,
    "dropoutLinearAE": DropoutLinearAutoEncoder,
    "linearVAE": betaLinearVAE,
    "VAE": betaVAE,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    batch = 4
    seq_len = 512
    input_dim = 7
    horizon = 96
    x = torch.randn(batch * seq_len, input_dim)
    rows = []
    for adapter_name, cls in ADAPTERS.items():
        for n_components in [2, 5, 7]:
            for use_revin in [False, True]:
                try:
                    module = cls(
                        input_dim=input_dim,
                        n_components=n_components,
                        context_length=seq_len,
                        forecast_horizon=horizon,
                        use_revin=use_revin,
                    )
                    module.eval()
                    with torch.no_grad():
                        y = module(x)
                    ok = tuple(y.shape) == tuple(x.shape)
                    rows.append(
                        {
                            "adapter": adapter_name,
                            "n_components": n_components,
                            "input_dim": input_dim,
                            "use_revin": use_revin,
                            "forward_ok": True,
                            "output_shape": tuple(y.shape),
                            "restores_input_dim": ok,
                            "error": "",
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "adapter": adapter_name,
                            "n_components": n_components,
                            "input_dim": input_dim,
                            "use_revin": use_revin,
                            "forward_ok": False,
                            "output_shape": "",
                            "restores_input_dim": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "adapter_shape_consistency.csv", index=False)
    failures = df[~df["forward_ok"] | ~df["restores_input_dim"]]
    lines = [
        "# Adapter Shape Consistency Verification",
        "",
        "Synthetic input shape: `(batch * seq_len, input_dim) = (2048, 7)`.",
        "",
        f"- total cases: {len(df)}",
        f"- failing cases: {len(failures)}",
        "",
    ]
    if not failures.empty:
        lines += [
            "## Failures",
            "",
            "| Adapter | n_components | use_revin | Error |",
            "|---|---:|---:|---|",
        ]
        for _, row in failures.iterrows():
            lines.append(f"| {row.adapter} | {row.n_components} | {row.use_revin} | {row.error} |")
    lines += [
        "",
        "Interpretation:",
        "- If LinearAE / DropoutLinearAE fail when `use_revin=True` and `n_components != input_dim`, this is a confirmed shape bug for latent-dimension ablations.",
        "- Current ETTh1/Illness/Stage7 runs use `n_components == input_dim`, so this does not explain the current main results.",
        "",
        "Minimal fix suggestion:",
        "- In LinearAE and DropoutLinearAE RevIN branches, reshape normalized input back to `(-1, input_dim)` before feeding the encoder, not `(-1, n_components)`.",
        "- Add unit tests for `n_components in [2, 5, 7]` with `use_revin=True/False`.",
    ]
    (OUT_DIR / "adapter_shape_consistency.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
