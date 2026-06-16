from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path("results/stage9_targeted_code_verification")
MODEL_NAME = "AutonLab/MOMENT-1-small"
HORIZON = 96


def classify(name: str) -> str:
    lower = name.lower()
    if "encoder" in lower:
        return "encoder"
    if "embed" in lower:
        return "embedder"
    if "head" in lower or "forecast" in lower:
        return "head"
    return "others"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    error = None
    try:
        from adapts.icl.moment import load_moment_model

        model = load_moment_model(MODEL_NAME, HORIZON)
        for name, param in model.named_parameters():
            rows.append(
                {
                    "name": name,
                    "shape": tuple(param.shape),
                    "numel": int(param.numel()),
                    "requires_grad": bool(param.requires_grad),
                    "module_group": classify(name),
                }
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "name": "__MODEL_LOAD_ERROR__",
                "shape": "",
                "numel": 0,
                "requires_grad": False,
                "module_group": "error",
                "error": error,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "moment_trainable_params.csv", index=False)
    trainable = df[df["requires_grad"] == True].copy()  # noqa: E712
    summary = trainable.groupby("module_group", dropna=False)["numel"].sum().reset_index(name="trainable_numel")
    total = int(trainable["numel"].sum()) if not trainable.empty else 0
    encoder_trainable = int(summary[summary.module_group.eq("encoder")]["trainable_numel"].sum()) if not summary.empty else 0
    embedder_trainable = int(summary[summary.module_group.eq("embedder")]["trainable_numel"].sum()) if not summary.empty else 0
    head_trainable = int(summary[summary.module_group.eq("head")]["trainable_numel"].sum()) if not summary.empty else 0
    lines = [
        "# MOMENT Trainable Parameter Verification",
        "",
        f"- model: `{MODEL_NAME}`",
        f"- horizon: `{HORIZON}`",
    ]
    if error:
        lines += ["", f"Model load failed: `{error}`"]
    else:
        lines += [
            "",
            f"- total trainable params: `{total}`",
            f"- encoder trainable params: `{encoder_trainable}`",
            f"- embedder trainable params: `{embedder_trainable}`",
            f"- head trainable params: `{head_trainable}`",
            "",
            "Interpretation:",
            f"- encoder frozen: `{encoder_trainable == 0}`",
            f"- embedder frozen: `{embedder_trainable == 0}`",
            f"- head trainable: `{head_trainable > 0}`",
            "- The optimizer receives `self.model.parameters()`, but frozen parameters should not update when `requires_grad=False`.",
            "",
            "Trainable parameter rows are saved to `moment_trainable_params.csv`.",
        ]
    (OUT_DIR / "moment_trainable_params.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
