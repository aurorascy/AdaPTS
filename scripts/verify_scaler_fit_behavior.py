from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from adapts.adapts import ADAPTS
from adapts.adapters import MultichannelProjector


OUT_DIR = Path("results/stage9_targeted_code_verification")


@dataclass
class DummyICLObject:
    mean_arr: np.ndarray | None = None
    mode_arr: np.ndarray | None = None
    sigma_arr: np.ndarray | None = None


class DummyICLTrainer:
    def __init__(self, n_features: int, horizon: int):
        self.n_features = n_features
        self.horizon = horizon
        self.last_context = None

    def update_context(self, time_series, context_length=None):
        self.last_context = time_series

    def predict_long_horizon(self, prediction_horizon: int, **kwargs):
        batch = self.last_context.shape[0]
        objects = []
        for _ in range(self.n_features):
            arr = np.zeros((batch, 1, prediction_horizon), dtype=np.float32)
            objects.append(DummyICLObject(mean_arr=arr, mode_arr=arr, sigma_arr=np.zeros_like(arr)))
        return objects

    def eval(self):
        return self


class CountingScaler:
    def __init__(self):
        self.calls = []
        self.fit_count = 0
        self.fit_transform_count = 0
        self.transform_count = 0
        self.inverse_transform_count = 0
        self.mean_ = None
        self.std_ = None

    def _record(self, method: str, x):
        self.calls.append(
            {
                "step": len(self.calls),
                "method": method,
                "shape": tuple(x.shape),
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
            }
        )

    def fit(self, x):
        self.fit_count += 1
        self._record("fit", x)
        self.mean_ = np.mean(x, axis=(0, 2), keepdims=True)
        self.std_ = np.std(x, axis=(0, 2), keepdims=True) + 1e-6
        return self

    def transform(self, x):
        self.transform_count += 1
        self._record("transform", x)
        if self.mean_ is None:
            raise RuntimeError("transform called before fit")
        return (x - self.mean_) / self.std_

    def fit_transform(self, x):
        self.fit_transform_count += 1
        self._record("fit_transform", x)
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        self.inverse_transform_count += 1
        self._record("inverse_transform", x)
        if self.mean_ is None:
            raise RuntimeError("inverse_transform called before fit")
        return x * self.std_ + self.mean_


def build_model(n_features: int = 2, horizon: int = 3) -> tuple[ADAPTS, CountingScaler]:
    adapter = MultichannelProjector(
        num_channels=n_features,
        new_num_channels=n_features,
        base_projector=None,
        device="cpu",
        use_revin=False,
        context_length=5,
        forecast_horizon=horizon,
    )
    model = ADAPTS(
        adapter=adapter,
        iclearner=DummyICLTrainer(n_features=n_features, horizon=horizon),
        n_features=n_features,
        n_components=n_features,
    )
    scaler = CountingScaler()
    model.scaler = scaler
    return model, scaler


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(13)
    n_features = 2
    horizon = 3
    context = 5
    x_context = rng.normal(loc=100.0, scale=10.0, size=(4, n_features, context)).astype("float32")
    y_future = rng.normal(loc=-50.0, scale=5.0, size=(4, n_features, horizon)).astype("float32")
    time_series = np.concatenate([x_context, y_future], axis=-1)

    model, scaler = build_model(n_features=n_features, horizon=horizon)
    model.transform(x_context)
    for call in scaler.calls:
        call["phase"] = "direct_transform"

    model2, scaler2 = build_model(n_features=n_features, horizon=horizon)
    model2.predict_multi_step(time_series, prediction_horizon=horizon, n_samples=2, batch_size=2)
    trace = scaler.calls + [{**call, "phase": "predict_multi_step"} for call in scaler2.calls]
    trace_df = pd.DataFrame(trace)
    trace_df.to_csv(OUT_DIR / "scaler_fit_call_trace.csv", index=False)

    predict_fit = trace_df[(trace_df["phase"] == "predict_multi_step") & (trace_df["method"].isin(["fit", "fit_transform"]))]
    future_mean = float(np.mean(y_future))
    context_mean = float(np.mean(x_context))
    report = [
        "# Scaler Fit Behavior Verification",
        "",
        "Synthetic data used distinct distributions for context and future target:",
        f"- context mean: {context_mean:.6f}",
        f"- future y mean: {future_mean:.6f}",
        "",
        "Findings:",
        f"- Direct `ADAPTS.transform(X_context)` calls fit-related scaler methods: {not trace_df[(trace_df.phase == 'direct_transform') & (trace_df.method.isin(['fit', 'fit_transform']))].empty}",
        f"- `ADAPTS.predict_multi_step(time_series)` calls fit-related scaler methods during prediction: {not predict_fit.empty}",
        "- `predict_multi_step` passes only `X[:, :, :-prediction_horizon]` into `transform`, so the fit uses test context X, not future test y.",
        "- This is test-context refit, not direct test-y leakage.",
        "- It can still affect fairness and protocol interpretation because the scaler state used for inference is not the scaler fitted during training.",
        "",
        "Call trace saved to `results/stage9_targeted_code_verification/scaler_fit_call_trace.csv`.",
    ]
    (OUT_DIR / "scaler_fit_behavior.md").write_text("\n".join(report), encoding="utf-8")
    print(trace_df.to_string(index=False))
    print(f"prediction fit calls: {len(predict_fit)}")


if __name__ == "__main__":
    main()
