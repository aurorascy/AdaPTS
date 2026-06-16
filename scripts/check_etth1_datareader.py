from __future__ import annotations

from pathlib import Path

from adapts.utils.data_readers import DataReader


def shape_of(value: object) -> tuple[int, ...] | str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return type(value).__name__
    return tuple(int(dim) for dim in shape)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "external_data"
    target = data_root / "forecasting" / "ETTh1" / "ETTh1.csv"

    print(f"repo_root: {repo_root}")
    print(f"data_root: {data_root}")
    print(f"target: {target}")
    print(f"target exists: {target.exists()}")
    if not target.exists():
        raise FileNotFoundError(target)

    reader = DataReader(
        data_path=str(data_root) + "/",
        transform_ts_size=512,
        univariate=False,
    )

    dataset_name = "ETTh1_pred=96"
    print(f"dataset_name: {dataset_name}")
    for setting in ["train", "val", "test"]:
        x, y = reader.read_dataset(dataset_name=dataset_name, setting=setting)
        print(f"{setting} X shape: {shape_of(x)}")
        print(f"{setting} y shape: {shape_of(y)}")

    print("DataReader ETTh1 validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
