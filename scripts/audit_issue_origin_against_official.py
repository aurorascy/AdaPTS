from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import pandas as pd


LOGICAL_FILES = [
    "scripts/run.py",
    "src/adapts/adapts.py",
    "src/adapts/adapters.py",
    "src/adapts/utils/main_script.py",
    "src/adapts/utils/data_readers.py",
    "src/adapts/icl/moment.py",
]

DIFF_NAMES = {
    "scripts/run.py": "diff_run_py.patch",
    "src/adapts/adapts.py": "diff_adapts_py.patch",
    "src/adapts/adapters.py": "diff_adapters_py.patch",
    "src/adapts/utils/main_script.py": "diff_main_script_py.patch",
    "src/adapts/utils/data_readers.py": "diff_data_readers_py.patch",
    "src/adapts/icl/moment.py": "diff_moment_py.patch",
}

PATTERNS = {
    "fit_transform": "fit_transform",
    "scaler_fit": "scaler.fit",
    "scaler_transform": "scaler.transform",
    "def_transform": "def transform",
    "def_predict_multi_step": "def predict_multi_step",
    "base_projector_train": "base_projector_.train",
    "base_projector_eval": "base_projector_.eval",
    "reparameterize": "reparameterize",
    "torch_randn_like": "torch.randn_like",
    "justrevin_branch": "not args.adapter and args.use_revin",
    "JustRevIn": "JustRevIn",
    "ft_then_supervised": "ft_then_supervised",
    "adapter_supervised_fine_tuning": "adapter_supervised_fine_tuning",
    "local_files_only_true": "local_files_only=True",
    "local_files_only_false": "local_files_only=False",
    "freeze_encoder_true": '"freeze_encoder": True',
    "freeze_embedder_true": '"freeze_embedder": True',
    "freeze_head_false": '"freeze_head": False',
    "author_abs_path": "/mnt/data_2/abenechehab/AdaPTS/external_data/",
    "repo_root_path": "repo_root = Path(__file__).resolve().parents[3]",
    "reshape_n_components": "reshape(-1, self.n_components)",
    "reshape_input_dim": "reshape(-1, self.input_dim)",
}


def sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_official_file(official_root: Path, logical_file: str) -> Path | None:
    direct = official_root / logical_file
    if direct.exists():
        return direct
    name = Path(logical_file).name
    candidates = list(official_root.rglob(name))
    suffix = Path(logical_file).as_posix().lower()
    for candidate in candidates:
        if candidate.as_posix().lower().endswith(suffix):
            return candidate
    return candidates[0] if candidates else None


def line_hits(path: Path | None, pattern: str) -> list[str]:
    if not path or not path.exists():
        return []
    hits = []
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        if pattern in line:
            hits.append(f"{idx}: {line.strip()}")
    return hits


def has(path: Path | None, pattern: str) -> bool:
    return bool(line_hits(path, pattern))


def build_mapping(official_root: Path) -> pd.DataFrame:
    rows = []
    for logical in LOGICAL_FILES:
        local = Path(logical)
        official = find_official_file(official_root, logical)
        rows.append(
            {
                "logical_file": logical,
                "local_path": str(local),
                "official_path": str(official) if official else "",
                "local_exists": local.exists(),
                "official_exists": bool(official and official.exists()),
                "notes": "" if official else "official file not found",
            }
        )
    return pd.DataFrame(rows)


def write_diffs(mapping: pd.DataFrame, output_dir: Path) -> None:
    for _, row in mapping.iterrows():
        logical = row["logical_file"]
        local = Path(row["local_path"])
        official = Path(row["official_path"]) if row["official_path"] else None
        out = output_dir / DIFF_NAMES[logical]
        if not official or not official.exists() or not local.exists():
            out.write_text("file missing; diff not generated\n", encoding="utf-8")
            continue
        result = subprocess.run(
            ["git", "diff", "--no-index", "--", str(official), str(local)],
            text=True,
            capture_output=True,
        )
        out.write_text(result.stdout + result.stderr, encoding="utf-8")


def build_hashes(mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in mapping.iterrows():
        local = Path(row["local_path"])
        official = Path(row["official_path"]) if row["official_path"] else None
        local_hash = sha256(local)
        official_hash = sha256(official) if official else ""
        rows.append(
            {
                "logical_file": row["logical_file"],
                "local_sha256": local_hash,
                "official_sha256": official_hash,
                "same_hash": bool(local_hash and official_hash and local_hash == official_hash),
            }
        )
    return pd.DataFrame(rows)


def build_patterns(mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in mapping.iterrows():
        logical = row["logical_file"]
        local = Path(row["local_path"])
        official = Path(row["official_path"]) if row["official_path"] else None
        for pattern_name, pattern in PATTERNS.items():
            official_hits = line_hits(official, pattern)
            local_hits = line_hits(local, pattern)
            if official_hits or local_hits:
                rows.append(
                    {
                        "logical_file": logical,
                        "pattern_name": pattern_name,
                        "pattern": pattern,
                        "official_present": bool(official_hits),
                        "local_present": bool(local_hits),
                        "official_hits": " | ".join(official_hits[:8]),
                        "local_hits": " | ".join(local_hits[:8]),
                    }
                )
    return pd.DataFrame(rows)


def first_hit(path: Path | None, pattern: str) -> str:
    hits = line_hits(path, pattern)
    return hits[0] if hits else ""


def build_issue_summary(mapping: pd.DataFrame) -> pd.DataFrame:
    paths = {
        row["logical_file"]: (
            Path(row["official_path"]) if row["official_path"] else None,
            Path(row["local_path"]),
        )
        for _, row in mapping.iterrows()
    }
    off_adapts, loc_adapts = paths["src/adapts/adapts.py"]
    off_adapters, loc_adapters = paths["src/adapts/adapters.py"]
    off_run, loc_run = paths["scripts/run.py"]
    off_moment, loc_moment = paths["src/adapts/icl/moment.py"]
    off_main, loc_main = paths["src/adapts/utils/main_script.py"]
    off_data, loc_data = paths["src/adapts/utils/data_readers.py"]

    issues = []

    def add(issue_id, issue_name, origin, official_evidence, local_evidence, changed_by_local, severity, notes):
        issues.append(
            {
                "issue_id": issue_id,
                "issue_name": issue_name,
                "origin": origin,
                "official_evidence": official_evidence,
                "local_evidence": local_evidence,
                "changed_by_local": changed_by_local,
                "severity": severity,
                "notes": notes,
            }
        )

    off_scaler = has(off_adapts, "return self.adapter.transform(self.scaler.fit_transform(X))")
    loc_scaler = has(loc_adapts, "return self.adapter.transform(self.scaler.fit_transform(X))")
    add(
        "B01_scaler_prediction_refit",
        "ADAPTS.transform uses scaler.fit_transform, causing prediction/test-context refit through predict_multi_step",
        "official-existing" if off_scaler and loc_scaler else "local-introduced" if loc_scaler else "uncertain",
        first_hit(off_adapts, "return self.adapter.transform(self.scaler.fit_transform(X))"),
        first_hit(loc_adapts, "return self.adapter.transform(self.scaler.fit_transform(X))"),
        "no" if off_scaler == loc_scaler else "yes",
        "high",
        "predict_multi_step calls transform on X context; both codebases share this logic" if off_scaler and loc_scaler else "",
    )

    off_train = has(off_adapts, "self.adapter.base_projector_.train()")
    loc_train = has(loc_adapts, "self.adapter.base_projector_.train()")
    add(
        "R01_prediction_train_mode",
        "predict_multi_step sets torch adapter base_projector to train mode",
        "official-existing" if off_train and loc_train else "local-introduced" if loc_train else "uncertain",
        first_hit(off_adapts, "self.adapter.base_projector_.train()"),
        first_hit(loc_adapts, "self.adapter.base_projector_.train()"),
        "no" if off_train == loc_train else "yes",
        "medium",
        "Enables dropout stochasticity during prediction",
    )

    off_vae_sample = has(off_adapters, "eps = torch.randn_like(std)")
    loc_vae_sample = has(loc_adapters, "eps = torch.randn_like(std)")
    add(
        "R02_vae_eval_sampling",
        "VAE reparameterize samples with torch.randn_like regardless of eval mode",
        "official-existing" if off_vae_sample and loc_vae_sample else "local-introduced" if loc_vae_sample else "uncertain",
        first_hit(off_adapters, "eps = torch.randn_like(std)"),
        first_hit(loc_adapters, "eps = torch.randn_like(std)"),
        "no" if off_vae_sample == loc_vae_sample else "yes",
        "medium",
        "Static code confirms unconditional sampling; eval-mode behavior was verified in Stage 9 local probes",
    )

    off_shape = has(off_adapters, "self.revin(revin_input, mode=\"norm\").reshape(-1, self.n_components)")
    loc_shape = has(loc_adapters, "self.revin(revin_input, mode=\"norm\").reshape(-1, self.n_components)")
    add(
        "B02_linearAE_revin_shape",
        "LinearAE RevIN branch reshapes normalized input to n_components",
        "official-existing" if off_shape and loc_shape else "local-introduced" if loc_shape else "uncertain",
        first_hit(off_adapters, "self.revin(revin_input, mode=\"norm\").reshape(-1, self.n_components)"),
        first_hit(loc_adapters, "self.revin(revin_input, mode=\"norm\").reshape(-1, self.n_components)"),
        "no" if off_shape == loc_shape else "yes",
        "medium",
        "Breaks when n_components != input_dim",
    )

    # Same exact pattern appears in DropoutLinearAE too; count occurrences as evidence.
    off_shape_hits = line_hits(off_adapters, "self.revin(revin_input, mode=\"norm\").reshape(-1, self.n_components)")
    loc_shape_hits = line_hits(loc_adapters, "self.revin(revin_input, mode=\"norm\").reshape(-1, self.n_components)")
    add(
        "B03_dropoutLinearAE_revin_shape",
        "DropoutLinearAE RevIN branch reshapes normalized input to n_components",
        "official-existing" if len(off_shape_hits) >= 2 and len(loc_shape_hits) >= 2 else "local-introduced" if len(loc_shape_hits) >= 2 else "uncertain",
        off_shape_hits[1] if len(off_shape_hits) >= 2 else "",
        loc_shape_hits[1] if len(loc_shape_hits) >= 2 else "",
        "no" if len(off_shape_hits) == len(loc_shape_hits) else "yes",
        "medium",
        "Breaks when n_components != input_dim",
    )

    off_just = has(off_run, "if not args.adapter and args.use_revin") and has(off_run, "JustRevIn(")
    loc_just = has(loc_run, "if not args.adapter and args.use_revin") and has(loc_run, "JustRevIn(")
    off_ft_then = has(off_run, "elif args.supervised == \"ft_then_supervised\"") and has(off_run, "adapts_model.adapter_supervised_fine_tuning(")
    loc_ft_then = has(loc_run, "elif args.supervised == \"ft_then_supervised\"") and has(loc_run, "adapts_model.adapter_supervised_fine_tuning(")
    off_just_hit = first_hit(off_run, "if not args.adapter and args.use_revin")
    loc_just_hit = first_hit(loc_run, "if not args.adapter and args.use_revin")
    off_ft_hit = first_hit(off_run, 'elif args.supervised == "ft_then_supervised"')
    loc_ft_hit = first_hit(loc_run, 'elif args.supervised == "ft_then_supervised"')
    add(
        "P01_strong_baseline_justrevin",
        "adapter=None + use_revin=True creates JustRevIn and ft_then_supervised trains it",
        "official-existing" if off_just and loc_just and off_ft_then and loc_ft_then else "local-introduced" if loc_just and loc_ft_then else "uncertain",
        f"{off_just_hit} ; {off_ft_hit}",
        f"{loc_just_hit} ; {loc_ft_hit}",
        "no" if off_just == loc_just and off_ft_then == loc_ft_then else "yes",
        "high",
        "This explains why old baseline should be relabeled strong-baseline",
    )

    off_lfo_true = has(off_moment, "local_files_only=True")
    loc_lfo_false = has(loc_moment, "local_files_only=False")
    loc_lfo_true = has(loc_moment, "local_files_only=True")
    origin = "local-modified" if off_lfo_true and loc_lfo_false else "official-existing" if loc_lfo_true and off_lfo_true else "uncertain"
    add(
        "P02_moment_local_files_only",
        "MOMENT from_pretrained local_files_only setting",
        origin,
        first_hit(off_moment, "local_files_only="),
        first_hit(loc_moment, "local_files_only="),
        "yes" if origin == "local-modified" else "no",
        "low",
        "This likely affects first download/cache source, not inference code; checkpoint version may differ if cache was absent or updated",
    )

    off_abs = has(off_main, "/mnt/data_2/abenechehab/AdaPTS/external_data/")
    loc_repo = has(loc_main, "repo_root = Path(__file__).resolve().parents[3]")
    split_same = sha256(off_data) == sha256(loc_data) if off_data and loc_data.exists() else False
    add(
        "P03_data_path_change",
        "prepare_data data_path changed from author absolute path to repo-relative external_data",
        "local-modified" if off_abs and loc_repo else "uncertain",
        first_hit(off_main, "/mnt/data_2/abenechehab/AdaPTS/external_data/"),
        first_hit(loc_main, "repo_root = Path(__file__).resolve().parents[3]"),
        "yes",
        "low",
        f"data_readers.py same hash: {split_same}; path change should not alter split/scaler/window logic",
    )

    return pd.DataFrame(issues)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official_root", default=r"F:\AdaPTS\demo")
    parser.add_argument("--output_dir", default="results/issue_origin_audit")
    args = parser.parse_args()

    official_root = Path(args.official_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not official_root.exists():
        raise FileNotFoundError(f"Official root not found: {official_root}")

    mapping = build_mapping(official_root)
    mapping.to_csv(output_dir / "file_mapping.csv", index=False)

    hashes = build_hashes(mapping)
    hashes.to_csv(output_dir / "file_hashes.csv", index=False)

    write_diffs(mapping, output_dir)

    patterns = build_patterns(mapping)
    patterns.to_csv(output_dir / "pattern_presence_table.csv", index=False)

    issues = build_issue_summary(mapping)
    issues.to_csv(output_dir / "issue_origin_summary.csv", index=False)

    print(f"official_root: {official_root}")
    print(f"output_dir: {output_dir}")
    print("file mapping:")
    print(mapping.to_string(index=False))
    print("issue origins:")
    print(issues[["issue_id", "origin", "changed_by_local", "severity"]].to_string(index=False))


if __name__ == "__main__":
    main()
