"""Run paired SiT-L/2 SAE feature-steering sampling tasks."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffmechint.utils import info, ok  # noqa: E402

DEFAULT_MODES = (
    "baseline",
    "native_ablate",
    "native_clamp_q95",
    "native_clamp_q99",
    "native_clamp_2x_q99",
    "transfer_replace",
    "random_matched_control",
    "wrong_window_control",
)

DEFAULT_SITL2_ACTIVATIONS_ROOT = Path(
    "/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/by_model/sit_l_2/"
    "activations_ynull_val_s50_imagenet_20260602_223258"
)

MODE_TO_PATCH_MODE = {
    "baseline": "baseline",
    "native_ablate": "native_ablate",
    "native_clamp_q95": "native_clamp",
    "native_clamp_q99": "native_clamp",
    "native_clamp_2x_q99": "native_clamp",
    "transfer_replace": "transfer_replace",
    "random_matched_control": "random_matched_control",
    "wrong_window_control": "wrong_window_control",
}

MODE_CLAMP_QUANTILE = {
    "native_clamp_q95": "0.95",
    "native_clamp_q99": "0.99",
    "native_clamp_2x_q99": "0.99",
}

MODE_CLAMP_MULTIPLIER = {
    "native_clamp_2x_q99": "2.0",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty task file: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def csv_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def require(row: dict[str, str], field: str) -> str:
    value = row.get(field, "")
    if value == "":
        raise ValueError(f"manifest row {row.get('candidate_id')} lacks {field}")
    return value


def build_sampler_command(
    row: dict[str, str],
    mode: str,
    *,
    out_root: Path,
    n_samples: int,
    batch_size: int,
    cfg: float,
    sample_steps: int,
    sampler: str,
    seed: int,
    class_schedule: str,
    error_mode: str,
    dit_step: int,
    model_name: str,
    activations_root: Path,
    keep_images: bool,
) -> list[str]:
    if mode not in MODE_TO_PATCH_MODE:
        raise ValueError(f"unsupported steering mode {mode!r}")
    patch_mode = MODE_TO_PATCH_MODE[mode]
    candidate_id = require(row, "candidate_id")
    candidate_out_root = out_root / "sampling" / candidate_id / f"error_{error_mode}" / class_schedule
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval" / "cross_tokenizer_feature_patching.py"),
        "--target_run",
        require(row, "sampling_task_target_run"),
        "--target_adapter",
        require(row, "sampling_task_target_adapter"),
        "--target_condition",
        require(row, "target"),
        "--mode",
        patch_mode,
        "--dit_step",
        str(dit_step),
        "--model_name",
        model_name,
        "--n_samples",
        str(n_samples),
        "--batch_size",
        str(batch_size),
        "--cfg",
        str(cfg),
        "--sample_steps",
        str(sample_steps),
        "--sampler",
        sampler,
        "--seed",
        str(seed),
        "--out_root",
        str(candidate_out_root),
        "--sae_root",
        require(row, "sampling_task_sae_root"),
        "--activations_root",
        str(activations_root),
        "--class_schedule",
        class_schedule,
        "--target_class_idx",
        require(row, "target_top_class_idx"),
        "--error_mode",
        error_mode,
    ]
    if not csv_bool(row.get("sampling_task_normalize", "true")):
        cmd.append("--no_normalize")
    if keep_images:
        cmd.append("--keep_images")
    if patch_mode != "baseline":
        cmd.extend(
            [
                "--layer",
                require(row, "layer"),
                "--t_bin",
                require(row, "t_bin"),
                "--target_feature_id",
                require(row, "target_feature_id"),
            ]
        )
    if mode in MODE_CLAMP_QUANTILE:
        cmd.extend(["--clamp_quantile", MODE_CLAMP_QUANTILE[mode]])
    if mode in MODE_CLAMP_MULTIPLIER:
        cmd.extend(["--clamp_multiplier", MODE_CLAMP_MULTIPLIER[mode]])
    if patch_mode in {"transfer_replace", "random_matched_control", "wrong_window_control"}:
        cmd.extend(
            [
                "--source_condition",
                require(row, "source"),
                "--source_feature_id",
                require(row, "source_feature_id"),
                "--random_source_feature_id",
                require(row, "sampling_task_random_source_feature_id"),
                "--source_to_target_scale",
                require(row, "sampling_task_scale"),
                "--source_to_target_bias",
                require(row, "sampling_task_bias"),
                "--wrong_t_bin",
                require(row, "sampling_task_wrong_t_bin"),
            ]
        )
    return cmd


def build_tasks(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = read_rows(args.manifest)
    if args.candidate_ids:
        wanted = set(args.candidate_ids)
        rows = [row for row in rows if row["candidate_id"] in wanted]
    if args.limit:
        rows = rows[: args.limit]
    tasks: list[dict[str, object]] = []
    for row in rows:
        for mode in args.modes:
            cmd = build_sampler_command(
                row,
                mode,
                out_root=args.out_root,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                cfg=args.cfg,
                sample_steps=args.sample_steps,
                sampler=args.sampler,
                seed=args.seed,
                class_schedule=args.class_schedule,
                error_mode=args.error_mode,
                dit_step=args.dit_step,
                model_name=args.model_name,
                activations_root=args.activations_root,
                keep_images=args.keep_images,
            )
            tasks.append(
                {
                    "task_id": f"steer_{len(tasks):04d}",
                    "candidate_id": row["candidate_id"],
                    "selection_role": row.get("selection_role", ""),
                    "mode": mode,
                    "patch_mode": MODE_TO_PATCH_MODE[mode],
                    "source": row["source"],
                    "target": row["target"],
                    "layer": row["layer"],
                    "t_bin": row["t_bin"],
                    "source_feature_id": row["source_feature_id"],
                    "target_feature_id": row["target_feature_id"],
                    "target_top_class_idx": row["target_top_class_idx"],
                    "target_top_label": row["target_top_label"],
                    "class_schedule": args.class_schedule,
                    "error_mode": args.error_mode,
                    "n_samples": args.n_samples,
                    "seed": args.seed,
                    "command": shlex.join(cmd),
                }
            )
    return tasks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--task_file", type=Path, default=None)
    p.add_argument("--candidate_ids", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--sampler", choices=["sde", "ode"], default="ode")
    p.add_argument("--cfg", type=float, default=1.5)
    p.add_argument("--sample_steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--class_schedule", type=str, default="target")
    p.add_argument("--error_mode", choices=["preserve", "drop", "only"], default="preserve")
    p.add_argument("--dit_step", type=int, default=350_000)
    p.add_argument("--model_name", type=str, default="SiT-L/2")
    p.add_argument("--activations_root", type=Path, default=DEFAULT_SITL2_ACTIVATIONS_ROOT)
    p.add_argument("--keep_images", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    tasks = build_tasks(args)
    task_path = args.task_file if args.task_file is not None else args.out_root / "metrics" / "steering_tasks.tsv"
    write_tsv(task_path, tasks)
    ok(f"wrote {len(tasks)} steering tasks to {task_path}")
    if args.dry_run:
        return 0
    for task in tasks:
        cmd = shlex.split(str(task["command"]))
        info(f"running {task['task_id']} {task['candidate_id']} {task['mode']}")
        subprocess.run(cmd, check=True)
    ok("steering tasks complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
