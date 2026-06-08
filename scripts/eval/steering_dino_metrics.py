"""Compute DINOv2 smoke metrics for paired feature-steering samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from timm import create_model
from timm.data import create_transform, resolve_model_data_config
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor

from diffmechint.utils import ok, warn, write_csv

DEFAULT_DASHBOARD_ROOT = Path("outputs/phase4_15_sitl2_ynull_atlas_350k_20260603_145105/feature_viz")
DEFAULT_IMAGENET_VAL = Path(
    "/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/imagenet_val_imagefolder"
)
DEFAULT_DINO_MODEL = "vit_base_patch14_dinov2.lvd142m"
DEFAULT_CLASSIFIER_MODEL = "resnet50.a1_in1k"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def candidate_row(manifest: Path, candidate_id: str) -> dict[str, str]:
    rows = [row for row in read_csv(manifest) if row["candidate_id"] == candidate_id]
    if len(rows) != 1:
        raise ValueError(f"expected one manifest row for {candidate_id}, found {len(rows)}")
    return rows[0]


def load_feature_top_paths(
    row: dict[str, str],
    *,
    dashboard_root: Path,
    imagenet_val_root: Path,
) -> list[Path]:
    feature_json = (
        dashboard_root
        / f"{row['target']}_L{row['layer']}_T{row['t_bin']}"
        / "features"
        / f"feature_{row['target_feature_id']}.json"
    )
    payload = json.loads(feature_json.read_text(encoding="utf-8"))
    paths = []
    for item in payload.get("top", []):
        path = imagenet_val_root / str(item["synset"]) / str(item["filename"])
        if path.exists():
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"no top-example images resolved from {feature_json}")
    return paths


def _canonical_mode(payload: dict[str, object]) -> str:
    mode = str(payload["mode"])
    if mode != "native_clamp":
        return mode
    run_tag = str(payload.get("run_tag", ""))
    if "__q0p95__m1p0__" in run_tag:
        return "native_clamp_q95"
    if "__q0p99__m1p0__" in run_tag:
        return "native_clamp_q99"
    if "__q0p99__m2p0__" in run_tag:
        return "native_clamp_2x_q99"
    return "native_clamp_ambiguous"


def load_sampling_runs(sampling_roots: list[Path]) -> list[dict[str, object]]:
    runs = []
    seen_modes: set[str] = set()
    for sampling_root in sampling_roots:
        for fid_path in sorted(sampling_root.glob("*/fid.json")):
            payload = json.loads(fid_path.read_text(encoding="utf-8"))
            mode = _canonical_mode(payload)
            if mode == "native_clamp_ambiguous":
                warn(f"skip ambiguous native_clamp run: {fid_path.parent}")
                continue
            if mode in seen_modes:
                raise ValueError(f"duplicate mode {mode!r} across sampling roots")
            sample_dir = fid_path.parent / "samples"
            metadata_path = sample_dir / "sample_metadata.tsv"
            if not metadata_path.exists():
                warn(f"skip {fid_path.parent}: missing sample_metadata.tsv")
                continue
            seen_modes.add(mode)
            runs.append(
                {
                    "run_dir": fid_path.parent,
                    "sample_dir": sample_dir,
                    "metadata_path": metadata_path,
                    "mode": mode,
                    "run_tag": payload["run_tag"],
                    "fid": payload.get("fid"),
                    "hook_stats": payload.get("hook_stats", {}),
                }
            )
    if not runs:
        raise FileNotFoundError(f"no sampling runs found under {sampling_roots}")
    return runs


def image_tensor(path: Path, transform) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transform(image)


@torch.no_grad()
def embed_paths(paths: list[Path], *, model, transform, device: torch.device, batch_size: int) -> torch.Tensor:
    chunks = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = torch.stack([image_tensor(path, transform) for path in batch_paths]).to(device)
        emb = model(batch)
        if isinstance(emb, (list, tuple)):
            emb = emb[0]
        chunks.append(F.normalize(emb.float(), dim=-1).cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def classifier_scores(
    paths: list[Path],
    *,
    model,
    transform,
    device: torch.device,
    batch_size: int,
    target_class_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = []
    probs = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = torch.stack([image_tensor(path, transform) for path in batch_paths]).to(device)
        out = model(batch).float()
        logits.append(out[:, target_class_idx].detach().cpu())
        probs.append(out.softmax(dim=-1)[:, target_class_idx].detach().cpu())
    return torch.cat(logits, dim=0), torch.cat(probs, dim=0)


@torch.no_grad()
def clip_text_scores(
    paths: list[Path],
    *,
    model: CLIPModel,
    processor: CLIPProcessor,
    text: str,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    def as_feature_tensor(raw):
        return raw.pooler_output if hasattr(raw, "pooler_output") else raw

    text_inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    text_emb = as_feature_tensor(model.get_text_features(**text_inputs))
    text_emb = F.normalize(text_emb.float(), dim=-1)
    sims = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        image_emb = as_feature_tensor(model.get_image_features(**inputs))
        image_emb = F.normalize(image_emb.float(), dim=-1)
        sims.append((image_emb @ text_emb.T).reshape(-1).detach().cpu())
    return torch.cat(sims, dim=0)


def run_metrics(args: argparse.Namespace) -> int:
    row = candidate_row(args.candidate_manifest, args.candidate_id)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dino_model = create_model(args.dino_model, pretrained=True, num_classes=0).to(device).eval()
    dino_transform = create_transform(**resolve_model_data_config(dino_model))
    classifier_model = create_model(args.classifier_model, pretrained=True).to(device).eval()
    classifier_transform = create_transform(**resolve_model_data_config(classifier_model))
    clip_model = CLIPModel.from_pretrained(args.clip_model, local_files_only=True).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model, local_files_only=True)
    target_class_idx = int(row["target_top_class_idx"])
    target_text = args.target_text or f"a photo of a {row['target_top_label']}"

    top_paths = load_feature_top_paths(row, dashboard_root=args.dashboard_root, imagenet_val_root=args.imagenet_val_root)
    top_emb = embed_paths(
        top_paths,
        model=dino_model,
        transform=dino_transform,
        device=device,
        batch_size=args.batch_size,
    )
    top_centroid = F.normalize(top_emb.mean(dim=0, keepdim=True), dim=-1)

    runs = load_sampling_runs(args.sampling_root)
    baseline_runs = [run for run in runs if run["mode"] == "baseline"]
    if len(baseline_runs) != 1:
        raise ValueError(f"expected one baseline run, found {len(baseline_runs)}")
    baseline_meta = read_tsv(baseline_runs[0]["metadata_path"])
    baseline_paths = [baseline_runs[0]["sample_dir"] / row["filename"] for row in baseline_meta]
    baseline_emb = embed_paths(
        baseline_paths,
        model=dino_model,
        transform=dino_transform,
        device=device,
        batch_size=args.batch_size,
    )
    baseline_by_sample = {
        int(meta["sample_id"]): baseline_emb[idx]
        for idx, meta in enumerate(baseline_meta)
    }

    per_image_rows = []
    summary_rows = []
    for run in runs:
        meta_rows = read_tsv(run["metadata_path"])
        paths = [run["sample_dir"] / row["filename"] for row in meta_rows]
        emb = embed_paths(
            paths,
            model=dino_model,
            transform=dino_transform,
            device=device,
            batch_size=args.batch_size,
        )
        cls_logits, cls_probs = classifier_scores(
            paths,
            model=classifier_model,
            transform=classifier_transform,
            device=device,
            batch_size=args.batch_size,
            target_class_idx=target_class_idx,
        )
        clip_scores = clip_text_scores(
            paths,
            model=clip_model,
            processor=clip_processor,
            text=target_text,
            device=device,
            batch_size=args.batch_size,
        )
        top_sim = (emb @ top_centroid.T).reshape(-1)
        preservation = []
        for idx, meta in enumerate(meta_rows):
            base = baseline_by_sample[int(meta["sample_id"])]
            preservation.append(float((emb[idx] * base).sum().item()))
            per_image_rows.append(
                {
                    "candidate_id": args.candidate_id,
                    "run_tag": run["run_tag"],
                    "mode": run["mode"],
                    "sample_id": meta["sample_id"],
                    "seed": meta["seed"],
                    "class_id": meta["class_id"],
                    "filename": meta["filename"],
                    "classifier_target_logit": float(cls_logits[idx].item()),
                    "classifier_target_probability": float(cls_probs[idx].item()),
                    "clip_target_text_similarity": float(clip_scores[idx].item()),
                    "dino_top_example_similarity": float(top_sim[idx].item()),
                    "dino_preservation_vs_baseline": preservation[-1],
                }
            )
        hook_stats = run.get("hook_stats", {})
        summary_rows.append(
            {
                "candidate_id": args.candidate_id,
                "run_tag": run["run_tag"],
                "mode": run["mode"],
                "fid_smoke": run.get("fid"),
                "n_images": len(meta_rows),
                "classifier_target_logit_mean": float(cls_logits.mean().item()),
                "classifier_target_logit_median": float(cls_logits.median().item()),
                "classifier_target_probability_mean": float(cls_probs.mean().item()),
                "classifier_target_probability_median": float(cls_probs.median().item()),
                "clip_target_text_similarity_mean": float(clip_scores.mean().item()),
                "clip_target_text_similarity_median": float(clip_scores.median().item()),
                "dino_top_example_similarity_mean": float(top_sim.mean().item()),
                "dino_top_example_similarity_median": float(top_sim.median().item()),
                "dino_preservation_vs_baseline_mean": float(torch.tensor(preservation).mean().item()),
                "dino_preservation_vs_baseline_median": float(torch.tensor(preservation).median().item()),
                "hook_active": hook_stats.get("active", ""),
                "hook_skipped": hook_stats.get("skipped", ""),
                "hook_no_t": hook_stats.get("no_t", ""),
            }
        )
    write_csv(args.out_dir / "metrics" / "steering_dino_per_image.csv", per_image_rows)
    write_csv(args.out_dir / "metrics" / "steering_dino_mode_summary.csv", summary_rows)
    payload = {
        "candidate_id": args.candidate_id,
        "dino_model": args.dino_model,
        "classifier_model": args.classifier_model,
        "clip_model": args.clip_model,
        "target_text": target_text,
        "top_example_paths": [str(path) for path in top_paths],
        "n_runs": len(runs),
        "n_per_image_rows": len(per_image_rows),
        "classifier_metric": "target_class_logit_and_probability",
        "clip_text_metric": "target_text_cosine_similarity",
    }
    (args.out_dir / "metrics" / "steering_dino_metric_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    ok(f"DINO steering metrics complete: {args.out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate_manifest", type=Path, required=True)
    p.add_argument("--candidate_id", type=str, required=True)
    p.add_argument("--sampling_root", type=Path, nargs="+", required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--dashboard_root", type=Path, default=DEFAULT_DASHBOARD_ROOT)
    p.add_argument("--imagenet_val_root", type=Path, default=DEFAULT_IMAGENET_VAL)
    p.add_argument("--dino_model", type=str, default=DEFAULT_DINO_MODEL)
    p.add_argument("--classifier_model", type=str, default=DEFAULT_CLASSIFIER_MODEL)
    p.add_argument("--clip_model", type=str, default=DEFAULT_CLIP_MODEL)
    p.add_argument("--target_text", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return run_metrics(args)


if __name__ == "__main__":
    raise SystemExit(main())
