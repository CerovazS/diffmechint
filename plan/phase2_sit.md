# Phase 2 — SiT Training Pipeline (FM-OT)

Lightning module wrapping SiT, FM-OT interpolant, matched-compute stopping, and the fractional checkpoint schedule. See [README](README.md) for navigation.

## LightningModule

`src/diffmechint/training/sit_module.py`:

```python
class SiTLightningModule(pl.LightningModule):
    def __init__(self, model_cfg, transport_cfg, tokenizer, optimizer_cfg, ema_decay=0.9999):
        super().__init__()
        self.model = hydra.utils.instantiate(model_cfg, in_channels=tokenizer.in_channels)
        self.transport = create_transport(**transport_cfg)
        self.ema = EMA(self.model, decay=ema_decay)
        # Tokenizer is frozen; latents arrive pre-encoded from datamodule
    
    def training_step(self, batch, batch_idx):
        z, labels = batch                          # latents, class labels
        loss_dict = self.transport.training_losses(self.model, z, dict(y=labels))
        return loss_dict["loss"].mean()
    
    def configure_optimizers(self): ...           # AdamW + warm-up
```

## FM-OT interpolant

Hydra config `conf/transport/fm_ot.yaml`:

```yaml
_target_: diffmechint.sit.transport.create_transport
path_type: Linear         # alpha_t = t, sigma_t = 1 - t  → optimal-transport coupling
prediction: velocity       # most stable, used by REPA, LightningDiT
loss_weight: null          # uniform weighting
train_eps: 0
sample_eps: 0
```

The proposal mentions a *Laplace logSNR schedule concentrated around
logSNR=0*. SiT's transport layer does not implement a Laplace logSNR
sampling; default is uniform t ∈ [0, 1]. **Action:** add a
`t_sampler: uniform | laplace` field to the transport config, implement
Laplace sampling in `src/diffmechint/sit/transport/sampling.py`, and wire
it into `training_losses`. Default `uniform` keeps parity with SiT
upstream; switch to `laplace` for the proposal's stated schedule.

## Optimizer + matched compute

- AdamW lr 1e-4, β=(0.9, 0.999), wd 0, warm-up 10k steps (matches SiT
  paper). Same across all conditions.
- **Stopping criterion** (`matched_compute.py`): trainer halts when
  *either* the compute budget is reached *or* gFID drops below the
  matched-target window. Reuse `clean-fid` package for FID computation.
  Per condition, log `(steps, wall-clock GPU-h, gFID)` to
  `outputs/<run_id>/training_curve.csv`.

## Fractional checkpoint schedule

`src/diffmechint/training/checkpointing.py`:

The proposal requires checkpoints at training fractions
`{2%, 5%, 10%, 25%, 50%, 75%, 100%}`. With matched-compute stopping the
total step count `S` is known per run, so checkpoint steps are
`{0.02·S, 0.05·S, 0.10·S, 0.25·S, 0.50·S, 0.75·S, 1.00·S}`. Implement as
a Lightning `ModelCheckpoint` with a custom `filename` and
`every_n_train_steps=None` plus a manual `on_train_batch_end` hook that
triggers a save when `global_step` crosses the next fractional target.

Each checkpoint writes:
- `step_{step:08d}.safetensors` (model weights)
- `step_{step:08d}_ema.safetensors` (EMA weights — these are the analysis
  target, since SAE work historically uses EMA)
- `step_{step:08d}_metadata.json` (loss, gFID, lr, fractional position)

Stored under `$FAST/diffmechint/runs/<vae>/<seed>/checkpoints/`.

## Distributed training on CINECA

- 4× A100 64GB per node, DDP via Lightning, batch 32/GPU = 128 global.
- `bf16-mixed` precision; FA2 attention (until grant unlocks H100+FA3).
- `torch.compile` enabled by default; fall back to eager if compile bugs.
- SLURM template lives at `slurm/train_sit.slurm` with placeholders for
  `--vae`, `--model`, `--seed`. Sets `http_proxy/https_proxy` to login01
  per CINECA rules.
- Per-condition wall-clock estimate (DiT-B, 256², matched compute):
  ~480 H100-h ≈ 850 A100-h ≈ 9 days on 4× A100.
- **Strategy on CINECA:** sequential conditions, not concurrent — fits the
  $FAST quota and the 24-h SLURM partition limit (use checkpoint resume
  across SLURM jobs).

## Acceptance gate for Phase 2

- 1k-step smoke run on SD-VAE produces decreasing loss, valid samples.
- Full DiT-B run on SD-VAE matches the SiT paper's gFID at 400k steps
  within 0.5 points.
- Checkpoint files at the 7 fractional steps land in `$FAST` with the
  correct schema.
