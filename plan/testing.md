# Verification / Test Plan

End-to-end checks an implementing agent should run, in order, with their acceptance criteria. See [README](README.md) for navigation.

End-to-end checks, in order, that an implementing agent should run.

| Stage | Command | Acceptance |
|---|---|---|
| Env | `uv sync && uv run python -c "import diffmechint"` | imports cleanly |
| Tests | `uv run pytest tests/` | all green |
| Adapter round-trip | `uv run python -m diffmechint.training.precompute_latents tokenizer=sd_vae max_images=64` | PSNR > 25 dB |
| All adapters | repeat for each tokenizer | all PSNR > 22 dB |
| FM-OT smoke | `uv run python -m diffmechint.training.train_sit tokenizer=sd_vae model=sit_b_2 trainer.max_steps=1000` | loss decreases, no NaN |
| Hooks smoke | `uv run pytest tests/test_hooks.py -v` | activation buffer contains expected shapes |
| Checkpoint schedule | `uv run python -m diffmechint.training.train_sit ... trainer.max_steps=10000 +ckpt_fractions=[0.1,0.5,1.0]` | 3 ckpt files written at expected steps |
| SAE smoke | `uv run python -m diffmechint.sae.train sae=topk_k32 layer=6 t_bin=200 ckpt=<path>` | recon cosine > 0.85, density 1-5% |
| Probe smoke | `uv run python -m diffmechint.probing.run_probes ckpt=<path>` | per-cell accuracy emitted |
| EAP smoke | `uv run python -m diffmechint.circuits.eap concept=dog ckpt=<path>` | non-empty edge list |
| Hungarian smoke | `uv run python -m diffmechint.analysis.hungarian_match sae_a=... sae_b=...` | matrix produced |

Full-pipeline sanity at the end of M3: 1 condition (SD-VAE), 1 checkpoint
(50% fraction), full Level-1 + Level-2 + Level-3 readout in < 24 h on
4× A100. If that works, scale.
