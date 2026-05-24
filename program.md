# Program

## 2026-05-24 — Resume recovered Phase 4.11 / Phase 6 bridge plan

Request: continue executing the canonical feature-level activation patching plan,
using `outputs/phase4_11_feature_activation_patching/recovered_long_plan_20260522.md`
as the fixed reference. Use CINECA GPU interactive sessions for debugging and
batch jobs for compute-heavy final runs.

Success criteria:
- [ ] Preserve `recovered_long_plan_20260522.md` as the canonical reference.
- [ ] Do not overwrite existing E20 outputs; every new run gets a unique run id.
- [ ] Implement the missing group-to-group feature-family activation patching
      path required by the long plan.
- [ ] Debug group patching on a small GPU-backed smoke run before final jobs.
- [ ] Run full group/supernode patching for top shared and top unmatched concepts.
- [ ] Run slope-gated/source-dependent final sampling candidates rather than
      bias-only target interventions.
- [ ] Start Phase 6 EAP/sparse circuit candidate discovery after group patching.
- [ ] Log completed bundles to Flywheel with corrected interpretation.

Current reference state:
- E20 is corrected: it supports no strict one-to-one dictionary identity, but
  the two N=5000 positives are target-dominant after calibration.
- Completed artifacts: full atlas bank, full Hungarian matching, descriptive
  feature-family grouping, activation-space patching, and partial final sampling.
- Missing artifacts: group-to-group causal patching, EAP/sparse circuits,
  source-dependent nonzero-slope final sampling, and full taxonomy separation.

Immediate next run family:
- `longplan_group_patch_<timestamp>` for implementation smoke and full
  activation-space group patching.

## 2026-05-23 — y-null Matryoshka vs TopK comparison

Request: compare the existing E04 TopK and Matryoshka SAE families on the
`y_null` activation distribution before replacing any old Flywheel E04 plots.

Success criteria:
- [x] Reuse existing trained SAE checkpoints without overwriting them.
- [x] Evaluate TopK k=128 d=32k and Matryoshka k=256 d=32k on matched
      `activations_ynull` shards.
- [x] Restrict headline plots to the available y-null production DiT checkpoint
      `200000` (the y-null activation tree only exists at 200k).
- [x] Write isolated outputs under a unique run directory.
- [x] Produce CSV/JSON metrics and E04-style plots for validation EV and dead%.
- [x] Report whether Matryoshka still wins on the downstream y-null distribution.

Run id: `ynull_e04_sae_compare_20260523`

Current run:
- Failed TopK y-null eval job: SLURM `42336456` (`rglob` filesystem failure
  before writing metrics)
- Retry TopK y-null eval job: SLURM `42342148`, output
  `topk_eval_retry_ckptlist/`, using explicit `topk_step200k_final_ckpts.txt`
  Completed with 27/27 rows in 01:23:03.
- Matryoshka y-null eval source: `outputs/phase4_7_ynull/eval/y_null/aggregate.csv`
- Isolated output root:
  `outputs/phase4_12_ynull_topk_vs_matryoshka/ynull_e04_sae_compare_20260523/`

Result:
- Matryoshka wins EV on 27/27 matched y-null cells.
- Mean EV: TopK `0.94737`, Matryoshka `0.97779`, delta `+0.03042`.
- Mean dead percentage: TopK `7.55%`, Matryoshka `12.89%`, delta
  `+5.34` percentage points.
- Reports and plots are under `plots/` and `reports/` in the run root.
- E04-compatible replacement plots were written to
  `flywheel_e04_replacement_plots/`, copied into
  `flywheel/sae/e04_matryoshka_vs_topk/plots/`, and uploaded to Flywheel node
  `E04 Matryoshka SAE Beats Plain TopK on Held-Out Diffusion Activations`
  (`cool-scene-6995`, node id `f03466d1-3ebc-5ce2-898a-47a05fdee120`).
  Old local E04 plots were backed up under `e04_old_plots_backup_20260523/`.
