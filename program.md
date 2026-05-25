# Program

## 2026-05-24 — SiT-L/2 smoke and 500k training launch

Request: launch a small SiT-L/2 smoke run, then if it starts cleanly launch the
three K=3 SiT-L/2 production runs (`sd_vae`, `repa_e`, `eq_vae`) for 500k
steps. Prefer 4xA100 jobs while preserving the SiT-B/2 global batch size; if
4xA100 jobs wait too long in queue, fall back to 2xA100.

Success criteria:
- [x] Keep all new outputs under the `by_model/sit_l_2` namespace.
- [x] Do not overwrite existing run directories or legacy SiT-B/2 outputs.
- [x] Smoke test SiT-L/2 on 4xA100 with global batch 256.
- [x] If the smoke reaches training without OOM/import/config failure, submit
      all three 500k-step production jobs.
- [x] Record job IDs and output roots for monitoring.

Current launch defaults:
- 4xA100: `TRAINER_CONFIG=cineca_4xa100`, `BATCH_SIZE=64`, global batch 256.
- 2xA100 fallback: `TRAINER_CONFIG=cineca_2xa100`, `BATCH_SIZE=128`, global
  batch 256.
- Production target: `MAX_STEPS=500000`.

Launch record:
- Smoke `42380277` (`dmi_sit_l2_smoke_v2`) completed 2 steps on 4xA100.
- Validation-interval smoke `42381473` completed 2 steps with
  `VAL_CHECK_INTERVAL=5000` after aligning `cineca_4xa100` with the 2xA100
  global-step validation semantics.
- First 500k submission (`42380736`, `42380739`, `42380740`) failed before
  training because `cineca_4xa100` lacked `check_val_every_n_epoch: null`; the
  failed run directories were deleted on cleanup request.
- Active 500k v2 jobs:
  - `sd_vae`: `42382016`,
    `/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/by_model/sit_l_2/runs/sit_l_2_sd_vae_l2_500k_v2_full_20260524_130510`
  - `repa_e`: `42382017`,
    `/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/by_model/sit_l_2/runs/sit_l_2_repa_e_l2_500k_v2_full_20260524_130510`
  - `eq_vae` no-zscore: `42382018`,
    `/leonardo_scratch/large/userexternal/lcerovaz/diffmechint/by_model/sit_l_2/runs/sit_l_2_eq_vae_l2_noz_500k_v2_full_20260524_130510`
- All three v2 jobs reached Lightning training and had populated
  `lightning_logs/version_0/metrics.csv` through step 899 at the first check.

## 2026-05-24 — Resume recovered Phase 4.11 / Phase 6 bridge plan

Request: continue executing the canonical feature-level activation patching plan,
using `outputs/phase4_11_feature_activation_patching/recovered_long_plan_20260522.md`
as the fixed reference. Use CINECA GPU interactive sessions for debugging and
batch jobs for compute-heavy final runs.

Success criteria:
- [x] Preserve `recovered_long_plan_20260522.md` as the canonical reference.
- [x] Do not overwrite existing E20 outputs; every new run gets a unique run id.
- [x] Implement the missing group-to-group feature-family activation patching
      path required by the long plan.
- [x] Debug group patching on a small GPU-backed smoke run before final jobs.
- [x] Run full group/supernode patching for top shared and top unmatched concepts.
- [x] Run slope-gated/source-dependent final sampling candidates rather than
      bias-only target interventions.
- [x] Start Phase 6 EAP/sparse circuit candidate discovery after group patching.
- [x] Log completed bundles to Flywheel with corrected interpretation.

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

Completed in this continuation:
- Group patch smoke: `longplan_group_patch_smoke_20260524_113338`.
- Full group patch jobs: `longplan_group_patch_full_20260524_113515_*`.
- Group aggregate: `longplan_group_patch_full_20260524_113515_aggregate`.
- Concept-EAP smoke: `longplan_concept_eap_smoke_20260524_115900`.
- Full concept-EAP jobs: `longplan_concept_eap_full_20260524_120304_*`.
- Concept-EAP aggregate:
  `longplan_concept_eap_full_20260524_120304_aggregate`.
- Flywheel E21: `E21 Group Feature-Family Activation Patching Finds No Source-Specific Transfer`
  (`square-mode-2448`), 9 artifacts.
- Flywheel E22: `E22 Concept-Margin SAE EAP Shows Feature Circuits Depend on Error Node`
  (`muddy-brook-0282`), 13 artifacts.

Current result snapshot:
- Group patching covered 110 directed feature-family tasks across 45 groups and
  all six tokenizer directions. Mean group coefficient corr is 0.255, mean R2
  is 0.012, and transfer is essentially indistinguishable from shuffled pairing
  on average (`transfer - shuffled delta EV = 3.69e-7`).
- Concept-EAP covered 36 circuits (3 conditions x 3 production cells x 4 binary
  concepts), 5428 SAE feature nodes, and 36 reconstruction-error nodes. Mean
  probe accuracy is 0.834; mean reconstruction-error margin share is 0.264.
  Top-100 feature sufficiency with the error node retains about 0.273x of the
  clean concept-margin gap on average, while features-only retain about 0.020x.

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
