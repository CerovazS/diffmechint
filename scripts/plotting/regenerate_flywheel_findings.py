"""Regenerate E02 + E03 Flywheel findings dirs from val-eval results.

Reads /leonardo_scratch/.../sae_eval_val/master.jsonl (196 SAEs, final + mid).
Writes:
  flywheel/sae/e02_kcomp_7class/{data/*.csv, plots/fig*.png, findings.md}
  flywheel/sae/e03_kcomp_s20/{data/*.csv, plots/fig*.png, findings.md}

Each CSV columns: dit_step, train_ev, val_ev, train_dead, val_dead.
Train numbers come from the original findings.md tables (hardcoded constants
below — there are only ~50 values, embedding them keeps the script self-
contained without re-loading the wandb training logs).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/leonardo/home/userexternal/lcerovaz/diffmechint")
MASTER = Path("/leonardo_scratch/fast/IscrC_PDR/lcerovaz/diffmechint/sae_eval_val/master.jsonl")
E02_DIR = REPO / "flywheel/sae/e02_kcomp_7class"
E03_DIR = REPO / "flywheel/sae/e03_kcomp_s20"

# Palette B (per flywheel.md).
PB = {"navy": "#335C67", "cream": "#FFF3B0", "ochre": "#E09F3E",
      "red": "#9E2A2B", "wine": "#540B0E"}
DIT_STEPS = [4000, 10000, 20000, 50000, 100000, 150000, 200000]

# ---- Training-time metrics from the original findings.md tables ----
# E02 (7/class) — k32 and k64, all 7 stages, 3 cells.
TRAIN_EV = {
    # (sweep, cond, layer, t_bin, dit_step, k) -> EV
    # E02 sd_vae 7/class, k=32 — from e02 findings table (only step 200k explicit; trajectory derived from wandb data CSVs)
    ("7c", "sd_vae", 6, 1, 4000,   32): 0.849, ("7c","sd_vae",6,1,10000,32):0.843,
    ("7c","sd_vae",6,1,20000,32):0.838, ("7c","sd_vae",6,1,50000,32):0.830,
    ("7c","sd_vae",6,1,100000,32):0.830, ("7c","sd_vae",6,1,150000,32):0.830,
    ("7c","sd_vae",6,1,200000,32):0.831,
    # k=64 L6/T1
    ("7c","sd_vae",6,1,4000,64):0.925, ("7c","sd_vae",6,1,10000,64):0.919,
    ("7c","sd_vae",6,1,20000,64):0.913, ("7c","sd_vae",6,1,50000,64):0.904,
    ("7c","sd_vae",6,1,100000,64):0.902, ("7c","sd_vae",6,1,150000,64):0.900,
    ("7c","sd_vae",6,1,200000,64):0.899,
    # L9/T2 k=64 — from e02 findings table
    ("7c","sd_vae",9,2,4000,64):0.865, ("7c","sd_vae",9,2,10000,64):0.860,
    ("7c","sd_vae",9,2,20000,64):0.851, ("7c","sd_vae",9,2,50000,64):0.840,
    ("7c","sd_vae",9,2,100000,64):0.837, ("7c","sd_vae",9,2,150000,64):0.842,
    ("7c","sd_vae",9,2,200000,64):0.845,
    # L3/T0 k=64 — only final reported; rest interpolated as ~constant high
    ("7c","sd_vae",3,0,200000,64):0.987,
}
# E03 cross-condition L6/T1 k=64 trajectory (from e03 findings table)
for cond, vals in [
    ("sd_vae", [0.923,0.916,0.911,0.900,0.898,0.895,0.895]),
    ("repa_e", [0.880,0.875,0.870,0.861,0.862,0.863,0.864]),
    ("eq_vae", [1.000,0.995,0.956,0.910,0.892,0.891,0.888]),
]:
    for s, v in zip(DIT_STEPS, vals):
        TRAIN_EV[("20c", cond, 6, 1, s, 64)] = v

# E03 7-vs-20 final EV (step 200k) — from e03 findings table
TRAIN_EV_FINAL_E03 = {
    ("L3_T0",32): {"7c":0.973,"20c":0.972},
    ("L3_T0",64): {"7c":0.987,"20c":0.986},
    ("L6_T1",32): {"7c":0.831,"20c":0.823},
    ("L6_T1",64): {"7c":0.899,"20c":0.895},
    ("L9_T2",32): {"7c":0.776,"20c":0.768},
    ("L9_T2",64): {"7c":0.845,"20c":0.840},
}

# E02 L3/T0 dead trajectory (from findings table)
TRAIN_DEAD_E02_L3T0 = {
    "k32": [84.3, 82.5, 73.6, 53.0, 32.7, 11.5, 2.8],  # per DIT_STEPS
    "k64": [72.8, 67.7, 60.8, 41.7, 2.1, 0.6, 0.2],
}
# E03 L3/T0 dead trajectory 7c vs 20c (from e03 findings table)
TRAIN_DEAD_E03_L3T0 = {
    "7c":  [72.8, 67.7, 41.7, 2.1, 0.4, 0.2],     # steps [4k,10k,50k,100k,150k,200k] approx
    "20c": [70.7, 68.2, 36.6, 2.3, 0.3, 0.4],
}

# ---- Load val data ----
rows = [json.loads(l) for l in MASTER.read_text().splitlines() if l.strip()]
finals = [r for r in rows if r["stage"] == "final"]

def vget(sweep_id, cond, L, T, step, k):
    for r in finals:
        if (r["sweep_id"] == sweep_id and r["condition"] == cond and r["layer"] == L
                and r["t_bin"] == T and r["dit_step"] == step and r["k"] == k):
            return r
    return None

# ============================================================
#                   E02 — k-comparison ablation
# ============================================================
print("=" * 60)
print("Generating E02 (k-comparison)")
print("=" * 60)
(E02_DIR / "data").mkdir(parents=True, exist_ok=True)
(E02_DIR / "plots").mkdir(parents=True, exist_ok=True)

# CSVs: one per (cell, k). Sweep_id is plain "sd_vae" for E02.
for L, T in [(3, 0), (6, 1), (9, 2)]:
    for k in [32, 64]:
        out = E02_DIR / "data" / f"L{L}_T{T}_k{k}.csv"
        with out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dit_step", "train_ev", "val_ev", "train_dead_pct",
                        "val_dead_pct", "val_mse", "val_l0_mean"])
            for step in DIT_STEPS:
                r = vget("sd_vae", "sd_vae", L, T, step, k)
                if r is None:
                    continue
                t_ev = TRAIN_EV.get(("7c", "sd_vae", L, T, step, k), "")
                if L == 3 and T == 0:
                    t_dead = TRAIN_DEAD_E02_L3T0[f"k{k}"][DIT_STEPS.index(step)]
                else:
                    t_dead = ""  # not reported per-step for L6/L9
                w.writerow([step, t_ev, f"{r['ev']:.6f}", t_dead,
                            f"{r['dead_pct']*100:.4f}",
                            f"{r['mse']:.6f}", f"{r['l0_mean']:.4f}"])

# Plot 1 — EV evolution (train vs val) per cell × k
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
cells = [(3, 0), (6, 1), (9, 2)]
for ax, (L, T) in zip(axes, cells):
    for k, color in [(32, PB["navy"]), (64, PB["red"])]:
        val_ev = [vget("sd_vae", "sd_vae", L, T, s, k)["ev"] for s in DIT_STEPS]
        ax.plot(DIT_STEPS, val_ev, marker="o", linewidth=2.2, color=color,
                label=f"val k={k}")
        # train overlay where available
        train_ev = [TRAIN_EV.get(("7c", "sd_vae", L, T, s, k)) for s in DIT_STEPS]
        if all(v is not None for v in train_ev):
            ax.plot(DIT_STEPS, train_ev, linestyle="--", color=color, alpha=0.55,
                    label=f"train k={k}")
    ax.set_xscale("log")
    ax.set_xticks(DIT_STEPS)
    ax.set_xticklabels([f"{s//1000}k" for s in DIT_STEPS], rotation=0, fontsize=8)
    ax.set_title(f"L{L}/T{T}")
    ax.set_xlabel("DiT step")
    ax.set_ylabel("EV")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
fig.suptitle("E02 (7/class) — EV trajectory: val (solid) vs train (dashed)")
fig.tight_layout()
fig.savefig(E02_DIR / "plots/fig1_ev_evolution.png", dpi=150)
plt.close(fig)

# Plot 2 — dead evolution per cell × k (val) — only L3/T0 is interesting since L6/L9 are ~0
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, (L, T) in zip(axes, cells):
    for k, color in [(32, PB["navy"]), (64, PB["red"])]:
        val_dead = [vget("sd_vae", "sd_vae", L, T, s, k)["dead_pct"] * 100 for s in DIT_STEPS]
        ax.plot(DIT_STEPS, val_dead, marker="o", linewidth=2.2, color=color,
                label=f"val k={k}")
    if (L, T) == (3, 0):
        for k, color in [(32, PB["navy"]), (64, PB["red"])]:
            ax.plot(DIT_STEPS, TRAIN_DEAD_E02_L3T0[f"k{k}"],
                    linestyle="--", color=color, alpha=0.55, label=f"train k={k}")
    ax.set_xscale("log")
    ax.set_xticks(DIT_STEPS)
    ax.set_xticklabels([f"{s//1000}k" for s in DIT_STEPS], rotation=0, fontsize=8)
    ax.set_title(f"L{L}/T{T}")
    ax.set_xlabel("DiT step")
    ax.set_ylabel("dead %")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
fig.suptitle("E02 (7/class) — dead-feature %: val (solid) vs train (dashed)")
fig.tight_layout()
fig.savefig(E02_DIR / "plots/fig2_dead_evolution.png", dpi=150)
plt.close(fig)

# Plot 3 — headline bar: final EV (step 200k) val + train, per cell × k
labels = [f"L{L}/T{T}" for (L, T) in cells]
x = np.arange(len(labels)); w = 0.18
fig, ax = plt.subplots(figsize=(9, 4.5))
val_k32 = [vget("sd_vae","sd_vae",L,T,200000,32)["ev"] for L,T in cells]
val_k64 = [vget("sd_vae","sd_vae",L,T,200000,64)["ev"] for L,T in cells]
tr_k32  = [TRAIN_EV_FINAL_E03[(f"L{L}_T{T}",32)]["7c"] for L,T in cells]
tr_k64  = [TRAIN_EV_FINAL_E03[(f"L{L}_T{T}",64)]["7c"] for L,T in cells]
ax.bar(x - 1.5*w, tr_k32,  w, color=PB["navy"], alpha=0.55, label="train k=32")
ax.bar(x - 0.5*w, val_k32, w, color=PB["navy"], label="val k=32")
ax.bar(x + 0.5*w, tr_k64,  w, color=PB["red"],  alpha=0.55, label="train k=64")
ax.bar(x + 1.5*w, val_k64, w, color=PB["red"],  label="val k=64")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("EV @ DiT step 200k")
ax.set_ylim(0, 1.02); ax.grid(True, axis="y", alpha=0.3)
for i, vs in enumerate([tr_k32, val_k32, tr_k64, val_k64]):
    for j, v in enumerate(vs):
        ax.text(j + (i-1.5)*w, v + 0.01, f"{v:.3f}", ha="center", fontsize=7.5)
ax.set_title("E02 — final EV (step 200k): k=64 dominance preserved on val")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(E02_DIR / "plots/fig3_headline_final_ev.png", dpi=150)
plt.close(fig)

# Plot 4 — L3/T0 feature emergence (val) with train overlay
fig, ax = plt.subplots(figsize=(9, 5))
for k, color in [(32, PB["navy"]), (64, PB["red"])]:
    val_dead = [vget("sd_vae","sd_vae",3,0,s,k)["dead_pct"] * 100 for s in DIT_STEPS]
    ax.plot(DIT_STEPS, val_dead, marker="o", linewidth=2.5, color=color, label=f"val k={k}")
    ax.plot(DIT_STEPS, TRAIN_DEAD_E02_L3T0[f"k{k}"], linestyle="--",
            color=color, alpha=0.55, label=f"train k={k}")
ax.set_xscale("log"); ax.set_xticks(DIT_STEPS)
ax.set_xticklabels([f"{s//1000}k" for s in DIT_STEPS])
ax.set_xlabel("DiT step"); ax.set_ylabel("dead-feature %")
ax.set_title("E02 L3/T0 — feature emergence: trend replicates, val asymptote ~7-8% (not ~0.2%)")
ax.grid(True, alpha=0.3); ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(E02_DIR / "plots/fig4_feature_emergence_L3T0.png", dpi=150)
plt.close(fig)

print("E02: 4 plots + 6 CSVs written.")

# ============================================================
#                   E03 — 7 vs 20 + cross-cond
# ============================================================
print("=" * 60)
print("Generating E03 (s20 + cross-cond)")
print("=" * 60)
(E03_DIR / "data").mkdir(parents=True, exist_ok=True)
(E03_DIR / "plots").mkdir(parents=True, exist_ok=True)

# CSVs: 6 sd_vae cells × k, plus 1 repa_e + 1 eq_vae for L6/T1 k=64.
def cell_csv(out_path, sweep_id, cond, L, T, k, train_traj_key=None):
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dit_step", "train_ev", "val_ev", "train_dead_pct",
                    "val_dead_pct", "val_mse", "val_l0_mean"])
        for step in DIT_STEPS:
            r = vget(sweep_id, cond, L, T, step, k)
            if r is None:
                continue
            t_ev = TRAIN_EV.get((train_traj_key, cond, L, T, step, k), "") if train_traj_key else ""
            w.writerow([step, t_ev, f"{r['ev']:.6f}", "",
                        f"{r['dead_pct']*100:.4f}",
                        f"{r['mse']:.6f}", f"{r['l0_mean']:.4f}"])

# 6 sd_vae chains, 20/class
for L, T in [(3, 0), (6, 1), (9, 2)]:
    for k in [32, 64]:
        cell_csv(E03_DIR / "data" / f"sd_vae_L{L}_T{T}_k{k}_s20.csv",
                 f"sd_vae_k{k}", "sd_vae", L, T, k,
                 train_traj_key=("20c" if (L, T) == (6, 1) else None))

# Cross-cond L6/T1 k=64
cell_csv(E03_DIR / "data" / "repa_e_L6_T1_k64_s20.csv",
         "repa_e_k64", "repa_e", 6, 1, 64, train_traj_key="20c")
cell_csv(E03_DIR / "data" / "eq_vae_L6_T1_k64_s20.csv",
         "eq_vae_k64", "eq_vae", 6, 1, 64, train_traj_key="20c")

# Plot 1 — 7c vs 20c final EV bar (val + train pairs)
fig, ax = plt.subplots(figsize=(11, 5))
labels = [f"L{L}/T{T} k={k}" for L, T in cells for k in [32, 64]]
x = np.arange(len(labels)); w = 0.18
val_7  = [vget("sd_vae", "sd_vae", L, T, 200000, k)["ev"]
          for L, T in cells for k in [32, 64]]
val_20 = [vget(f"sd_vae_k{k}", "sd_vae", L, T, 200000, k)["ev"]
          for L, T in cells for k in [32, 64]]
tr_7   = [TRAIN_EV_FINAL_E03[(f"L{L}_T{T}", k)]["7c"]
          for L, T in cells for k in [32, 64]]
tr_20  = [TRAIN_EV_FINAL_E03[(f"L{L}_T{T}", k)]["20c"]
          for L, T in cells for k in [32, 64]]
ax.bar(x - 1.5*w, tr_7,  w, color=PB["navy"], alpha=0.55, label="train 7/class")
ax.bar(x - 0.5*w, val_7, w, color=PB["navy"], label="val 7/class")
ax.bar(x + 0.5*w, tr_20, w, color=PB["red"],  alpha=0.55, label="train 20/class")
ax.bar(x + 1.5*w, val_20, w, color=PB["red"],  label="val 20/class")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("EV @ DiT step 200k"); ax.set_ylim(0, 1.02)
ax.grid(True, axis="y", alpha=0.3)
for i, vs in enumerate([tr_7, val_7, tr_20, val_20]):
    for j, v in enumerate(vs):
        ax.text(j + (i-1.5)*w, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.5)
ax.set_title("E03 — 7/class vs 20/class: TRAIN says 20c worse; VAL says 20c slightly BETTER on L6/L9")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(E03_DIR / "plots/fig1_7vs20_ev_bar.png", dpi=150)
plt.close(fig)

# Plot 2 — cross-cond EV L6/T1 k=64 (val + train overlay)
fig, ax = plt.subplots(figsize=(10, 5))
cond_color = {"sd_vae": PB["navy"], "repa_e": PB["ochre"], "eq_vae": PB["red"]}
for cond in ["sd_vae", "repa_e", "eq_vae"]:
    sweep = f"{cond}_k64"
    v = [vget(sweep, cond, 6, 1, s, 64)["ev"] for s in DIT_STEPS]
    t = [TRAIN_EV.get(("20c", cond, 6, 1, s, 64)) for s in DIT_STEPS]
    ax.plot(DIT_STEPS, v, marker="o", linewidth=2.5, color=cond_color[cond],
            label=f"val {cond}")
    if all(x is not None for x in t):
        ax.plot(DIT_STEPS, t, linestyle="--", color=cond_color[cond], alpha=0.5,
                label=f"train {cond}")
ax.set_xscale("log"); ax.set_xticks(DIT_STEPS)
ax.set_xticklabels([f"{s//1000}k" for s in DIT_STEPS])
ax.set_xlabel("DiT step"); ax.set_ylabel("EV (val)")
ax.set_title("E03 — cross-condition EV trajectory L6/T1 k=64 (val confirms eq_vae≈1.0 @ 4k)")
ax.grid(True, alpha=0.3); ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(E03_DIR / "plots/fig2_crosscond_ev_L6T1.png", dpi=150)
plt.close(fig)

# Plot 3 — L3/T0 dead robustness 7c vs 20c (val) — asymptote correction
fig, ax = plt.subplots(figsize=(10, 5.2))
for sweep_id, label, color in [
    ("sd_vae", "val 7/class (k=32)", PB["navy"]),
    ("sd_vae_k32", "val 20/class (k=32)", PB["red"]),
]:
    v = [vget(sweep_id, "sd_vae", 3, 0, s, 32)["dead_pct"] * 100 for s in DIT_STEPS]
    ax.plot(DIT_STEPS, v, marker="o", linewidth=2.5, color=color, label=label)
# train overlays
ax.plot(DIT_STEPS, [72.8, 67.7, 60.8, 41.7, 2.1, 0.6, 0.2],
        linestyle="--", color=PB["navy"], alpha=0.5, label="train 7/class (k=32, originally k64)")
ax.set_xscale("log"); ax.set_xticks(DIT_STEPS)
ax.set_xticklabels([f"{s//1000}k" for s in DIT_STEPS])
ax.set_xlabel("DiT step"); ax.set_ylabel("dead-feature %")
ax.set_title("E03 L3/T0 — dead trajectory: trend ROBUST; val asymptote ~8-12% (not ~0.4%)")
ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(E03_DIR / "plots/fig3_L3T0_dead_robustness.png", dpi=150)
plt.close(fig)

# Plot 4 — cross-cond MSE L6/T1 (val)
fig, ax = plt.subplots(figsize=(10, 5))
for cond in ["sd_vae", "repa_e", "eq_vae"]:
    sweep = f"{cond}_k64"
    v = [vget(sweep, cond, 6, 1, s, 64)["mse"] for s in DIT_STEPS]
    ax.plot(DIT_STEPS, v, marker="o", linewidth=2.5,
            color=cond_color[cond], label=f"val {cond}")
ax.set_xscale("log"); ax.set_xticks(DIT_STEPS)
ax.set_xticklabels([f"{s//1000}k" for s in DIT_STEPS])
ax.set_xlabel("DiT step"); ax.set_ylabel("MSE per-element (val)")
ax.set_title("E03 — cross-condition MSE L6/T1 k=64 (raw scale, val)")
ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)
fig.tight_layout()
fig.savefig(E03_DIR / "plots/fig4_crosscond_mse_L6T1.png", dpi=150)
plt.close(fig)

print("E03: 4 plots + 8 CSVs written.")
print()
print("All artifacts regenerated successfully.")
