"""Experiment 1 (v2) — is position written into phase? (readout direction)

Changes from v1, driven by the first run:
- Part A: slope is now fit on ONLY the lowest-|fx| bins (naive all-bin fit
  was poisoned by wrapped/aliased high-freq bins — the paper's low-frequency
  argument, now enacted), with a wrap-proof phase-correlation estimate
  printed alongside. Expect BOTH to track true dx out to 32.
- Part B: tiny per-frame sensor noise added so the still scene yields
  realistic near-zero metrics instead of nan (bit-identical frames).
- Part C: unchanged (the VAE's own encoder jitter already plays that role;
  a negative still-scene pair-corr = mean-reverting jitter is expected).

Run from code/experiments/:  python exp1_readout.py [--skip_latents]
"""
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from common import (render_scene, to_gray, phase_diff, phase_correlation,
                    top_peaks, freq_grid, motion_phase_energy,
                    dphi_pair_correlation, eq4_check, load_vae, encode_video)

OUT = "outputs/exp1"; os.makedirs(OUT, exist_ok=True)

def fit_lowfreq_slope(dphi_row, weight_row, fx_row, n_bins=3):
    """Weighted LS slope of dphi vs fx using ONLY the 2*n_bins lowest
    nonzero-|fx| bins. With W=384, n_bins=3 -> max |fx| = 3/384, which stays
    unwrapped for any dx < 64. Recovered shift: dx_hat = -slope / (2*pi)."""
    order = np.argsort(np.abs(fx_row))
    sel = order[1:1 + 2 * n_bins]                 # skip DC
    w, x, y = weight_row[sel], fx_row[sel], dphi_row[sel]
    return float(np.sum(w * x * y) / (np.sum(w * x * x) + 1e-12))

def part_a():
    print("\n=== A. Global shift (exact shift theorem) ===")
    g = to_gray(render_scene(F=1, H=256, W=384, seed=0))[0]
    H, W = g.shape
    FX, _ = freq_grid(H, W)
    row = H // 2                                  # ky = 0 after fftshift
    shifts = [0, 1, 2, 4, 8, 16, 24, 32]
    rec_slope, rec_pc = [], []
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for dx in shifts:
        g2 = np.roll(g, dx, axis=1)               # circular: theorem is exact
        dphi, w = phase_diff(g, g2)
        slope = fit_lowfreq_slope(dphi[row], w[row], FX[row])
        dx_s = -slope / (2 * np.pi)
        dx_p = top_peaks(phase_correlation(g, g2), k=1)[0][0]
        rec_slope.append(dx_s); rec_pc.append(dx_p)
        print(f"  dx={dx:>2}  ->  lowfreq-slope dx_hat={dx_s:6.2f}   "
              f"phase-corr dx_hat={dx_p:>3}")
        ok = np.abs(dphi[row]) < np.pi
        ax[0].plot(FX[row][ok], dphi[row][ok], ".", ms=2, label=f"dx={dx}")
    ax[0].set(xlabel="fx (cycles/px)", ylabel="dphi at ky=0",
              title="ramp along kx; steeper = larger dx; wrap at high fx")
    ax[0].legend(fontsize=6)
    ax[1].plot(shifts, rec_slope, "o-", label="low-freq slope fit")
    ax[1].plot(shifts, rec_pc, "s-", label="phase-corr peak")
    ax[1].plot(shifts, shifts, "k--", lw=1, label="truth")
    ax[1].set(xlabel="true dx (px)", ylabel="recovered dx",
              title="readout linearity (v2 estimators)")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/A_shift_theorem.png", dpi=150)

def part_b():
    print("\n=== B. Local sprite: still vs moving (pixel space) ===")
    rng = np.random.default_rng(0)
    still = to_gray(render_scene(F=30, dx_per_frame=0, seed=1))
    mov = to_gray(render_scene(F=30, dx_per_frame=3, seed=1))
    still = np.clip(still + rng.normal(0, 0.002, still.shape), 0, 1)
    mov = np.clip(mov + rng.normal(0, 0.002, mov.shape), 0, 1)
    for name, v in [("still", still), ("moving", mov)]:
        e = motion_phase_energy(v); c = dphi_pair_correlation(v)
        print(f"  {name:>6}: low-freq |dphi| energy={e:.4f}  "
              f"pair-to-pair dphi corr={c:.3f}")
    for gap in [1, 5, 10]:
        peaks = top_peaks(phase_correlation(mov[0], mov[gap]), k=2)
        print(f"  moving, frames 0 vs {gap}: peaks={peaks}   "
              f"[sprite ~({3*gap},0) dominates via whitening; bg ~(0,0) minor]")

def part_c(model_id):
    print("\n=== C. Same question in VAE latents ===")
    import torch
    vae = load_vae(model_id, device="cuda" if torch.cuda.is_available() else "cpu")
    still = render_scene(F=33, dx_per_frame=0, seed=1)    # (33-1)%4==0
    mov = render_scene(F=33, dx_per_frame=3, seed=1)
    z_s, z_m = encode_video(vae, still).numpy(), encode_video(vae, mov).numpy()
    C, FL = z_m.shape[:2]
    print(f"  latents: C={C}, F_lat={FL}, spatial={z_m.shape[2:]}")
    for name, z in [("still", z_s), ("moving", z_m)]:
        e = np.mean([motion_phase_energy(z[c]) for c in range(C)])
        c_ = np.nanmean([dphi_pair_correlation(z[c]) for c in range(C)])
        print(f"  {name:>6}: latent low-freq energy={e:.4f}  pair corr={c_:.3f}"
              f"{'   [negative = mean-reverting encoder jitter, expected]' if name=='still' else '   [positive = persistent transport]'}")
    peaks = top_peaks(phase_correlation(z_m[0, 0], z_m[0, FL - 1]), k=2)
    print(f"  latent ch0, frames 0 vs {FL-1}: peaks={peaks} "
          f"[sprite expectation ≈ ({round(3*32/8)},0) latent px]")
    r = np.mean([eq4_check(z_m[c, f], z_m[c, f + 1])
                 for c in range(C) for f in range(FL - 1)])
    print(f"  Eq.4 check  corr(|F(delta_z)|, A*|dphi|) = {r:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_latents", action="store_true")
    ap.add_argument("--model_id", default="THUDM/CogVideoX-5b-I2V")
    args = ap.parse_args()
    part_a(); part_b()
    if not args.skip_latents:
        part_c(args.model_id)