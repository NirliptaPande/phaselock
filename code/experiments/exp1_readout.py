"""Experiment 1 — is position written into phase? (readout direction)

A. Shift theorem, exact: roll a whole frame by dx, verify dphi is a linear
   ramp along kx only (flat along ky), slope -2*pi*dx, and recover dx from
   the slope. Sweep dx to see linearity and the wrap point.
B. Local motion: sprite on static background. Phase correlation should show
   TWO peaks (background at (0,0), sprite at (dx,0)); low-freq phase metrics
   should separate still vs moving.
C. Same in CogVideoX VAE latents: the relation must survive encoding, per
   channel, low-freq only. Plus the paper's Eq. 4 check:
   |F(delta_z)| correlates with A*|dphi|.

Expected: A) dx_hat ≈ dx (linearity until |2*pi*fx*dx| wraps at high fx);
B) moving scene: second peak marches right, energy/corr >> still scene;
C) same ordering in latents (noisier), eq4 r strongly positive (> ~0.6).

Run from code/experiments/:  python exp1_readout.py [--skip_latents]
"""
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from common import (render_scene, to_gray, phase_diff, phase_correlation,
                    top_peaks, fit_kx_slope, freq_grid, lowfreq_mask,
                    motion_phase_energy, dphi_pair_correlation, eq4_check,
                    load_vae, encode_video)

OUT = "outputs/exp1"; os.makedirs(OUT, exist_ok=True)

def part_a():
    print("\n=== A. Global shift (exact shift theorem) ===")
    g = to_gray(render_scene(F=1, H=256, W=384, seed=0))[0]
    H, W = g.shape
    FX, _ = freq_grid(H, W)
    row = H // 2                       # ky = 0 after fftshift
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    recovered = []
    shifts = [0, 1, 2, 4, 8, 16, 24, 32]
    for dx in shifts:
        g2 = np.roll(g, dx, axis=1)    # circular -> no window needed
        dphi, w = phase_diff(g, g2)
        slope = fit_kx_slope(dphi[row], w[row], FX[row])
        dx_hat = -slope / (2 * np.pi) if np.isfinite(slope) else np.nan
        recovered.append(dx_hat)
        print(f"  dx={dx:>2}  ->  dx_hat={dx_hat:6.2f}")
        ok = np.abs(dphi[row]) < np.pi
        ax[0].plot(FX[row][ok], dphi[row][ok], ".", ms=2, label=f"dx={dx}")
    ax[0].set(xlabel="fx (cycles/px)", ylabel="dphi at ky=0",
              title="ramp along kx; steeper = larger dx; wrap at high fx")
    ax[0].legend(fontsize=6)
    ax[1].plot(shifts, recovered, "o-"); ax[1].plot(shifts, shifts, "k--", lw=1)
    ax[1].set(xlabel="true dx (px)", ylabel="recovered dx",
              title="readout linearity")
    fig.tight_layout(); fig.savefig(f"{OUT}/A_shift_theorem.png", dpi=150)

def part_b():
    print("\n=== B. Local sprite: still vs moving (pixel space) ===")
    still = to_gray(render_scene(F=30, dx_per_frame=0, seed=1))
    mov = to_gray(render_scene(F=30, dx_per_frame=3, seed=1))
    for name, v in [("still", still), ("moving", mov)]:
        e = motion_phase_energy(v); c = dphi_pair_correlation(v)
        print(f"  {name:>6}: low-freq |dphi| energy={e:.4f}  "
              f"pair-to-pair dphi corr={c:.3f}")
    # two populations in the phase-correlation surface, growing gap
    for gap in [1, 5, 10]:
        peaks = top_peaks(phase_correlation(mov[0], mov[gap]), k=2)
        print(f"  moving, frames 0 vs {gap}: peaks (dx,dy,score) = {peaks}"
              f"   [expect ~(0,0) bg and ~({3*gap},0) sprite]")

def part_c(model_id):
    print("\n=== C. Same question in VAE latents ===")
    import torch
    vae = load_vae(model_id, device="cuda" if torch.cuda.is_available() else "cpu")
    still = render_scene(F=33, dx_per_frame=0, seed=1)   # (33-1)%4==0
    mov = render_scene(F=33, dx_per_frame=3, seed=1)
    z_s, z_m = encode_video(vae, still).numpy(), encode_video(vae, mov).numpy()
    C, FL = z_m.shape[:2]
    print(f"  latents: C={C}, F_lat={FL}, spatial={z_m.shape[2:]} "
          f"(3 px/frame -> ~1.5 latent px per latent frame)")
    for name, z in [("still", z_s), ("moving", z_m)]:
        e = np.mean([motion_phase_energy(z[c]) for c in range(C)])
        c_ = np.nanmean([dphi_pair_correlation(z[c]) for c in range(C)])
        print(f"  {name:>6}: latent low-freq energy={e:.4f}  pair corr={c_:.3f}")
    # long-gap phase correlation on a latent channel: sprite peak in latent px
    peaks = top_peaks(phase_correlation(z_m[0, 0], z_m[0, FL - 1]), k=2)
    print(f"  latent ch0, frames 0 vs {FL-1}: peaks={peaks} "
          f"[sprite expectation ≈ ({round(3*32/8)},0) latent px]")
    r = np.mean([eq4_check(z_m[c, f], z_m[c, f + 1])
                 for c in range(C) for f in range(FL - 1)])
    print(f"  Eq.4 check  corr(|F(delta_z)|, A*|dphi|) = {r:.3f}  "
          f"[the bridge to Latent Delta Guidance]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_latents", action="store_true")
    ap.add_argument("--model_id", default="THUDM/CogVideoX-5b-I2V")
    args = ap.parse_args()
    part_a(); part_b()
    if not args.skip_latents:
        part_c(args.model_id)