"""Experiment 2a (v2) — hard spectral edit: matched vs mismatched donors

Round-1 finding (yours): with donor and recipient being the SAME scene,
magnitudes nearly coincide, so |F(still)|*exp(j*phase(moving)) is almost the
honest moving latent — the swap barely left the manifold, and it decoded
well. Easy mode. v2 runs both conditions:

  matched    : still(seed 1) vs moving(seed 1)            (round-1 replica)
  mismatched : still(seed 1) vs moving(seed 7), whole scene rolled +80 px
               -> different background, sprite texture, AND start position.

New measurements:
- rel_err(hybrid, honest_moving) and rel_err(hybrid, honest_still): how far
  the chimera sits from each honest latent. Matched: expect full-swap hybrid
  ~close to moving (explains round 1). Mismatched: far from both.
- roughness v2: frames 8+ only (skip the causal 3D-VAE warm-up you saw as an
  early ghost) and a background-only strip (rows above the sprite lane),
  so real motion is not counted as artifact.

Prediction: pixel hybrid stays semi-legible in both conditions (mismatched =
the classic phase-dominance demo in its proper form: moving layout wearing
still texture statistics); latent hybrids are fine when matched, degrade
hard when mismatched — dose of mismatch -> dose of breakage. Before blaming
any early-frame ghost on the swap, compare recon_moving's first frames.

Run from code/experiments/:  python exp2a_phase_swap.py
"""
import os
import numpy as np
import torch
from common import (render_scene, to_gray, lowfreq_mask, load_vae,
                    encode_video, decode_latents, centroid_track, save_video)

OUT = "outputs/exp2a"; os.makedirs(OUT, exist_ok=True)
SPRITE_LANE = (100, 156)      # rows the disc occupies at H=256 (cy=128, r=28)
BG_ROWS = (8, 92)             # background-only strip, clear of the lane

def swap_phase_2d(z_mag_src, z_phase_src, lowfreq_only=False, radius=0.4):
    Fa = np.fft.fftshift(np.fft.fft2(z_mag_src))
    Fb = np.fft.fftshift(np.fft.fft2(z_phase_src))
    phase = np.angle(Fb)
    if lowfreq_only:
        m = lowfreq_mask(*z_mag_src.shape, radius)
        phase = np.where(m, phase, np.angle(Fa))
    Fh = np.abs(Fa) * np.exp(1j * phase)
    return np.fft.ifft2(np.fft.ifftshift(Fh)).real

def latent_hybrid(z_s, z_m, lowfreq_only):
    zh = np.empty_like(z_s)
    C, FL = z_s.shape[:2]
    for c in range(C):
        for f in range(FL):
            zh[c, f] = swap_phase_2d(z_s[c, f], z_m[c, f], lowfreq_only)
    return zh

def rel_err(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))

def roughness(video_u8, skip=8, rows=BG_ROWS):
    v = video_u8[skip:, rows[0]:rows[1]].astype(np.float32)
    return float(np.mean(np.abs(np.diff(v, axis=0))))

def run_condition(vae, still, mov, name):
    outdir = f"{OUT}/{name}"; os.makedirs(outdir, exist_ok=True)
    print(f"\n=== condition: {name} ===")

    gs, gm = to_gray(still), to_gray(mov)
    hyb_px = np.stack([swap_phase_2d(gs[f], gm[f]) for f in range(len(gs))])
    hyb_px = (np.clip(hyb_px, 0, 1) * 255).astype(np.uint8)
    save_video(f"{outdir}/hybrid_pixel.mp4",
               np.repeat(hyb_px[..., None], 3, -1))

    z_s = encode_video(vae, still).numpy()
    z_m = encode_video(vae, mov).numpy()
    zh_full = latent_hybrid(z_s, z_m, False)
    zh_low = latent_hybrid(z_s, z_m, True)
    for hname, zh in [("full", zh_full), ("lowfreq", zh_low)]:
        print(f"  latent proximity, hybrid_{hname}: "
              f"rel_err vs moving={rel_err(zh, z_m):.3f}  "
              f"vs still={rel_err(zh, z_s):.3f}")

    vids = {
        "recon_still":          torch.from_numpy(z_s),
        "recon_moving":         torch.from_numpy(z_m),
        "hybrid_latent_full":   torch.from_numpy(zh_full),
        "hybrid_latent_lowfreq": torch.from_numpy(zh_low),
    }
    for vname, z in vids.items():
        v = decode_latents(vae, z)
        save_video(f"{outdir}/{vname}.mp4", v)
        tr = centroid_track(v, mode="bright")
        vx = np.nanmean(np.diff(tr[:, 0]))
        print(f"  {vname:>22}: vx={vx:6.2f} px/frame   "
              f"bg-roughness(frames>=8)={roughness(v):7.3f}")

def main():
    vae = load_vae(device="cuda" if torch.cuda.is_available() else "cpu")
    still = render_scene(F=33, dx_per_frame=0, seed=1)

    mov_matched = render_scene(F=33, dx_per_frame=3, seed=1)
    run_condition(vae, still, mov_matched, "matched")

    mov_mismatched = np.roll(render_scene(F=33, dx_per_frame=3, seed=7),
                             80, axis=2)      # new textures + new start x
    run_condition(vae, still, mov_mismatched, "mismatched")

    print("\nRead: matched full-swap should sit close to the honest moving "
          "latent (small rel_err) — that's why round 1 looked good. "
          "Mismatched hybrids should sit far from both honest latents and "
          "decode visibly worse: dose of spectral mismatch -> dose of "
          "breakage. That reframed claim is the real negative control "
          "under exp2b's soft guidance.")

if __name__ == "__main__":
    main()