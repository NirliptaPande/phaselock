"""Experiment 2a — hard spectral edit as a control mechanism (negative control)

Encode a STILL clip and a MOVING clip of the identical scene. Build hybrids
that keep the still clip's magnitude but take the moving clip's phase:
  (i)  in PIXEL space  -> expect: recognizable, mostly-moving video
                          (phase dominates perception),
  (ii) in LATENT space, full phase swap        -> expect: broken/ghosting,
  (iii) in LATENT space, low-freq phase only   -> expect: still broken
                          (mirrors paper D.3's Low-Freq Phase Injection).
The still-recon vs hybrid gap, with no diffusion loop anywhere, shows the
failure is a property of the LATENT SPACE, not of the sampler.

Run from code/experiments/:  python exp2a_phase_swap.py
"""
import os
import numpy as np
import torch
from common import (render_scene, to_gray, lowfreq_mask, load_vae,
                    encode_video, decode_latents, centroid_track, save_video)

OUT = "outputs/exp2a"; os.makedirs(OUT, exist_ok=True)

def swap_phase_2d(z_mag_src, z_phase_src, lowfreq_only=False, radius=0.4):
    """Per-2D-slice hybrid: |F(mag_src)| * exp(j*angle(F(phase_src)))."""
    Fa = np.fft.fftshift(np.fft.fft2(z_mag_src))
    Fb = np.fft.fftshift(np.fft.fft2(z_phase_src))
    phase = np.angle(Fb)
    if lowfreq_only:
        m = lowfreq_mask(*z_mag_src.shape, radius)
        phase = np.where(m, phase, np.angle(Fa))
    Fh = np.abs(Fa) * np.exp(1j * phase)
    return np.fft.ifft2(np.fft.ifftshift(Fh)).real

def main():
    still = render_scene(F=33, dx_per_frame=0, seed=1)
    mov = render_scene(F=33, dx_per_frame=3, seed=1)

    # -- (i) pixel-space hybrid ------------------------------------------
    gs, gm = to_gray(still), to_gray(mov)
    hyb_px = np.stack([swap_phase_2d(gs[f], gm[f]) for f in range(len(gs))])
    hyb_px = (np.clip(hyb_px, 0, 1) * 255).astype(np.uint8)
    save_video(f"{OUT}/hybrid_pixel.mp4", np.repeat(hyb_px[..., None], 3, -1))

    # -- (ii)/(iii) latent-space hybrids ---------------------------------
    vae = load_vae(device="cuda" if torch.cuda.is_available() else "cpu")
    z_s, z_m = encode_video(vae, still).numpy(), encode_video(vae, mov).numpy()
    C, FL = z_s.shape[:2]

    def latent_hybrid(lowfreq_only):
        zh = np.empty_like(z_s)
        for c in range(C):
            for f in range(FL):
                zh[c, f] = swap_phase_2d(z_s[c, f], z_m[c, f], lowfreq_only)
        return torch.from_numpy(zh)

    vids = {
        "recon_still":  decode_latents(vae, torch.from_numpy(z_s)),
        "recon_moving": decode_latents(vae, torch.from_numpy(z_m)),
        "hybrid_latent_full":    decode_latents(vae, latent_hybrid(False)),
        "hybrid_latent_lowfreq": decode_latents(vae, latent_hybrid(True)),
    }
    for name, v in vids.items():
        save_video(f"{OUT}/{name}.mp4", v)
        tr = centroid_track(v, mode="bright")
        vx = np.nanmean(np.diff(tr[:, 0]))
        rough = float(np.mean(np.abs(np.diff(v.astype(np.float32), axis=0))))
        print(f"  {name:>22}: centroid vx={vx:6.2f} px/frame   "
              f"temporal roughness={rough:7.2f}")
    print("\nRead: pixel hybrid moves and stays legible; latent hybrids "
          "should show high roughness / incoherent motion despite carrying "
          "the 'correct' phase — the hard edit leaves the latent manifold.")

if __name__ == "__main__":
    main()