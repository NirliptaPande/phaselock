"""Experiment 2b — is phase(-delta) a CONTROL KNOB? (interventional direction)

Alg. 1 injected via diffusers callback_on_step_end, prior from three sources:

  natural   : model's own 2-step run, same prompt        (paper reproduction)
  cross     : 2-step run of a DIFFERENT, motion-explicit prompt, injected
              into an ambiguous-prompt generation        (does motion transfer?)
  synthetic : NO generation at all — constant-velocity prior built by rolling
              the first latent frame right by d latent px per latent frame.
  sweep     : synthetic prior, dose–response over d x lambda_0. If output
              horizontal velocity rises monotonically with d and with
              lambda_0, the relation is a steering wheel, not a speedometer.

Seam into the repo instead of the callback: wherever code/phaselock/guidance.py
assigns M_prior from the few-step latents, substitute make_synthetic_prior()'s
tensor — everything else in their pipeline stays untouched.

Calibration: 1 latent px / latent frame ≈ 8 px per ~4 pixel frames
≈ 2 px/frame on screen (spatial 8x, temporal 4x). Layouts must roughly
overlap for cross-priors: the delta is a dipole AT the object's location.

Runtime: one 50-step CogVideoX-5B-I2V run is minutes even on an H100;
the default sweep is 8 runs. Use --quick / --full_steps 30 to iterate.
Verify two version-dependent assumptions against your diffusers install:
(1) loop latents are [B, F_lat, C, H', W'] with frames on dim 1;
(2) output_type='latent' returns those latents in .frames.

Run from code/experiments/:  python exp2b_control.py --mode sweep
"""
import argparse, csv, os
import numpy as np
import torch
from PIL import Image
from common import render_scene, save_video, centroid_track, mean_flow_x

OUT = "outputs/exp2b"; os.makedirs(OUT, exist_ok=True)
MODEL = "THUDM/CogVideoX-5b-I2V"
H, W, F = 480, 720, 49          # -> latents [1, 13, 16, 60, 90]

P_AMBIG = ("a small white textured ball resting on a wooden table, "
           "static camera, photorealistic")
P_MOTION = ("a small white textured ball sliding steadily from left to right "
            "across a wooden table, static camera, photorealistic")

def load_pipe():
    from diffusers import CogVideoXImageToVideoPipeline
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe

def fixed_noise(pipe, seed):
    """Sample z_T ONCE; pass a clone to every call so stage 1 and stage 2
    share the initial noise exactly (paper: 'reuse the same zT')."""
    tf = getattr(pipe, "vae_scale_factor_temporal", 4)
    sf = getattr(pipe, "vae_scale_factor_spatial", 8)
    c = pipe.vae.config.latent_channels           # 16
    shape = (1, (F - 1) // tf + 1, c, H // sf, W // sf)
    g = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32)

def delta(z):                      # z: [B, F_lat, C, h, w]
    return z[:, 1:] - z[:, :-1]

@torch.no_grad()
def few_step_latents(pipe, image, prompt, noise, steps=2):
    out = pipe(image=image, prompt=prompt, height=H, width=W, num_frames=F,
               num_inference_steps=steps, guidance_scale=6.0,
               latents=noise.clone().to(pipe.dtype), output_type="latent")
    return out.frames.float().cpu()             # [B, F_lat, C, h, w]

def make_synthetic_prior(z_ref_frame, f_lat, d):
    """Constant-velocity prior: content-matched dipole train moving RIGHT.
    z_ref_frame: [B, C, h, w] (use the few-step run's own first latent frame
    so the delta sits where the object actually is)."""
    z = torch.stack([torch.roll(z_ref_frame, shifts=d * f, dims=-1)
                     for f in range(f_lat)], dim=1)
    return delta(z)                             # [B, f_lat-1, C, h, w]

@torch.no_grad()
def guided_run(pipe, image, prompt, noise, m_prior=None, lam0=0.05,
               k_start=0, k_end=None, steps=50, tag="run"):
    k_end = steps // 2 if k_end is None else k_end

    def cb(p, i, t, kw):
        z = kw["latents"]
        if m_prior is not None and lam0 > 0 and k_start <= i < k_end:
            lam = lam0 * (1 - (i - k_start) / (k_end - k_start))
            mp = m_prior.to(z.device, z.dtype)
            g = mp - (z[:, 1:] - z[:, :-1])     # G = M_prior - M_current
            z = z.clone()
            z[:, 1:] = z[:, 1:] + lam * g       # frame 0 = anchor, untouched
        return {"latents": z}

    out = pipe(image=image, prompt=prompt, height=H, width=W, num_frames=F,
               num_inference_steps=steps, guidance_scale=6.0,
               latents=noise.clone().to(pipe.dtype),
               callback_on_step_end=cb,
               callback_on_step_end_tensor_inputs=["latents"])
    video = np.stack([np.asarray(fr) for fr in out.frames[0]])
    save_video(f"{OUT}/{tag}.mp4", video)
    return video

def report(tag, video):
    tr = centroid_track(video, mode="edges")    # paper-style (Appx B.2)
    vx = np.nanmean(np.diff(tr[:, 0]))
    fx = mean_flow_x(video)
    print(f"  {tag:>28}: centroid vx={vx:6.2f} px/frame   mean flow_x={fx:6.2f}")
    return vx, fx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sweep",
                    choices=["natural", "cross", "synthetic", "sweep"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full_steps", type=int, default=50)
    ap.add_argument("--lam0", type=float, default=0.05)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    image = Image.fromarray(render_scene(F=1, H=H, W=W, seed=0)[0])
    image.save(f"{OUT}/input.png")
    pipe = load_pipe()
    noise = fixed_noise(pipe, args.seed)

    if args.mode == "natural":
        z_few = few_step_latents(pipe, image, P_MOTION, noise)
        mp = delta(z_few)
        report("baseline (no guidance)",
               guided_run(pipe, image, P_MOTION, noise, None, 0,
                          steps=args.full_steps, tag="nat_base"))
        report("PhaseLock (own prior)",
               guided_run(pipe, image, P_MOTION, noise, mp, args.lam0,
                          steps=args.full_steps, tag="nat_guided"))

    elif args.mode == "cross":
        mp = delta(few_step_latents(pipe, image, P_MOTION, noise))
        report("ambiguous, no guidance",
               guided_run(pipe, image, P_AMBIG, noise, None, 0,
                          steps=args.full_steps, tag="cross_base"))
        report("ambiguous + motion prior",
               guided_run(pipe, image, P_AMBIG, noise, mp, args.lam0,
                          steps=args.full_steps, tag="cross_guided"))

    elif args.mode == "synthetic":
        z_few = few_step_latents(pipe, image, P_AMBIG, noise)
        mp = make_synthetic_prior(z_few[:, 0], z_few.shape[1], args.d)
        report("no guidance",
               guided_run(pipe, image, P_AMBIG, noise, None, 0,
                          steps=args.full_steps, tag="syn_base"))
        report(f"synthetic prior d={args.d}",
               guided_run(pipe, image, P_AMBIG, noise, mp, args.lam0,
                          steps=args.full_steps, tag=f"syn_d{args.d}"))

    else:  # sweep — the dose–response curve
        z_few = few_step_latents(pipe, image, P_AMBIG, noise)
        z0, f_lat = z_few[:, 0], z_few.shape[1]
        ds = [0, 2] if args.quick else [0, 1, 2, 3]
        lams = [0.05] if args.quick else [0.02, 0.05, 0.10]
        rows = [("d", "lam0", "vx_centroid", "vx_flow")]
        for lam in lams:
            for d in ds:
                mp = make_synthetic_prior(z0, f_lat, d) if d > 0 else None
                v = guided_run(pipe, image, P_AMBIG, noise, mp,
                               lam if d > 0 else 0.0, steps=args.full_steps,
                               tag=f"sweep_d{d}_l{lam}")
                vx, fx = report(f"d={d} lam0={lam}", v)
                rows.append((d, lam, vx, fx))
        with open(f"{OUT}/dose_response.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print("\nRead: vx should rise with d at fixed lam0, and with lam0 at "
              "fixed d, holding at ~0 for the d=0 controls. Sanity: "
              "d=1 latent px/latent frame ≈ 2 px/frame on screen.")

if __name__ == "__main__":
    main()