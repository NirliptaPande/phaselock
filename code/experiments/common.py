"""Shared utilities for the phase/position experiments.

Conventions
-----------
- Pixel videos: uint8 [F, H, W, 3]. Analysis on grayscale float in [0, 1].
- VAE latents (encode/decode helpers): [C=16, F_lat, H/8, W/8].
- Pipeline latents (denoising loop, exp2b): [B, F_lat, C, H/8, W/8], frames on dim 1.
- np.fft sign convention: f(x - d) <-> F(k) * exp(-2j*pi*k*d), so for a
  RIGHTWARD shift by d px, angle(F2 * conj(F1)) = -2*pi*fx*d.
- CogVideoX VAE wants (F - 1) % 4 == 0 (temporal compression 4, first frame kept).
"""
import numpy as np
import torch

# ------------------------------------------------------------------ scenes

def _texture(h, w, rng, smooth):
    import cv2
    t = rng.standard_normal((h, w)).astype(np.float32)
    t = cv2.GaussianBlur(t, (0, 0), smooth)
    return (t - t.min()) / (np.ptp(t) + 1e-8)

def render_scene(F=33, H=256, W=384, dx_per_frame=0.0, r=28, seed=0,
                 global_shift=False):
    """Textured background + textured disc sprite ("the swan").

    dx_per_frame > 0 moves the sprite RIGHT. global_shift=True rolls the whole
    frame instead (circular -> shift theorem holds exactly, no leakage).
    Returns uint8 [F, H, W, 3].
    """
    rng = np.random.default_rng(seed)
    bg = 0.30 + 0.35 * _texture(H, W, rng, smooth=13)
    spr = 0.60 + 0.40 * _texture(2 * r, 2 * r, rng, smooth=3)
    yy, xx = np.mgrid[-r:r, -r:r]
    disc = (xx ** 2 + yy ** 2) < r ** 2
    cy, cx0 = H // 2, W // 5
    frames, base = [], None
    for f in range(F):
        if global_shift:
            if base is None:
                base = bg.copy()
                view = base[cy - r:cy + r, cx0 - r:cx0 + r]
                view[disc] = spr[disc]
            img = np.roll(base, int(round(dx_per_frame * f)), axis=1)
        else:
            img = bg.copy()
            cx = int(np.clip(round(cx0 + dx_per_frame * f), r, W - r - 1))
            view = img[cy - r:cy + r, cx - r:cx + r]
            view[disc] = spr[disc]
        frames.append(img)
    v = (np.clip(np.stack(frames), 0, 1) * 255).astype(np.uint8)
    return np.repeat(v[..., None], 3, axis=-1)

def to_gray(video_u8):
    return video_u8[..., :3].astype(np.float32).mean(-1) / 255.0

def save_video(path, video_u8, fps=8):
    import imageio
    imageio.mimwrite(path, video_u8, fps=fps, quality=8)

# ------------------------------------------------------------- fft / phase

def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(x, axes=(-2, -1)), axes=(-2, -1))

def freq_grid(h, w):
    fy = np.fft.fftshift(np.fft.fftfreq(h))   # cycles/px in [-0.5, 0.5)
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    return np.meshgrid(fx, fy)                # FX, FY

def lowfreq_mask(h, w, radius=0.4):
    FX, FY = freq_grid(h, w)
    return np.sqrt((FX / 0.5) ** 2 + (FY / 0.5) ** 2) < radius  # paper's <0.4

def phase_diff(a, b, window=False):
    """angle(F(b)*conj(F(a))) (centered) and cross magnitude as weight.
    a = frame t, b = frame t+1. window=True: Hann + mean removal, for
    non-periodic content (real sprites, latents)."""
    if window:
        h, w = a.shape
        w2 = np.outer(np.hanning(h), np.hanning(w))
        a = (a - a.mean()) * w2
        b = (b - b.mean()) * w2
    Fa, Fb = fft2c(a), fft2c(b)
    cross = Fb * np.conj(Fa)
    return np.angle(cross), np.abs(cross)

def phase_correlation(a, b):
    """Normalized cross-power surface: peaks sit at the displacements present."""
    r = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    r /= np.abs(r) + 1e-9
    return np.abs(np.fft.ifft2(r))

def top_peaks(surf, k=2, exclude=4):
    s, out = surf.copy(), []
    H, W = s.shape
    yy, xx = np.ogrid[:H, :W]
    for _ in range(k):
        iy, ix = np.unravel_index(np.argmax(s), s.shape)
        dy = iy - H if iy > H // 2 else iy
        dx = ix - W if ix > W // 2 else ix
        out.append((dx, dy, float(s[iy, ix])))
        d2 = (np.minimum(abs(yy - iy), H - abs(yy - iy)) ** 2 +
              np.minimum(abs(xx - ix), W - abs(xx - ix)) ** 2)
        s[d2 <= exclude ** 2] = 0
    return out  # [(dx, dy, score), ...]

def fit_kx_slope(dphi_row, weight_row, fx_row, max_abs=np.pi / 2):
    """Weighted LS slope of dphi vs fx on the ky=0 row, wrap-safe bins only.
    Recovered shift: dx_hat = -slope / (2*pi)."""
    ok = (np.abs(dphi_row) < max_abs) & (np.abs(fx_row) > 0)
    if ok.sum() < 4:
        return np.nan
    w, x, y = weight_row[ok], fx_row[ok], dphi_row[ok]
    return float(np.sum(w * x * y) / (np.sum(w * x * x) + 1e-12))

# ------------------------------------------------------------ phase metrics

def motion_phase_energy(video, radius=0.4, window=True):
    """Weighted mean |dphi| over low-freq bins, averaged over frame pairs.
    ~0 for a static scene, grows with motion."""
    m = lowfreq_mask(*video.shape[-2:], radius)
    vals = []
    for f in range(len(video) - 1):
        d, w = phase_diff(video[f], video[f + 1], window)
        vals.append(np.sum(w[m] * np.abs(d[m])) / (np.sum(w[m]) + 1e-9))
    return float(np.mean(vals))

def dphi_pair_correlation(video, radius=0.4, window=True):
    """Pearson corr between consecutive dphi maps (low-freq bins).
    Steady motion -> same ramp every pair -> ~1. Static -> noise -> ~0."""
    m = lowfreq_mask(*video.shape[-2:], radius)
    prev, vals = None, []
    for f in range(len(video) - 1):
        d, _ = phase_diff(video[f], video[f + 1], window)
        cur = d[m].ravel()
        if prev is not None and np.std(cur) > 1e-9 and np.std(prev) > 1e-9:
            vals.append(np.corrcoef(prev, cur)[0, 1])
        prev = cur
    return float(np.mean(vals)) if vals else np.nan

def eq4_check(z_prev, z_next, radius=0.4):
    """Paper Eq. 4: |F(delta)| ≈ A * |dphi|. Returns Pearson r between the two
    sides over low-freq bins. z_*: 2D float arrays (one latent channel-frame)."""
    m = lowfreq_mask(*z_prev.shape, radius)
    Fa, Fb = fft2c(z_prev), fft2c(z_next)
    lhs = np.abs(Fb - Fa)[m]
    A = 0.5 * (np.abs(Fa) + np.abs(Fb))
    dphi = np.angle(Fb * np.conj(Fa))
    rhs = (A * np.abs(dphi))[m]
    return float(np.corrcoef(lhs.ravel(), rhs.ravel())[0, 1])

# ----------------------------------------------------------------- the VAE

def load_vae(model_id="THUDM/CogVideoX-5b-I2V", device="cuda",
             dtype=torch.bfloat16):
    from diffusers import AutoencoderKLCogVideoX
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype).to(device).eval()
    vae.enable_tiling()
    return vae

@torch.no_grad()
def encode_video(vae, video_u8):
    """uint8 [F, H, W, 3] -> float32 latents [C, F_lat, H/8, W/8].
    No scaling_factor applied; decode_latents is the exact inverse convention,
    so encode->decode round-trips are self-consistent (phase analysis is
    invariant to a global scale anyway)."""
    x = torch.from_numpy(video_u8).float() / 127.5 - 1.0
    x = x.permute(3, 0, 1, 2)[None].to(vae.device, vae.dtype)  # [1,3,F,H,W]
    return vae.encode(x).latent_dist.mode()[0].float().cpu()

@torch.no_grad()
def decode_latents(vae, z):
    """float [C, F_lat, H/8, W/8] -> uint8 [F, H, W, 3]"""
    x = vae.decode(z[None].to(vae.device, vae.dtype)).sample[0]
    x = ((x.float().clamp(-1, 1) + 1) * 127.5).byte().cpu()
    return x.permute(1, 2, 3, 0).numpy()

# ---------------------------------------------------------- motion metrics

def centroid_track(video_u8, mode="edges"):
    """Per-frame centroid, paper-style (Canny + image moments, Appx B.2) or
    'bright' (top-decile brightness) which is more robust on synthetic scenes.
    Returns float [F, 2] = (cx, cy)."""
    import cv2
    pts = []
    for fr in video_u8:
        g = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
        if mode == "edges":
            b = cv2.Canny(g, 60, 140)
        else:
            b = (g > np.quantile(g, 0.90)).astype(np.uint8)
        M = cv2.moments(b, binaryImage=True)
        pts.append((M["m10"] / M["m00"], M["m01"] / M["m00"])
                   if M["m00"] > 0 else (np.nan, np.nan))
    return np.asarray(pts)

def mean_flow_x(video_u8):
    """Mean horizontal Farneback flow (px/frame). RAFT is the paper's choice;
    Farneback is a dependency-free stand-in with the same sign convention."""
    import cv2
    prev = cv2.cvtColor(video_u8[0], cv2.COLOR_RGB2GRAY)
    xs = []
    for fr in video_u8[1:]:
        g = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
        fl = cv2.calcOpticalFlowFarneback(prev, g, None,
                                          0.5, 3, 21, 3, 5, 1.1, 0)
        xs.append(float(fl[..., 0].mean()))
        prev = g
    return float(np.mean(xs))