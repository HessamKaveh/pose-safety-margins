"""Image corruptions that a robot camera actually meets, applied to the full image before the person crop.

Each corruption has 3 severities. All randomness is seeded per (instance, corruption, severity), so every
run sees exactly the same corrupted images.
"""
import cv2
import numpy as np

CORRUPTIONS = ["motion_blur", "gaussian_noise", "low_light", "jpeg", "occluder"]


def motion_blur(img, s, rng):
    k = [7, 15, 25][s - 1]
    kern = np.zeros((k, k), np.float32)
    kern[k // 2] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kern = cv2.warpAffine(kern, M, (k, k))
    kern /= kern.sum()
    return cv2.filter2D(img, -1, kern)


def gaussian_noise(img, s, rng):
    sd = [0.04, 0.08, 0.14][s - 1] * 255
    return np.clip(img.astype(np.float32) + rng.normal(0, sd, img.shape), 0, 255).astype(np.uint8)


def low_light(img, s, rng):
    # darker exposure (gamma + gain) with the shot/read noise that comes with it, as in a dim workcell
    gain, gamma, sd = [(0.5, 1.5, 0.02), (0.3, 2.0, 0.035), (0.18, 2.5, 0.05)][s - 1]
    x = gain * (img.astype(np.float32) / 255.0) ** gamma
    x = x + rng.normal(0, sd, img.shape)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def jpeg(img, s, rng):
    q = [20, 10, 5][s - 1]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def occluder(img, s, rng, box, patch_src=None):
    """Paste a random texture patch covering 10/25/40 % of the person box (a tool, a part, another arm)."""
    frac = [0.10, 0.25, 0.40][s - 1]
    x, y, w, h = box
    a = frac * w * h
    r = rng.uniform(0.5, 2.0)
    ow, oh = int(min(w, np.sqrt(a * r))), int(min(h, np.sqrt(a / r)))
    if ow < 2 or oh < 2:
        return img
    ox = int(x + rng.uniform(0, max(1, w - ow))); oy = int(y + rng.uniform(0, max(1, h - oh)))
    H, W = img.shape[:2]
    ox, oy = min(max(ox, 0), W - 1), min(max(oy, 0), H - 1)
    ow, oh = min(ow, W - ox), min(oh, H - oy)
    out = img.copy()
    if patch_src is not None and patch_src.shape[0] > oh and patch_src.shape[1] > ow:
        py, px = rng.integers(0, patch_src.shape[0] - oh), rng.integers(0, patch_src.shape[1] - ow)
        out[oy:oy + oh, ox:ox + ow] = patch_src[py:py + oh, px:px + ow]
    else:
        out[oy:oy + oh, ox:ox + ow] = rng.integers(0, 255, 3)
    return out


def apply(img, name, severity, seed, box=None, patch_src=None):
    if name == "clean":
        return img
    rng = np.random.default_rng(seed)
    if name == "occluder":
        return occluder(img, severity, rng, box, patch_src)
    return globals()[name](img, severity, rng)
