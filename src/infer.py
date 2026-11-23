"""Run the pose model on every annotated COCO val2017 person, clean and corrupted, and cache per-joint outputs.

For each person i and joint j we store
  pred[i, j]   predicted (x, y) in image pixels (flip test, quarter-pixel refinement)
  conf[i, j]   heatmap peak value
  sigma[i, j]  the heatmap's own spread around the peak, in image pixels (second moment, target blur removed)
  gt[i, j]     ground-truth (x, y, v) with v = 0 not labelled, 1 labelled but occluded, 2 visible
plus the GT box, the image id and how many annotated people share the image.

    python src/infer.py                      # clean + 5 corruptions x 3 severities -> cache/outputs_*.npz
"""
import argparse
import os
import zlib

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

import corruptions
from model import SimpleBaseline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCO_ROOT = os.environ.get("COCO_ROOT", os.path.expanduser("~/datasets/portfolio/coco"))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
IN_WH, HM_WH, TARGET_SIGMA = (192, 256), (48, 64), 2.0


def box_to_center_scale(x, y, w, h, aspect=192 / 256, pad=1.25):
    c = np.array([x + w * 0.5, y + h * 0.5], np.float32)
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    return c, np.array([w, h], np.float32) * pad


def affine(center, size, out_wh, inv=False):
    """Axis-aligned crop of (center, size) to out_wh (same construction as SimpleBaseline, no rotation)."""
    src = np.float32([center, center + [0, -0.5 * size[0]], center + [-0.5 * size[0], -0.5 * size[0]]])
    c = np.array(out_wh, np.float32) * 0.5
    dst = np.float32([c, c + [0, -0.5 * out_wh[0]], c + [-0.5 * out_wh[0], -0.5 * out_wh[0]]])
    return cv2.getAffineTransform(dst, src) if inv else cv2.getAffineTransform(src, dst)


class ValPersons(Dataset):
    def __init__(self, corruption="clean", severity=0):
        self.coco = COCO(os.path.join(COCO_ROOT, "annotations", "person_keypoints_val2017.json"))
        self.img_dir = os.path.join(COCO_ROOT, "val2017")
        self.corruption, self.severity = corruption, severity
        self.items = []
        for img_id in sorted(self.coco.getImgIds(catIds=[1])):
            info = self.coco.imgs[img_id]
            anns = [a for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
                    if a["num_keypoints"] > 0]
            for a in anns:
                x, y, w, h = a["bbox"]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(info["width"] - 1, x1 + max(0, w - 1)), min(info["height"] - 1, y1 + max(0, h - 1))
                if a["area"] <= 0 or x2 <= x1 or y2 <= y1:
                    continue
                c, s = box_to_center_scale(x1, y1, x2 - x1, y2 - y1)
                self.items.append(dict(img_id=img_id, file=info["file_name"], center=c, size=s,
                                       box=np.float32([x1, y1, x2 - x1, y2 - y1]), ann_id=a["id"],
                                       area=float(a["area"]), n_people=len(anns),
                                       kpts=np.array(a["keypoints"], np.float32).reshape(17, 3)))
        self.files = sorted({it["file"] for it in self.items})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.imread(os.path.join(self.img_dir, it["file"]))
        seed = zlib.crc32(f"{it['ann_id']}-{self.corruption}-{self.severity}".encode())
        patch = None
        if self.corruption == "occluder":   # texture from another (deterministic) val image
            patch = cv2.imread(os.path.join(self.img_dir, self.files[seed % len(self.files)]))
        img = corruptions.apply(img, self.corruption, self.severity, seed, box=it["box"], patch_src=patch)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        M = affine(it["center"], it["size"], IN_WH)
        crop = cv2.warpAffine(img, M, IN_WH, flags=cv2.INTER_LINEAR)
        x = torch.from_numpy(((crop.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy())
        return x, i


def flip_back(hm):
    hm = hm.flip(3)
    idx = list(range(17))
    for a, b in FLIP_PAIRS:
        idx[a], idx[b] = b, a
    hm = hm[:, idx]
    hm[..., 1:] = hm[..., :-1].clone()
    return hm


def decode(hm):
    """Peak location (quarter-pixel refined), peak value and spread of each heatmap, all in heatmap pixels."""
    B, J, H, W = hm.shape
    flat = hm.reshape(B, J, -1)
    idx, mx = flat.argmax(2), flat.max(2)
    xs, ys = (idx % W).astype(np.float32), (idx // W).astype(np.float32)
    spread = np.zeros((B, J), np.float32)
    gy, gx = np.mgrid[0:H, 0:W]
    for b in range(B):
        for j in range(J):
            x, y = int(xs[b, j]), int(ys[b, j])
            if 1 < x < W - 1 and 1 < y < H - 1:
                xs[b, j] += 0.25 * np.sign(hm[b, j, y, x + 1] - hm[b, j, y, x - 1])
                ys[b, j] += 0.25 * np.sign(hm[b, j, y + 1, x] - hm[b, j, y - 1, x])
            # second moment of the positive heatmap mass around the peak; the training targets were Gaussians
            # with sigma 2, so that blur is removed to get the model's own localisation spread
            p = np.clip(hm[b, j], 0, None)
            tot = p.sum()
            if tot <= 1e-8:
                spread[b, j] = float(max(H, W))
                continue
            var = (p * ((gx - xs[b, j]) ** 2 + (gy - ys[b, j]) ** 2)).sum() / tot / 2.0  # per-axis variance
            spread[b, j] = np.sqrt(max(var - TARGET_SIGMA ** 2, 0.25))
    return np.stack([xs, ys], 2), mx, spread


@torch.no_grad()
def run(model, ds, dev, workers=12):
    N = len(ds)
    pred = np.zeros((N, 17, 2), np.float32); conf = np.zeros((N, 17), np.float32)
    sigma = np.zeros((N, 17), np.float32)
    for x, idx in DataLoader(ds, batch_size=64, num_workers=workers):
        x = x.to(dev).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            hm = model(x).float()
            hm = 0.5 * (hm + flip_back(model(x.flip(3)).float()))
        c, m, s = decode(hm.cpu().numpy())
        c *= IN_WH[0] / HM_WH[0]
        for b, i in enumerate(idx.tolist()):
            it = ds.items[i]
            Minv = affine(it["center"], it["size"], IN_WH, inv=True)
            pred[i] = c[b] @ Minv[:, :2].T + Minv[:, 2]
            conf[i] = m[b]
            sigma[i] = s[b] * it["size"][0] / HM_WH[0]      # heatmap px -> image px
    return pred, conf, sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=None, help="e.g. clean motion_blur:2")
    args = ap.parse_args()
    conds = args.conditions or ["clean"] + [f"{c}:{s}" for c in corruptions.CORRUPTIONS for s in (1, 2, 3)]
    dev = torch.device("cuda")
    model = SimpleBaseline("resnet50", pretrained=False).to(dev).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(os.path.join(ROOT, "cache", "simplebaseline_resnet50.pt"), map_location=dev))
    model.eval()
    os.makedirs(os.path.join(ROOT, "cache"), exist_ok=True)
    for cond in conds:
        name, sev = (cond.split(":") + ["0"])[:2]
        out = os.path.join(ROOT, "cache", f"outputs_{name}{'' if name == 'clean' else '_s' + sev}.npz")
        if os.path.exists(out):
            continue
        ds = ValPersons(name, int(sev))
        pred, conf, sigma = run(model, ds, dev)
        np.savez_compressed(out, pred=pred, conf=conf, sigma=sigma,
                            gt=np.stack([it["kpts"] for it in ds.items]),
                            box=np.stack([it["box"] for it in ds.items]),
                            area=np.float32([it["area"] for it in ds.items]),
                            img_id=np.int64([it["img_id"] for it in ds.items]),
                            n_people=np.int32([it["n_people"] for it in ds.items]))
        print(f"{cond}: {len(ds)} people -> {os.path.basename(out)}", flush=True)


if __name__ == "__main__":
    main()
