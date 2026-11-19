"""All experiments and figures from the cached model outputs (run src/infer.py first).

Calibration and test sets are a random 50/50 split of COCO val2017 *images* (people from one image are
never on both sides), repeated R times; we report the mean and the 2.5-97.5 % range over repeats.

    python src/analyze.py            # -> results/*.json, results/figures/*.png
"""
import json
import os
import zlib

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import corruptions
from conformal import METHODS, Calibrator, whole_body

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "results", "figures")
COCO_ROOT = os.environ.get("COCO_ROOT", os.path.expanduser("~/datasets/portfolio/coco"))
JOINTS = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
          "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
GROUPS = {"head": [0, 1, 2, 3, 4], "shoulders/elbows": [5, 6, 7, 8], "wrists": [9, 10], "hips": [11, 12],
          "knees/ankles": [13, 14, 15, 16]}
LABEL = {"heatmap-gaussian": "A  heatmap as Gaussian (no calibration)", "cp-fixed": "B  conformal, fixed radius",
         "cp-adaptive": "C  conformal, adaptive", "cp-mondrian": "D  conformal, Mondrian (confidence)",
         "cp-whole-body": "W  conformal, whole body"}
COLOR = {"heatmap-gaussian": "#7f7f7f", "cp-fixed": "#1f77b4", "cp-adaptive": "#d62728", "cp-mondrian": "#2ca02c",
         "cp-whole-body": "#9467bd"}
BODY_CM = 170.0   # only to translate a fraction of person height into an illustrative distance
R = int(os.environ.get("REPEATS", 200))


def load(cond="clean"):
    z = np.load(os.path.join(ROOT, "cache", f"outputs_{cond}.npz"))
    gt, pred = z["gt"], z["pred"]
    P, J = gt.shape[:2]
    lab = gt[:, :, 2] > 0
    person, joint = np.nonzero(lab)
    h = z["box"][:, 3]
    d = dict(err=np.linalg.norm(pred[lab] - gt[lab][:, :2], axis=1).astype(np.float64),
             sigma=z["sigma"][lab].astype(np.float64), conf=z["conf"][lab], v=gt[lab][:, 2].astype(int),
             joint=joint, person=person, h=h[person].astype(np.float64), img=z["img_id"][person],
             area=z["area"][person], n_people=z["n_people"][person], n_persons=P)
    return d, z


def subset(d, k):
    return {key: (v[k] if isinstance(v, np.ndarray) and len(v) == len(d["err"]) else v) for key, v in d.items()}


def splits(d):
    imgs = np.unique(d["img"])
    for r in range(R):
        cal_imgs = np.random.default_rng(r).permutation(imgs)[: len(imgs) // 2]
        c = np.isin(d["img"], cal_imgs)
        yield c, ~c


def body_cov(t, covered):
    """Fraction of people whose labelled joints are all inside their radius."""
    bad = np.zeros(t["n_persons"], bool); bad[t["person"][~covered]] = True
    return 1 - bad[np.unique(t["person"])].mean()


def evaluate(cal, test, alpha, groups=None):
    """Coverage, whole-body coverage and radii of every method for one split."""
    out = {}
    for m in METHODS + ["cp-whole-body"]:
        if m == "cp-whole-body":
            r = whole_body(cal, test, alpha)
        else:
            r = Calibrator(m, alpha).fit(cal).radius(test)
        cov = test["err"] <= r
        res = {"coverage": cov.mean(), "body_coverage": body_cov(test, cov),
               "radius_rel_median": np.median(r / test["h"]), "radius_rel_mean": np.mean(np.minimum(r / test["h"], 2.0)),
               "radius_px_median": np.median(r)}
        if groups is not None:
            for g, k in groups(test).items():
                res[f"cov|{g}"] = cov[k].mean()
                res[f"rad|{g}"] = np.median(r[k] / test["h"][k])
        out[m] = res
    return out


def aggregate(runs):
    agg = {}
    for m in runs[0]:
        agg[m] = {}
        for key in runs[0][m]:
            v = np.array([r[m][key] for r in runs], np.float64)
            agg[m][key] = [float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return agg


def subgroups(t):
    g = {"visible (v=2)": t["v"] == 2, "occluded (v=1)": t["v"] == 1,
         "medium person": t["area"] < 96 ** 2, "large person": t["area"] >= 96 ** 2,
         "alone in image": t["n_people"] == 1, "2-4 people": (t["n_people"] >= 2) & (t["n_people"] <= 4),
         "5+ people": t["n_people"] >= 5}
    for name, js in GROUPS.items():
        g[name] = np.isin(t["joint"], js)
    return g


def per_joint(t):
    return {JOINTS[j]: t["joint"] == j for j in range(17)}


# --------------------------------------------------------------------------------------------- experiments
def exp_calibration(d):
    """Q1/Q2: nominal vs empirical coverage on clean data."""
    alphas = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
    out = {}
    for a in alphas:
        runs = [evaluate(subset(d, c), subset(d, t), a) for c, t in splits(d)]
        out[str(a)] = aggregate(runs)
        print(f"alpha {a}: " + "  ".join(f"{m} {out[str(a)][m]['coverage'][0]:.3f}" for m in out[str(a)]), flush=True)
    return out


def exp_subgroups(d, alpha=0.1):
    runs = [evaluate(subset(d, c), subset(d, t), alpha, groups=lambda t: {**subgroups(t), **per_joint(t)})
            for c, t in splits(d)]
    return aggregate(runs)


def exp_shift(d_clean, alpha=0.1):
    """Q4: calibrate on clean images, deploy on corrupted ones (same people, same split)."""
    out = {}
    conds = [f"{c}_s{s}" for c in corruptions.CORRUPTIONS for s in (1, 2, 3)]
    for cond in conds:
        d, _ = load(cond)
        runs = []
        for c, t in splits(d_clean):
            runs.append(evaluate(subset(d_clean, c), subset(d, t), alpha))
        out[cond] = aggregate(runs)
        out[cond]["median_error_rel"] = float(np.median(d["err"] / d["h"]))
        print(cond, {m: round(out[cond][m]["coverage"][0], 3) for m in METHODS + ["cp-whole-body"]}, flush=True)
    out["clean"] = {"median_error_rel": float(np.median(d_clean["err"] / d_clean["h"]))}
    return out


def exp_crowd_shift(d, alpha=0.1):
    """Calibrate only on people who are alone in the image, deploy in crowds (5+ people)."""
    runs = []
    for c, t in splits(d):
        cal = subset(d, c & (d["n_people"] == 1))
        test = subset(d, t & (d["n_people"] >= 5))
        runs.append(evaluate(cal, test, alpha))
    return aggregate(runs)


# ------------------------------------------------------------------------------------------------- figures
def fig_reliability(cal):
    alphas = sorted(float(a) for a in cal)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0.5, 1], [0.5, 1], "k--", lw=1, label="perfect calibration")
    for m in METHODS + ["cp-whole-body"]:
        key = "body_coverage" if m == "cp-whole-body" else "coverage"
        y = [cal[str(a)][m][key][0] for a in alphas]
        ax.plot([1 - a for a in alphas], y, "o-", color=COLOR[m], label=LABEL[m] + (" (all joints)" if key == "body_coverage" else ""))
    ax.set(xlabel="target coverage 1 - α", ylabel="empirical coverage on test images", xlim=(0.48, 1.0), ylim=(0.3, 1.0),
           title="Does the promised coverage hold? (clean COCO)")
    ax.legend(fontsize=7.5, loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "reliability.png"), dpi=130); plt.close()


def fig_subgroups(sg):
    groups = ["visible (v=2)", "occluded (v=1)", "medium person", "large person", "alone in image", "2-4 people",
              "5+ people"] + list(GROUPS)
    ms = ["cp-fixed", "cp-adaptive", "cp-mondrian"]
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(groups)); w = 0.26
    for i, m in enumerate(ms):
        v = np.array([sg[m][f"cov|{g}"] for g in groups])
        ax.bar(x + (i - 1) * w, v[:, 0], w, yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=COLOR[m], label=LABEL[m],
               capsize=2)
    ax.axhline(0.9, color="k", ls="--", lw=1)
    ax.set_xticks(x, groups, rotation=25, ha="right"); ax.set_ylim(0.6, 1.0); ax.set_ylabel("coverage (target 0.90)")
    ax.set_title("Who is left unprotected? Coverage by subgroup (clean, α = 0.1)")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "subgroup_coverage.png"), dpi=130); plt.close()


def fig_shift(sh):
    names = corruptions.CORRUPTIONS
    ms = ["cp-fixed", "cp-adaptive", "cp-mondrian", "heatmap-gaussian"]
    fig, axs = plt.subplots(1, len(names), figsize=(16, 3.6), sharey=True)
    for ax, c in zip(axs, names):
        for m in ms:
            y = [sh[f"{c}_s{s}"][m]["coverage"] for s in (1, 2, 3)]
            ax.errorbar([1, 2, 3], [v[0] for v in y], yerr=[[v[0] - v[1] for v in y], [v[2] - v[0] for v in y]],
                        marker="o", color=COLOR[m], label=LABEL[m], capsize=2)
        ax.axhline(0.9, color="k", ls="--", lw=1)
        ax.set(title=c.replace("_", " "), xticks=[1, 2, 3], xlabel="severity"); ax.grid(alpha=0.3)
    axs[0].set_ylabel("coverage (target 0.90)")
    axs[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Calibrated on clean images, deployed on corrupted ones", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "shift_coverage.png"), dpi=130, bbox_inches="tight"); plt.close()


def fig_joint_radius(sg):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(17)
    for i, m in enumerate(["cp-fixed", "cp-adaptive"]):
        v = np.array([sg[m][f"rad|{j}"][0] for j in JOINTS]) * BODY_CM
        ax.bar(x + (i - 0.5) * 0.4, v, 0.4, color=COLOR[m], label=LABEL[m])
    ax.set_xticks(x, JOINTS, rotation=40, ha="right")
    ax.set_ylabel(f"median radius (cm, {BODY_CM:.0f} cm person)")
    ax.set_title("90 % safety radius per joint")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "joint_radius.png"), dpi=130); plt.close()


def fig_examples(d_clean, alpha=0.1):
    """Adaptive conformal circles on one test image: clean, motion blur, occluder."""
    c, t = next(splits(d_clean))
    cal = subset(d_clean, c)
    calib = Calibrator("cp-adaptive", alpha).fit(cal)
    z = np.load(os.path.join(ROOT, "cache", "outputs_clean.npz"))
    # a large, mostly visible person in a test image
    test_people = np.unique(d_clean["person"][t])
    cand = [p for p in test_people if z["area"][p] > 150 ** 2 and (z["gt"][p, :, 2] == 2).sum() >= 15]
    p = cand[len(cand) // 3]
    from pycocotools.coco import COCO
    coco = COCO(os.path.join(COCO_ROOT, "annotations", "person_keypoints_val2017.json"))
    info = coco.imgs[int(z["img_id"][p])]
    ann = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=info["id"], iscrowd=False))
           if a["num_keypoints"] > 0 and abs(a["area"] - z["area"][p]) < 1e-3][0]
    base = cv2.imread(os.path.join(COCO_ROOT, "val2017", info["file_name"]))
    files = sorted({coco.imgs[i]["file_name"] for i in coco.getImgIds(catIds=[1])})
    conds = [("clean", "clean", 0), ("motion_blur_s3", "motion_blur", 3), ("occluder_s3", "occluder", 3)]
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    x, y, w, h = z["box"][p]
    for ax, (cname, name, sev) in zip(axs, conds):
        zz = np.load(os.path.join(ROOT, "cache", f"outputs_{cname}.npz"))
        seed = zlib.crc32(f"{ann['id']}-{name}-{sev}".encode())
        patch = cv2.imread(os.path.join(COCO_ROOT, "val2017", files[seed % len(files)])) if name == "occluder" else None
        img = corruptions.apply(base, name, sev, seed, box=z["box"][p], patch_src=patch)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dd = {"err": np.zeros(17), "sigma": zz["sigma"][p].astype(np.float64), "joint": np.arange(17),
              "h": np.full(17, h), "conf": zz["conf"][p]}
        r = calib.radius(dd)
        gt = zz["gt"][p]
        for j in range(17):
            if gt[j, 2] == 0:
                continue
            px, py = zz["pred"][p, j]
            inside = np.hypot(px - gt[j, 0], py - gt[j, 1]) <= r[j]
            ax.add_patch(plt.Circle((px, py), r[j], fill=False, lw=1.6, color="#2ca02c" if inside else "#d62728"))
            ax.plot(gt[j, 0], gt[j, 1], "x", color="white", ms=5, mew=1.5)
        pad = 0.15 * h
        ax.set_xlim(x - pad, x + w + pad); ax.set_ylim(y + h + pad, y - pad); ax.axis("off")
        ax.set_title(f"{name.replace('_', ' ')}{'' if sev == 0 else f' (severity {sev})'}")
    fig.suptitle("90 % conformal safety radii (adaptive). White × = true joint; green = covered, red = missed", y=0.98)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "examples.png"), dpi=110); plt.close()


def main():
    os.makedirs(FIG, exist_ok=True)
    d, _ = load("clean")
    print(f"{d['n_persons']} people, {len(d['err'])} labelled joints, {len(np.unique(d['img']))} images", flush=True)
    res = {"n_people": int(d["n_persons"]), "n_joints": int(len(d["err"])), "n_images": int(len(np.unique(d["img"]))),
           "repeats": R}
    res["calibration"] = exp_calibration(d)
    res["subgroups_alpha0.1"] = exp_subgroups(d)
    res["crowd_shift_alpha0.1"] = exp_crowd_shift(d)
    res["corruption_shift_alpha0.1"] = exp_shift(d)
    json.dump(res, open(os.path.join(RES, "results.json"), "w"), indent=1)
    fig_reliability(res["calibration"]); fig_subgroups(res["subgroups_alpha0.1"])
    fig_shift(res["corruption_shift_alpha0.1"]); fig_joint_radius(res["subgroups_alpha0.1"])
    fig_examples(d)


if __name__ == "__main__":
    main()
