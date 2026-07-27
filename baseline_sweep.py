"""
baseline_sweep.py — validation sweep over the parameters that now matter.

The diagnosis said: the antipodal score is essentially solved (1.98/2.00), the
angle is right on ~74% of images, and IoU is the binding constraint. Two things
drive IoU, and both are swept here:

  tie_break     which of the many tied-at-2.0 pairs to take (WHERE along the
                object the grasp goes)
  width_margin  }  how big the rectangle is
  plate         }

min_dist / max_dist are still swept but expect them to matter little now.

Scoring uses a FIXED denominator: images where no prediction is produced count
as failures. Otherwise a configuration that answers less often looks better than
one that answers more, which is how the earlier numbers drifted.

    python baseline_sweep.py

Then put the winner in config.py and run the test split exactly once.
"""
import json
import os
import pickle
import time

import numpy as np

from config import BACKGROUNDS_DIR
from data_loading import parse_grasp_rectangles, load_rgb
from baseline import (
    load_backgrounds, create_mask, get_largest_contour, estimate_normals,
    xywh_theta_to_corners,
)
from config import PLATE_CAP_RATIO
from evaluate_baseline import is_correct_grasp

# --- the grid -------------------------------------------------------------
TIE_BREAKS = ["first", "centroid", "narrow", "wide"]
# extended past the measured GT/predicted width ratio (~1.17) deliberately, to
# find where accuracy turns over. If it never turns over, bigger boxes are
# gaming IoU rather than finding better grasps -- report that, do not exploit it.
WIDTH_MARGINS = [1.0, 1.15, 1.3, 1.5, 1.8]
PLATES = [26, 30, 34]
MIN_DISTS = [20, 26, 32]
MAX_DISTS = [90, 110]
# TIE_TOL is swept, not fixed. 0.0 reproduces the ORIGINAL algorithm exactly
# (strict `score > best_score`, i.e. first pair at the exact maximum), which is
# the control condition -- without it there is no way to tell whether a
# tie-break criterion helps or whether the tolerance itself hurts.
TIE_TOLS = [0.0, 0.01, 0.05, 0.15]
# Rejecting pairs whose midpoint falls off the object helps on concave shapes
# (a chord across a mug can cross the handle hole) but can eliminate EVERY
# candidate on some images. That trade-off is a question for the data, not a
# constant -- so it is swept, and the fixed denominator penalises the skips.
MASK_CHECKS = [True, False]
# None = old behaviour (clamp min_dist up to plate, skipping narrow objects)
PLATE_CAPS = [0.95, None]

SAMPLE_STEP = 4
CACHE_FILE = "sweep_cache_val.pkl"
SPLIT_KEY = "val"


# --- stage 1: per-image geometry, computed once ---------------------------
def build_cache(split_path="dataset_split.json", cache_file=CACHE_FILE):
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache, n_split = pickle.load(f)
        print(f"loaded cached geometry for {len(cache)} images (split has {n_split})")
        print(f"delete {cache_file} if you changed create_mask or estimate_normals")
        return cache, n_split

    with open(split_path, "r") as f:
        split = json.load(f)

    entries = split[SPLIT_KEY]
    backgrounds = load_backgrounds(BACKGROUNDS_DIR)
    print(f"building geometry cache for {len(entries)} {SPLIT_KEY} images...")

    cache = []
    t0 = time.time()
    for k, entry in enumerate(entries):
        pcd_id, folder = entry["id"], entry["folder"]
        try:
            img = load_rgb(folder, pcd_id)
            gt = parse_grasp_rectangles(f"{folder}/pcd{pcd_id}cpos.txt")
        except Exception:
            continue
        if img is None or len(gt) == 0:
            continue

        mask = create_mask(img, backgrounds)
        contour = get_largest_contour(mask)
        if contour is None or len(contour) < 10:
            continue

        normals = np.array(estimate_normals(contour))
        idx = np.arange(0, len(contour), SAMPLE_STEP)

        cache.append({
            "id": pcd_id,
            "points": contour[idx].astype(np.float64),
            "normals": normals[idx],
            "centroid": contour.astype(float).mean(axis=0),
            "mask": np.packbits(mask > 0),
            "mask_shape": mask.shape,
            "gt": gt,
        })
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(entries)}  ({time.time()-t0:.0f}s)")

    with open(cache_file, "wb") as f:
        pickle.dump((cache, len(entries)), f)
    print(f"cached {len(cache)} images in {time.time()-t0:.0f}s")
    return cache, len(entries)


def unpack_mask(item):
    n = item["mask_shape"][0] * item["mask_shape"][1]
    return np.unpackbits(item["mask"])[:n].reshape(item["mask_shape"]).astype(bool)


# --- stage 2: vectorised scoring, with tie-breaking -----------------------
def precompute(item):
    """All-pairs score and distance. Same formula as find_best_antipodal_pair."""
    p = item["points"]
    nrm = item["normals"]
    diff = p[None, :, :] - p[:, None, :]
    dist = np.linalg.norm(diff, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dirs = diff / dist[..., None]
    score = -np.einsum("ik,ijk->ij", nrm, dirs) + np.einsum("jk,ijk->ij", nrm, dirs)

    iu = np.triu_indices(len(p), k=1)
    item["_s"] = score[iu]
    item["_d"] = dist[iu]
    item["_i"], item["_j"] = iu

    mid = (p[iu[0]] + p[iu[1]]) / 2
    item["_mid"] = mid
    item["_cdist"] = np.linalg.norm(mid - item["centroid"], axis=1)

    m = unpack_mask(item)
    xi = np.clip(np.round(mid[:, 0]).astype(int), 0, m.shape[1] - 1)
    yi = np.clip(np.round(mid[:, 1]).astype(int), 0, m.shape[0] - 1)
    item["_inside"] = m[yi, xi]


def pick_pair(item, min_dist, max_dist, tie_break, mask_check=True, tie_tol=0.05):
    s, d = item["_s"], item["_d"]
    valid = (d >= min_dist) & (d <= max_dist) & np.isfinite(s)
    if mask_check:
        valid = valid & item["_inside"]
    if not valid.any():
        return None

    top = s[valid].max()
    fin = valid & (s >= top - tie_tol)
    idx = np.flatnonzero(fin)

    if tie_break == "centroid":
        k = idx[np.argmin(item["_cdist"][idx])]
    elif tie_break == "narrow":
        k = idx[np.argmin(d[idx])]
    elif tie_break == "wide":
        k = idx[np.argmax(d[idx])]
    else:
        k = idx[0]

    a = item["points"][item["_i"][k]]
    b = item["points"][item["_j"][k]]
    return a, b, float(d[k]), float(s[k]), int(fin.sum()), int(valid.sum())


if __name__ == "__main__":
    cache, n_split = build_cache()
    print("\nprecomputing pairwise scores...")
    for item in cache:
        precompute(item)

    # how much work is the tie-break actually doing?
    tied, tot = [], []
    for item in cache:
        r = pick_pair(item, MIN_DISTS[0], MAX_DISTS[0], "first",
                      mask_check=False, tie_tol=0.05)
        if r:
            tied.append(r[4])
            tot.append(r[5])
    if tied:
        print(f"\npairs tied within 0.05 of the top score: "
              f"median {np.median(tied):.0f} of {np.median(tot):.0f} valid pairs "
              f"per image")
        print("If that first number is large, the tie-break is choosing the grasp,")
        print("not the antipodal score — which is why it is being swept here.\n")

    print("--- images with NO prediction (these count as failures) ---")
    print(f"  {'min_dist':>9}{'mask_check':>12}{'no prediction':>16}")
    for md in MIN_DISTS:
        for mc in MASK_CHECKS:
            n_none = sum(1 for it in cache
                         if pick_pair(it, md, MAX_DISTS[-1], "first", mc, 0.05) is None)
            n_none += n_split - len(cache)
            print(f"  {md:>9}{str(mc):>12}{n_none:>10} / {n_split}")
    print("  If mask_check=True drops many more images than it fixes, that is the")
    print("  answer -- the fixed denominator below will price it in either way.\n")

    results = []
    t0 = time.time()
    for cap in PLATE_CAPS:
     for mc in MASK_CHECKS:
      for tol in TIE_TOLS:
       for tb in (["first"] if tol == 0.0 else TIE_BREAKS):
        for md in MIN_DISTS:
            for xd in MAX_DISTS:
                for plate in PLATES:
                    eff_md = md if cap is not None else max(md, plate)
                    picks_p = [(it, pick_pair(it, eff_md, xd, tb, mc, tol))
                               for it in cache]
                    for wm in WIDTH_MARGINS:
                        correct = attempted = 0
                        for it, r in picks_p:
                            if r is None:
                                continue
                            a, b, dist, _, _, _ = r
                            c = (a + b) / 2
                            theta = np.degrees(np.arctan2(*(b - a)[::-1]))
                            w_pred = dist * wm
                            h_pred = plate if cap is None else min(plate, w_pred * cap)
                            pred = xywh_theta_to_corners(
                                c[0], c[1], w_pred, h_pred, theta)
                            attempted += 1
                            if any(is_correct_grasp(pred, g) for g in it["gt"]):
                                correct += 1
                        results.append({
                            "tie": tb, "min": eff_md, "max": xd,
                            "plate": plate, "wm": wm, "mask": mc, "cap": cap, "tol": tol,
                            "correct": correct, "attempted": attempted,
                            "acc_fixed": correct / n_split,
                            "acc_attempted": correct / attempted if attempted else 0.0,
                        })

    # dedupe (clamping can make combos identical)
    seen, uniq = set(), []
    for r in results:
        key = (r["tie"], r["min"], r["max"], r["plate"], r["wm"], r["mask"], r["cap"], r["tol"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    uniq.sort(key=lambda r: (-r["acc_fixed"], r["plate"], r["wm"]))

    print(f"evaluated {len(uniq)} combinations in {time.time()-t0:.0f}s\n")
    print("=" * 76)
    print(f"TOP 12 ON {SPLIT_KEY.upper()}  (denominator = all {n_split} images)")
    print("=" * 76)
    print(f"  {'tie_break':<10}{'min':>5}{'max':>5}{'plate':>7}{'margin':>8}"
          f"{'mask':>7}{'cap':>7}{'tol':>7}{'accuracy':>11}{'correct':>10}{'skipped':>9}")
    for r in uniq[:12]:
        print(f"  {r['tie']:<10}{r['min']:>5}{r['max']:>5}{r['plate']:>7}"
              f"{r['wm']:>8.2f}{str(r['mask']):>7}{str(r['cap']):>7}{r['tol']:>7.2f}{r['acc_fixed']:>10.2%}"
              f"{r['correct']:>8}/{n_split}{n_split - r['attempted']:>9}")

    print("\n--- effect of each knob, averaged over everything else ---")
    for field, label in [("tie", "tie_break"), ("wm", "width_margin"),
                         ("plate", "plate_thickness"), ("mask", "mask_check"),
                         ("min", "min_dist"), ("cap", "plate_cap_ratio"),
                         ("tol", "tie_tol")]:
        print(f"  {label}:")
        for v in sorted({r[field] for r in uniq}, key=str):
            sub = [r["acc_fixed"] for r in uniq if r[field] == v]
            print(f"    {str(v):<10} mean {np.mean(sub):.2%}  best {max(sub):.2%}")

    # explicit control: the ORIGINAL algorithm, no tolerance, no tie-break
    ctrl = [r for r in uniq if r["tol"] == 0.0 and r["wm"] == 1.0
            and r["cap"] is None and not r["mask"]]
    if ctrl:
        c = max(ctrl, key=lambda r: r["acc_fixed"])
        print(f"\n--- CONTROL: original algorithm (tol=0, margin=1.0, no mask "
              f"check, no plate cap) ---")
        print(f"  best over min/max/plate: {c['acc_fixed']:.2%} "
              f"({c['correct']}/{n_split})  at min_dist={c['min']}, "
              f"max_dist={c['max']}, plate={c['plate']}")
        print("  Nothing below this by more than the standard error is an")
        print("  improvement. If the control wins, keep what you already had and")
        print("  report the swept variants as tested-and-rejected.")

    best = uniq[0]
    n_tied = sum(1 for r in uniq if r["correct"] == best["correct"])
    if n_tied > 1:
        print(f"\nnote: {n_tied} combinations tied at {best['correct']} correct. "
              f"On {n_split} images the standard error is about "
              f"{100*np.sqrt(best['acc_fixed']*(1-best['acc_fixed'])/n_split):.1f} "
              f"points, so treat differences smaller than that as noise.")

    print("\nPut these in config.py:\n")
    print(f"    MIN_DIST = {best['min']}")
    print(f"    MAX_DIST = {best['max']}")
    print(f"    PLATE_THICKNESS = {best['plate']}")
    print(f"    WIDTH_MARGIN = {best['wm']}")
    print(f"    TIE_BREAK = \"{best['tie']}\"")
    print(f"    MASK_CHECK = {best['mask']}")
    print(f"    PLATE_CAP_RATIO = {best['cap']}")
    print(f"    TIE_TOL = {best['tol']}")
    if best["wm"] > 1.2:
        print(f"\n  WARNING: width_margin={best['wm']} exceeds the measured "
              f"GT/predicted width ratio.")
        print("  Check the width_margin row in the per-knob table: if accuracy")
        print("  keeps rising with size and never turns over, this is IoU")
        print("  forgiveness, not a better grasp. Prefer the measured ratio and")
        print("  report the rest as a finding about the metric.")

    print("\nThen, once:\n\n    python evaluate_baseline.py")
    print("\nBefore accepting: check the winning plate_thickness and width_margin")
    print("against measure_gt_stats.py. Matching the measured GT distribution is")
    print("principled; drifting well above it is buying IoU with a bigger box.")