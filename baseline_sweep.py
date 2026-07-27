#imports
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

#parameter options to test
TIE_BREAKS = ["first", "centroid", "narrow", "wide"]
WIDTH_MARGINS = [1.0, 1.15, 1.3, 1.5, 1.8]
PLATES = [26, 30, 34]
MIN_DISTS = [20, 26, 32]
MAX_DISTS = [90, 110]
TIE_TOLS = [0.0, 0.01, 0.05, 0.15]
MASK_CHECKS = [True, False]
PLATE_CAPS = [0.95, None]

SAMPLE_STEP = 4
CACHE_FILE = "sweep_cache_val.pkl"
SPLIT_KEY = "val"


#builds or loads saved image data
def build_cache(split_path="dataset_split.json", cache_file=CACHE_FILE):
    #loads the cache if it already exists
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache, n_split = pickle.load(f)
        print(f"loaded cached geometry for {len(cache)} images (split has {n_split})")
        print(f"delete {cache_file} if you changed create_mask or estimate_normals")
        return cache, n_split

    #loads the validation split
    with open(split_path, "r") as f:
        split = json.load(f)

    entries = split[SPLIT_KEY]
    backgrounds = load_backgrounds(BACKGROUNDS_DIR)
    print(f"building geometry cache for {len(entries)} {SPLIT_KEY} images...")

    cache = []
    t0 = time.time()
    for k, entry in enumerate(entries):
        pcd_id, folder = entry["id"], entry["folder"]

        #loads the image and ground truths
        try:
            img = load_rgb(folder, pcd_id)
            gt = parse_grasp_rectangles(f"{folder}/pcd{pcd_id}cpos.txt")
        except Exception:
            continue
        if img is None or len(gt) == 0:
            continue

        #finds the object contour
        mask = create_mask(img, backgrounds)
        contour = get_largest_contour(mask)
        if contour is None or len(contour) < 10:
            continue

        #samples contour points and normals
        normals = np.array(estimate_normals(contour))
        idx = np.arange(0, len(contour), SAMPLE_STEP)

        #compresses the mask to save memory
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

    #saves the cache
    with open(cache_file, "wb") as f:
        pickle.dump((cache, len(entries)), f)
    print(f"cached {len(cache)} images in {time.time()-t0:.0f}s")
    return cache, len(entries)

#rebuilds the mask from packed bits
def unpack_mask(item):
    n = item["mask_shape"][0] * item["mask_shape"][1]
    return np.unpackbits(item["mask"])[:n].reshape(item["mask_shape"]).astype(bool)


#computes all point pairs at once
def precompute(item):
    p = item["points"]
    nrm = item["normals"]

    #gets pair distances and directions
    diff = p[None, :, :] - p[:, None, :]
    dist = np.linalg.norm(diff, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dirs = diff / dist[..., None]
    score = -np.einsum("ik,ijk->ij", nrm, dirs) + np.einsum("jk,ijk->ij", nrm, dirs)

    #keeps each pair only once
    iu = np.triu_indices(len(p), k=1)
    item["_s"] = score[iu]
    item["_d"] = dist[iu]
    item["_i"], item["_j"] = iu

    #gets pair midpoint data
    mid = (p[iu[0]] + p[iu[1]]) / 2
    item["_mid"] = mid
    item["_cdist"] = np.linalg.norm(mid - item["centroid"], axis=1)

    #checks if each midpoint is inside the mask
    m = unpack_mask(item)
    xi = np.clip(np.round(mid[:, 0]).astype(int), 0, m.shape[1] - 1)
    yi = np.clip(np.round(mid[:, 1]).astype(int), 0, m.shape[0] - 1)
    item["_inside"] = m[yi, xi]

#chooses the best point pair
def pick_pair(item, min_dist, max_dist, tie_break, mask_check=True, tie_tol=0.05):
    s, d = item["_s"], item["_d"]

    #keeps pairs within the distance limits
    valid = (d >= min_dist) & (d <= max_dist) & np.isfinite(s)
    if mask_check:
        valid = valid & item["_inside"]
    if not valid.any():
        return None

    #keeps pairs close to the top score
    top = s[valid].max()
    fin = valid & (s >= top - tie_tol)
    idx = np.flatnonzero(fin)

    #breaks ties using the selected method
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
    #loads and prepares cached data
    cache, n_split = build_cache()
    print("\nprecomputing pairwise scores...")
    for item in cache:
        precompute(item)

    #checks how many pairs are tied
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

    #counts images with no prediction
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

    #tests every parameter combination
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
                        #evaluates this combination
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

    #removes duplicate combinations
    seen, uniq = set(), []
    for r in results:
        key = (r["tie"], r["min"], r["max"], r["plate"], r["wm"], r["mask"], r["cap"], r["tol"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    uniq.sort(key=lambda r: (-r["acc_fixed"], r["plate"], r["wm"]))

    #prints the best combinations
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

    #shows the effect of each parameter
    print("\n--- effect of each knob, averaged over everything else ---")
    for field, label in [("tie", "tie_break"), ("wm", "width_margin"),
                         ("plate", "plate_thickness"), ("mask", "mask_check"),
                         ("min", "min_dist"), ("cap", "plate_cap_ratio"),
                         ("tol", "tie_tol")]:
        print(f"  {label}:")
        for v in sorted({r[field] for r in uniq}, key=str):
            sub = [r["acc_fixed"] for r in uniq if r[field] == v]
            print(f"    {str(v):<10} mean {np.mean(sub):.2%}  best {max(sub):.2%}")

    #compares against the original algorithm
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

    #checks if multiple combinations tied
    best = uniq[0]
    n_tied = sum(1 for r in uniq if r["correct"] == best["correct"])
    if n_tied > 1:
        print(f"\nnote: {n_tied} combinations tied at {best['correct']} correct. "
              f"On {n_split} images the standard error is about "
              f"{100*np.sqrt(best['acc_fixed']*(1-best['acc_fixed'])/n_split):.1f} "
              f"points, so treat differences smaller than that as noise.")

    #prints the winning settings
    print("\nPut these in config.py:\n")
    print(f"    MIN_DIST = {best['min']}")
    print(f"    MAX_DIST = {best['max']}")
    print(f"    PLATE_THICKNESS = {best['plate']}")
    print(f"    WIDTH_MARGIN = {best['wm']}")
    print(f"    TIE_BREAK = \"{best['tie']}\"")
    print(f"    MASK_CHECK = {best['mask']}")
    print(f"    PLATE_CAP_RATIO = {best['cap']}")
    print(f"    TIE_TOL = {best['tol']}")
    #warns if the predicted box may be too wide
    if best["wm"] > 1.2:
        print(f"\n  WARNING: width_margin={best['wm']} exceeds the measured "
              f"GT/predicted width ratio.")
        print("  Check the width_margin row in the per-knob table: if accuracy")
        print("  keeps rising with size and never turns over, this is IoU")
        print("  forgiveness, not a better grasp. Prefer the measured ratio and")
        print("  report the rest as a finding about the metric.")

    #shows the final evaluation command
    print("\nThen, once:\n\n    python evaluate_baseline.py")
    print("\nBefore accepting: check the winning plate_thickness and width_margin")
    print("against measure_gt_stats.py. Matching the measured GT distribution is")
    print("principled; drifting well above it is buying IoU with a bigger box.")