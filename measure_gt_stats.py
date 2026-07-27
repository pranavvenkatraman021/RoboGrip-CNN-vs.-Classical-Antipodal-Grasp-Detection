"""
measure_gt_stats.py — measure the real width/height distribution of ground-truth
grasp rectangles, so the tuning grid is grounded in data instead of guesses.

Uses the TRAIN split only. Never touches test.

    python measure_gt_stats.py
"""
import json
import numpy as np

from data_loading import parse_grasp_rectangles, convert

with open("dataset_split.json", "r") as f:
    split = json.load(f)

widths, heights, n_rects_per_img, missing = [], [], [], 0

for entry in split["train"]:
    path = f"{entry['folder']}/pcd{entry['id']}cpos.txt"
    try:
        rects = parse_grasp_rectangles(path)
    except FileNotFoundError:
        missing += 1
        continue

    n_rects_per_img.append(len(rects))
    for r in rects:
        _, _, w, h, _ = convert(r)
        widths.append(w)
        heights.append(h)

w = np.array(widths)
h = np.array(heights)

print(f"images scanned: {len(n_rects_per_img)}   label files missing: {missing}")
print(f"total ground-truth rectangles: {len(w)}")
print(f"rectangles per image: mean {np.mean(n_rects_per_img):.1f}, "
      f"median {np.median(n_rects_per_img):.0f}, "
      f"min {np.min(n_rects_per_img)}, max {np.max(n_rects_per_img)}")

def describe(name, a):
    print(f"\n{name} (px)")
    print(f"  mean {a.mean():6.1f}   median {np.median(a):6.1f}   std {a.std():6.1f}")
    print(f"  percentiles  1%={np.percentile(a,1):.0f}  5%={np.percentile(a,5):.0f}  "
          f"25%={np.percentile(a,25):.0f}  75%={np.percentile(a,75):.0f}  "
          f"95%={np.percentile(a,95):.0f}  99%={np.percentile(a,99):.0f}")

describe("width  (longer edge — gripper opening)", w)
describe("height (shorter edge — plate thickness)", h)

print("\n--- suggested tuning grid ---")
print(f"  min_dist   around the 5th percentile of width : {np.percentile(w,5):.0f}")
print(f"  max_dist   around the 95th percentile of width: {np.percentile(w,95):.0f}")
print(f"  plate_thickness around median height          : {np.median(h):.0f}")
print("\n  Remember the B2 constraint: min_dist must be >= plate_thickness.")
print(f"  Side-ratio (h/w) mean {np.mean(h/w):.2f} — near-square rectangles")
print(f"  (ratio > 0.9): {np.mean((h/w) > 0.9):.1%} of labels.")

print("\n--- copy these into baseline_sweep.py if they differ from the defaults ---")
lo = int(max(10, np.percentile(w, 5) // 5 * 5))
hi = int(max(np.percentile(w, 95) // 10 * 10, lo + 40))
pt = int(np.median(h) // 2 * 2)
print(f"MIN_DISTS  = [{lo}, {lo+5}, {lo+10}, {lo+15}]")
print(f"MAX_DISTS  = [{hi}, {hi+20}, {hi+40}]")
print(f"PLATES     = [{pt-4}, {pt}, {pt+4}]")