"""
diagnose_baseline.py — decompose baseline failures instead of guessing.

Answers four questions the accuracy number alone can't:
  1. With skipped images counted as failures (they should be), what is the
     accuracy over a FIXED denominator?
  2. Of the failures, how many are position/size (IoU) vs orientation (angle)?
  3. Is the antipodal search finding well-opposed pairs, or is the contour
     it's given already bad?
  4. Is the fixed plate_thickness the thing capping IoU? (Measured by asking
     what IoU we'd get keeping our center and angle but using the GT's size.)

    python diagnose_baseline.py val
    python diagnose_baseline.py test

Optionally saves the worst cases as images for eyeballing:

    python diagnose_baseline.py val --save 8
"""
import json
import sys

import numpy as np

from config import BACKGROUNDS_DIR, MIN_DIST, MAX_DIST, PLATE_THICKNESS
from data_loading import load_rgb, parse_grasp_rectangles, convert
from baseline import load_backgrounds, xywh_theta_to_corners
from evaluate_baseline import (
    predict_one, rectangle_iou, angle_diff_deg, is_correct_grasp,
)

IOU_THRESH = 0.25
ANGLE_THRESH = 30


def best_gt_match(pred_corners, gt_rects):
    """The GT rectangle this prediction came closest to, by IoU."""
    best = (-1.0, None)
    for gt in gt_rects:
        iou = rectangle_iou(pred_corners, gt)
        if iou > best[0]:
            best = (iou, gt)
    return best


def shape_limited_iou(pred_corners, gt):
    """IoU we'd get keeping our center and angle but adopting the GT's w/h.
    If this is much higher than the real IoU, the rectangle's SIZE is the
    binding constraint, not where we put it."""
    xp, yp, _, _, tp = convert(pred_corners)
    _, _, wg, hg, _ = convert(gt)
    resized = xywh_theta_to_corners(xp, yp, wg, hg, tp)
    return rectangle_iou(resized, gt)


def main(split_key="test", save_n=0):
    with open("dataset_split.json") as f:
        split = json.load(f)

    entries = split[split_key]
    backgrounds = load_backgrounds(BACKGROUNDS_DIR)

    n_total = len(entries)
    buckets = {"correct": [], "angle_only": [], "iou_only": [],
               "both": [], "skipped": [], "no_labels": []}
    scores, pred_w, gt_w, gt_h, best_ious, angle_errs, shape_ious = ([] for _ in range(7))

    for entry in entries:
        pcd_id, folder = entry["id"], entry["folder"]

        gt_rects = parse_grasp_rectangles(f"{folder}/pcd{pcd_id}cpos.txt")
        if len(gt_rects) == 0:
            buckets["no_labels"].append(pcd_id)
            continue

        pred, details = predict_one(
            folder, pcd_id, backgrounds,
            min_dist=MIN_DIST, max_dist=MAX_DIST,
            plate_thickness=PLATE_THICKNESS, return_details=True)

        if pred is None:
            buckets["skipped"].append(pcd_id)
            continue

        scores.append(details["score"])
        pred_w.append(details["width"])

        iou, gt = best_gt_match(pred, gt_rects)
        _, _, wg, hg, tg = convert(gt)
        _, _, _, _, tp = convert(pred)
        ang = angle_diff_deg(tp, tg)

        gt_w.append(wg)
        gt_h.append(hg)
        best_ious.append(iou)
        angle_errs.append(ang)
        shape_ious.append(shape_limited_iou(pred, gt))

        if any(is_correct_grasp(pred, g) for g in gt_rects):
            buckets["correct"].append(pcd_id)
        elif iou > IOU_THRESH:
            buckets["angle_only"].append((pcd_id, iou, ang))
        elif ang <= ANGLE_THRESH:
            buckets["iou_only"].append((pcd_id, iou, ang))
        else:
            buckets["both"].append((pcd_id, iou, ang))

    attempted = n_total - len(buckets["skipped"]) - len(buckets["no_labels"])
    n_ok = len(buckets["correct"])
    denom_fixed = n_total - len(buckets["no_labels"])

    print("=" * 68)
    print(f"BASELINE DIAGNOSIS — {split_key} split")
    print(f"min_dist={MIN_DIST}  max_dist={MAX_DIST}  plate_thickness={PLATE_THICKNESS}")
    print("=" * 68)
    print(f"images in split                 {n_total}")
    print(f"  no usable labels              {len(buckets['no_labels'])}")
    print(f"  no prediction produced        {len(buckets['skipped'])}")
    print(f"  predictions attempted         {attempted}")
    print()
    print(f"accuracy over attempted only    {n_ok}/{attempted} = "
          f"{n_ok/attempted:.2%}   <- what evaluate_baseline.py reports"
          if attempted else
          f"accuracy over attempted only    {n_ok}/0 = n/a (no predictions produced)")
    print(f"accuracy counting skips as fail {n_ok}/{denom_fixed} = "
          f"{n_ok/denom_fixed:.2%}   <- comparable across configurations"
          if denom_fixed else
          "accuracy counting skips as fail  n/a (no labelled images)")
    if buckets["skipped"]:
        print(f"\n  NOTE: {len(buckets['skipped'])} skipped images are silently dropped from")
        print("  the first number. Use the second when comparing runs, or a config")
        print("  that answers less often will look better than one that answers more.")

    print("\n--- failure breakdown ---")
    for name, label in [("angle_only", "IoU passed, ANGLE failed"),
                        ("iou_only",   "angle passed, IoU failed"),
                        ("both",       "both failed")]:
        b = buckets[name]
        pct = len(b) / attempted if attempted else 0
        print(f"  {label:28s} {len(b):4d}  ({pct:.1%})")

    if best_ious:
        bi = np.array(best_ious)
        ae = np.array(angle_errs)
        sc = np.array(scores)
        si = np.array(shape_ious)
        pw = np.array(pred_w)
        gw = np.array(gt_w)
        gh = np.array(gt_h)

        print("\n--- is the antipodal search working? ---")
        print(f"  score  mean {sc.mean():.2f}/2.00  median {np.median(sc):.2f}  "
              f"10th pct {np.percentile(sc,10):.2f}")
        print(f"  fraction of weak pairs (score < 1.5): {(sc < 1.5).mean():.1%}")
        print("  If this is high, the contour is the problem, not the search —")
        print("  the mask/background-subtraction stage is feeding it bad geometry.")

        print("\n--- where the IoU is going ---")
        print(f"  best IoU per image  mean {bi.mean():.3f}  median {np.median(bi):.3f}")
        print(f"  fraction above the 0.25 threshold: {(bi > IOU_THRESH).mean():.1%}")
        print(f"  IoU if we kept our center+angle but used the GT's w/h: "
              f"mean {si.mean():.3f}")
        gain = si.mean() - bi.mean()
        would_cross = int(((bi <= IOU_THRESH) & (si > IOU_THRESH)).sum())
        print(f"  -> rectangle SIZE is costing {gain:.3f} mean IoU")
        print(f"  -> images currently failing IoU that would PASS at the right "
              f"size: {would_cross} / {attempted} ({would_cross/attempted:.1%})")
        print("     Judge this against the 0.25 threshold and the current mean,")
        print("     not against 1.0 — most predictions sit close to the line, so a")
        print("     small mean shift moves a lot of images across it.")

        print("\n--- angles ---")
        print(f"  angle error  mean {ae.mean():.1f} deg  median {np.median(ae):.1f}")
        print(f"  within 30 deg: {(ae <= ANGLE_THRESH).mean():.1%}")
        near90 = ((ae > 70) & (ae < 110)).mean()
        print(f"  errors near 90 deg (convention flip, not a geometry error): {near90:.1%}")

        print("\n--- sizes ---")
        print(f"  predicted width  mean {pw.mean():5.1f} px   GT width  mean {gw.mean():5.1f} px")
        print(f"  plate_thickness  {PLATE_THICKNESS:5d} px   GT height mean {gh.mean():5.1f} px")

    print("\n--- worst near-misses (IoU passed, angle failed) ---")
    for pcd_id, iou, ang in sorted(buckets["angle_only"], key=lambda t: -t[1])[:8]:
        print(f"  pcd{pcd_id}: IoU {iou:.3f}, angle error {ang:.1f} deg")
    print("\ninspect any of these with:   python baseline.py <id>")

    if save_n:
        save_failures(buckets, backgrounds, save_n)


def save_failures(buckets, backgrounds, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = ([("angle", c) for c in buckets["angle_only"][:n]] +
             [("iou", c) for c in buckets["iou_only"][:n]])[:n]
    if not cases:
        print("\nno failures to save")
        return

    cols = min(4, len(cases))
    rows = int(np.ceil(len(cases) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)

    for ax, (kind, (pcd_id, iou, ang)) in zip(axes.ravel(), cases):
        folder = None
        with open("dataset_split.json") as f:
            for v in json.load(f).values():
                for e in v:
                    if e["id"] == pcd_id:
                        folder = e["folder"]
        img = load_rgb(folder, pcd_id)
        pred = predict_one(folder, pcd_id, backgrounds,
                           min_dist=MIN_DIST, max_dist=MAX_DIST,
                           plate_thickness=PLATE_THICKNESS)
        ax.imshow(img)
        for gt in parse_grasp_rectangles(f"{folder}/pcd{pcd_id}cpos.txt"):
            c = np.vstack([gt, gt[0]])
            ax.plot(c[:, 0], c[:, 1], "g-", lw=1)
        if pred is not None:
            c = np.vstack([pred, pred[0]])
            ax.plot(c[:, 0], c[:, 1], "r-", lw=2)
        ax.set_title(f"pcd{pcd_id}  IoU {iou:.2f}  ang {ang:.0f}d", fontsize=9)
        ax.axis("off")

    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("baseline_failures.png", dpi=110)
    print(f"\nsaved {len(cases)} failure cases to baseline_failures.png")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    split_key = args[0] if args else "test"
    save_n = 0
    if "--save" in sys.argv:
        i = sys.argv.index("--save")
        save_n = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 8
    main(split_key, save_n)