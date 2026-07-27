"""
check_angle_convention.py — settle whether the ~90-degree angle errors are a
systematic convention mismatch or just genuine multimodality.

convert() reports theta along whichever rectangle edge is LONGER. That is
self-consistent, but it assumes the longer edge of a Cornell grasp rectangle is
the gripper's CLOSING direction rather than the plate direction. If that
assumption is backwards, every prediction is 90 degrees off by construction and
no amount of parameter tuning will help.

Two independent tests, neither relying on the other:

  TEST 1 (label-side): keep the prediction and the IoU exactly as they are, and
  only change which GT edge the angle is measured from. If measuring GT theta
  from the SHORT edge passes far more often, the convention is mismatched.

  TEST 2 (physics-side): for each GT rectangle, ray-march out from its centre
  along the long-edge axis and along the short-edge axis until leaving the
  object mask. At a real pair of gripper contact points the object's surface
  normals should be parallel to the closing axis. Whichever axis scores higher
  is the physical closing direction, decided from the image alone with no
  reference to our predictions.

    python check_angle_convention.py val
"""
import json
import sys

import numpy as np

from config import BACKGROUNDS_DIR, MIN_DIST, MAX_DIST, PLATE_THICKNESS
from data_loading import load_rgb, parse_grasp_rectangles, convert
from baseline import (
    load_backgrounds, create_mask, get_largest_contour, estimate_normals,
)
from evaluate_baseline import predict_one, rectangle_iou, angle_diff_deg

IOU_THRESH = 0.25
ANGLE_THRESH = 30


def long_short_angles(corners):
    """theta along the longer edge (what convert returns) and along the shorter."""
    _, _, _, _, theta_long = convert(corners)
    return theta_long, theta_long + 90.0


# ---------------------------------------------------------------- TEST 2
def contact_normal_alignment(mask, contour, normals, center, theta_deg, max_steps=400):
    """March out from center along +/- theta until leaving the mask; return how
    parallel the surface normals at those two exit points are to the axis.
    2.0 = both normals perfectly aligned with the closing axis."""
    h, w = mask.shape[:2]
    u = np.array([np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))])

    exits = []
    for sign in (+1, -1):
        p = np.array(center, dtype=float)
        for _ in range(max_steps):
            p = p + sign * u
            xi, yi = int(round(p[0])), int(round(p[1]))
            if not (0 <= xi < w and 0 <= yi < h):
                return None
            if mask[yi, xi] == 0:
                exits.append(np.array([xi, yi], dtype=float))
                break
        else:
            return None

    if len(exits) != 2:
        return None

    score = 0.0
    for e in exits:
        k = int(np.argmin(np.linalg.norm(contour.astype(float) - e, axis=1)))
        score += abs(float(np.dot(normals[k], u)))
    return score


def main(split_key="val"):
    with open("dataset_split.json") as f:
        split = json.load(f)

    entries = split[split_key]
    backgrounds = load_backgrounds(BACKGROUNDS_DIR)

    # TEST 1 counters
    n = 0
    pass_long = pass_short = 0          # angle check only
    acc_long = acc_short = 0            # full metric (IoU AND angle)
    err_long, err_short = [], []

    # TEST 2 counters
    align_long, align_short, n_align = [], [], 0

    for entry in entries:
        pcd_id, folder = entry["id"], entry["folder"]

        gt_rects = parse_grasp_rectangles(f"{folder}/pcd{pcd_id}cpos.txt")
        if len(gt_rects) == 0:
            continue

        # ---- TEST 1 -----------------------------------------------------
        pred = predict_one(folder, pcd_id, backgrounds,
                           min_dist=MIN_DIST, max_dist=MAX_DIST,
                           plate_thickness=PLATE_THICKNESS)
        if pred is not None:
            n += 1
            theta_pred, _ = long_short_angles(pred)

            best_l = best_s = 180.0
            ok_l = ok_s = False
            for gt in gt_rects:
                iou = rectangle_iou(pred, gt)
                t_long, t_short = long_short_angles(gt)
                d_l = angle_diff_deg(theta_pred, t_long)
                d_s = angle_diff_deg(theta_pred, t_short)
                best_l, best_s = min(best_l, d_l), min(best_s, d_s)
                if iou > IOU_THRESH and d_l <= ANGLE_THRESH:
                    ok_l = True
                if iou > IOU_THRESH and d_s <= ANGLE_THRESH:
                    ok_s = True

            err_long.append(best_l)
            err_short.append(best_s)
            pass_long += best_l <= ANGLE_THRESH
            pass_short += best_s <= ANGLE_THRESH
            acc_long += ok_l
            acc_short += ok_s

        # ---- TEST 2 -----------------------------------------------------
        try:
            img = load_rgb(folder, pcd_id)
            mask = create_mask(img, backgrounds)
            contour = get_largest_contour(mask)
        except Exception:
            continue
        if contour is None or len(contour) < 10:
            continue
        normals = np.array(estimate_normals(contour))

        for gt in gt_rects:
            cx, cy, _, _, t_long = convert(gt)
            if mask[int(round(cy)), int(round(cx))] == 0:
                continue                      # centre not on the object
            a = contact_normal_alignment(mask, contour, normals, (cx, cy), t_long)
            b = contact_normal_alignment(mask, contour, normals, (cx, cy), t_long + 90)
            if a is None or b is None:
                continue
            align_long.append(a)
            align_short.append(b)
            n_align += 1

    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"ANGLE CONVENTION TEST — {split_key} split")
    print("=" * 70)

    print("\nTEST 1 — which GT edge does our predicted angle actually agree with?")
    print(f"  predictions compared: {n}")
    print(f"    GT theta from LONG edge  (current):  angle passes "
          f"{pass_long}/{n} = {pass_long/max(n,1):.1%}   "
          f"median error {np.median(err_long):.1f} deg")
    print(f"    GT theta from SHORT edge (rotated):  angle passes "
          f"{pass_short}/{n} = {pass_short/max(n,1):.1%}   "
          f"median error {np.median(err_short):.1f} deg")
    print(f"\n    full metric, current convention:  {acc_long}/{n} = "
          f"{acc_long/max(n,1):.2%}")
    print(f"    full metric, rotated convention:  {acc_short}/{n} = "
          f"{acc_short/max(n,1):.2%}")

    print("\nTEST 2 — which axis is physically a gripper closing direction?")
    if n_align:
        al, as_ = np.array(align_long), np.array(align_short)
        print(f"  GT rectangles tested: {n_align}")
        print(f"    along LONG edge   normal alignment {al.mean():.2f}/2.00 "
              f"(median {np.median(al):.2f})")
        print(f"    along SHORT edge  normal alignment {as_.mean():.2f}/2.00 "
              f"(median {np.median(as_):.2f})")
        print(f"    long edge wins on {100*(al > as_).mean():.1f}% of rectangles")
    else:
        print("  no usable rectangles (masks may not cover the labelled centres)")

    print("\n" + "-" * 70)
    print("HOW TO READ THIS")
    print("-" * 70)
    verdict_1 = acc_short > acc_long * 1.25
    verdict_2 = n_align and np.mean(align_short) > np.mean(align_long) * 1.15

    if verdict_1 and verdict_2:
        print("Both tests agree: the convention is 90 degrees off. convert()'s")
        print("longer-edge rule is picking the gripper PLATE, not the closing")
        print("direction. Fix by measuring theta from the SHORT edge in convert(),")
        print("consistently for predictions and ground truth. This is a large,")
        print("systematic gain, not a tuning matter.")
    elif verdict_1 != verdict_2:
        print("The two tests DISAGREE. Do not change convert() on this evidence —")
        print("one of them is being driven by something else. Look at")
        print("baseline_failures.png before acting.")
    else:
        print("Neither test supports a convention flip. The ~90-degree errors are")
        print("then genuine disagreement about WHICH valid grasp to take (a mug can")
        print("be grasped across the rim or by the handle), not a systematic offset.")
        print("That is a multimodality problem, and it is exactly the thing a")
        print("single-rectangle classical method cannot fix. Worth saying plainly")
        print("in the write-up: it is a real limitation of the approach, and it is")
        print("the specific weakness the CNN is expected to beat.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "val")