#imports
import os
import sys
import json
import numpy as np

from shapely.geometry import Polygon
from data_loading import load_rgb, parse_grasp_rectangles, convert
from config import (BACKGROUNDS_DIR, MIN_DIST, MAX_DIST, PLATE_THICKNESS,
                    WIDTH_MARGIN, TIE_BREAK, TIE_TOL, MASK_CHECK)
from baseline import (
    load_backgrounds, create_mask, get_largest_contour, estimate_normals,
    find_best_antipodal_pair, pair_to_grasp_rectangle, xywh_theta_to_corners,
    enforce_min_dist
)

#computes IoU (intersection over union) between two rectangles,
#each given as a (4,2) array of corner points
def rectangle_iou(corners_a, corners_b):
    poly_a = Polygon(corners_a)
    poly_b = Polygon(corners_b)

    if not poly_a.is_valid or not poly_b.is_valid:
        return 0.0
    if poly_a.area == 0 or poly_b.area == 0:
        return 0.0

    intersection = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    return intersection / union

#angle difference that accounts for wraparound AND grasp symmetry
#(a grasp rectangle rotated 180 degrees is physically the same grasp)
def angle_diff_deg(theta_a, theta_b):
    diff = abs(theta_a - theta_b) % 180
    return min(diff, 180 - diff)

#checks if a prediction counts as correct against ONE ground truth
#rectangle, using the standard metric: IoU > 0.25 AND angle within 30°
#
#UNCHANGED ON PURPOSE -- this exact function is shared with the CNN
#(train.py and evaluate_cnn.py both import it), so the two methods stay
#graded by identical code. The B2 fix is applied upstream instead.
def is_correct_grasp(pred_corners, gt_corners, iou_thresh=0.25, angle_thresh=30):
    iou = rectangle_iou(pred_corners, gt_corners)
    if iou <= iou_thresh:
        return False

    _, _, _, _, theta_pred = convert(pred_corners)
    _, _, _, _, theta_gt = convert(gt_corners)

    diff = angle_diff_deg(theta_pred, theta_gt)
    return diff <= angle_thresh

#runs just the PREDICTION part of the pipeline (no plotting) for one object.
#
#FIX: plate_thickness is now an explicit parameter and is explicitly passed
#down to pair_to_grasp_rectangle. Previously predict_one called it with no
#plate_thickness argument at all, so the function default silently decided the
#value regardless of what had been tuned. Returning the diagnostics too, so a
#sweep can see WHY a combination scored the way it did.
def predict_one(base_path, pcd_id, backgrounds,
                min_dist=MIN_DIST, max_dist=MAX_DIST, plate_thickness=PLATE_THICKNESS,
                width_margin=WIDTH_MARGIN, tie_break=TIE_BREAK, tie_tol=TIE_TOL,
                mask_check=MASK_CHECK, return_details=False):
    fail = (None, None) if return_details else None

    img_path = f"{base_path}/pcd{pcd_id}r.png"
    if not os.path.exists(img_path):
        print(f"MISSING FILE: {img_path}")
        return fail

    min_dist = enforce_min_dist(min_dist, plate_thickness)

    img = load_rgb(base_path, pcd_id)

    mask = create_mask(img, backgrounds)
    contour = get_largest_contour(mask)

    if contour is None or len(contour) < 10:
        return fail

    normals = estimate_normals(contour)
    best_pair, score = find_best_antipodal_pair(
        contour, normals, min_dist, max_dist,
        tie_break=tie_break, tie_tol=tie_tol,
        mask=mask if mask_check else None)

    if best_pair is None:
        return fail

    point_a, point_b = best_pair
    x, y, w, h, theta = pair_to_grasp_rectangle(point_a, point_b, plate_thickness,
                                                width_margin=width_margin)
    corners = xywh_theta_to_corners(x, y, w, h, theta)

    if return_details:
        return corners, {"score": score, "width": w, "theta": theta}
    return corners


#runs the baseline across one split and computes accuracy.
#
#split_key defaults to "test" but MUST be set to "val" while tuning --
#the test set is touched exactly once, at the very end.
def evaluate_baseline(backgrounds_dir=BACKGROUNDS_DIR, split_path="dataset_split.json",
                      min_dist=MIN_DIST, max_dist=MAX_DIST, plate_thickness=PLATE_THICKNESS,
                      width_margin=WIDTH_MARGIN, tie_break=TIE_BREAK,
                      split_key="test", verbose=True):
    with open(split_path, "r") as f:
        split = json.load(f)

    entries = split[split_key]
    backgrounds = load_backgrounds(backgrounds_dir)

    effective_min = enforce_min_dist(min_dist, plate_thickness, warn=verbose)

    total = 0
    correct = 0
    skipped = 0
    scores = []
    widths = []

    for entry in entries:
        pcd_id = entry["id"]
        folder = entry["folder"]

        pred_corners, details = predict_one(
            folder, pcd_id, backgrounds,
            min_dist=effective_min, max_dist=max_dist, plate_thickness=plate_thickness,
            width_margin=width_margin, tie_break=tie_break, return_details=True,
        )

        if pred_corners is None:
            skipped += 1
            continue

        gt_rects = parse_grasp_rectangles(f"{folder}/pcd{pcd_id}cpos.txt")

        if len(gt_rects) == 0:
            skipped += 1
            continue

        total += 1
        scores.append(details["score"])
        widths.append(details["width"])

        matched = any(is_correct_grasp(pred_corners, gt) for gt in gt_rects)

        if matched:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    if verbose:
        print(f"\nsplit: {split_key}   "
              f"min_dist={effective_min}  max_dist={max_dist}  plate_thickness={plate_thickness}")
        print(f"          width_margin={width_margin}  tie_break='{tie_break}'"
              f"  mask_check={MASK_CHECK}")
        denom = total + skipped
        print(f"Total images attempted: {total}")
        print(f"Skipped (no valid baseline prediction): {skipped}")
        print(f"Correct: {correct}")
        print(f"Baseline accuracy (attempted only): {accuracy:.2%}")
        print(f"Baseline accuracy (skips = failures): "
              f"{correct/denom if denom else 0:.2%}   <- use this to compare runs")

        if scores:
            scores = np.array(scores)
            widths = np.array(widths)
            #diagnostics for findings B1 and B2
            print(f"\n  antipodal score  mean {scores.mean():.2f} / 2.00, "
                  f"median {np.median(scores):.2f}, min {scores.min():.2f}")
            print(f"  weak grasps (score < 1.0): {(scores < 1.0).mean():.1%} of predictions")
            print(f"  predicted width  mean {widths.mean():.1f} px, "
                  f"median {np.median(widths):.1f}, min {widths.min():.1f}, max {widths.max():.1f}")
            n_flip = int((widths < plate_thickness).sum())
            print(f"  predictions narrower than the plate (B2 90-degree flip): "
                  f"{n_flip} / {total}"
                  + ("  <-- should be 0" if n_flip else "  (good)"))

    return accuracy

#diagnostic helper for inspecting ONE image in detail -- prints IoU and
#correctness against every ground truth rectangle, then digs deeper
#into whichever one had the BEST IoU (the "closest match"), since
#that's the one most likely to reveal WHY a near-miss failed
def analyze_single_prediction(base_path, pcd_id, pred_corners):
    if pred_corners is None:
        print(f"No prediction produced for pcd{pcd_id}.")
        return

    gt_rects = parse_grasp_rectangles(f"{base_path}/pcd{pcd_id}cpos.txt")

    print(f"\n--- Metrics for pcd{pcd_id} ---")
    any_correct = False
    best_iou = -1
    best_gt_idx = None

    for i, gt in enumerate(gt_rects):
        iou = rectangle_iou(pred_corners, gt)
        correct = is_correct_grasp(pred_corners, gt)
        print(f"GT rectangle {i}: IoU={iou:.3f}, correct={correct}")

        if correct:
            any_correct = True
        if iou > best_iou:
            best_iou = iou
            best_gt_idx = i

    print(f"Overall: {'CORRECT' if any_correct else 'INCORRECT'}")

    if best_gt_idx is None:
        print("\nNo ground truth rectangles found to compare against.")
        return

    gt_best = gt_rects[best_gt_idx]

    x_pred, y_pred, w_pred, h_pred, theta_pred = convert(pred_corners)
    x_gt, y_gt, w_gt, h_gt, theta_gt = convert(gt_best)
    diff = angle_diff_deg(theta_pred, theta_gt)

    print(f"\n--- Closest match: GT rectangle {best_gt_idx} (IoU={best_iou:.3f}) ---")
    print(f"Predicted:     w={w_pred:.1f}, h={h_pred:.1f}, theta={theta_pred:.1f}°")
    print(f"Ground truth:  w={w_gt:.1f}, h={h_gt:.1f}, theta={theta_gt:.1f}°")
    print(f"Angle difference: {diff:.1f}°")

if __name__ == "__main__":
    #  python evaluate_baseline.py          -> test split (final number, run once)
    #  python evaluate_baseline.py val      -> validation split (safe to re-run)
    split_key = sys.argv[1] if len(sys.argv) > 1 else "test"
    evaluate_baseline(split_key=split_key)