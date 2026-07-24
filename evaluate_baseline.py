#imports
import os
import json
import numpy as np

from shapely.geometry import Polygon
from data_loading import load_rgb, parse_grasp_rectangles, convert
from baseline import (
    load_backgrounds, create_mask, get_largest_contour, estimate_normals,
    find_best_antipodal_pair, pair_to_grasp_rectangle, xywh_theta_to_corners
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
def is_correct_grasp(pred_corners, gt_corners, iou_thresh=0.25, angle_thresh=30):
    iou = rectangle_iou(pred_corners, gt_corners)
    if iou <= iou_thresh:
        return False

    _, _, _, _, theta_pred = convert(pred_corners)
    _, _, _, _, theta_gt = convert(gt_corners)

    diff = angle_diff_deg(theta_pred, theta_gt)
    return diff <= angle_thresh

#runs just the PREDICTION part of the pipeline (no plotting) for one
#object -- shared by run_baseline's __main__ block and evaluate_baseline
def predict_one(base_path, pcd_id, backgrounds, min_dist=19, max_dist=90):
    img_path = f"{base_path}/pcd{pcd_id}r.png"
    if not os.path.exists(img_path):
        print(f"MISSING FILE: {img_path}")
        return None

    img = load_rgb(base_path, pcd_id)

    mask = create_mask(img, backgrounds)
    contour = get_largest_contour(mask)

    if contour is None or len(contour) < 10:
        return None

    normals = estimate_normals(contour)
    best_pair, score = find_best_antipodal_pair(contour, normals, min_dist, max_dist)

    if best_pair is None:
        return None

    point_a, point_b = best_pair
    x, y, w, h, theta = pair_to_grasp_rectangle(point_a, point_b)
    return xywh_theta_to_corners(x, y, w, h, theta)


#runs the baseline across the whole test split and computes accuracy
def evaluate_baseline(backgrounds_dir, split_path="dataset_split.json", min_dist=19, max_dist=90):
    with open(split_path, "r") as f:
        split = json.load(f)

    test_ids = split["test"]
    backgrounds = load_backgrounds(backgrounds_dir)

    total = 0
    correct = 0
    skipped = 0

    for entry in test_ids:
        pcd_id = entry["id"]
        folder = entry["folder"]

        pred_corners = predict_one(folder, pcd_id, backgrounds, min_dist=min_dist, max_dist=max_dist)

        if pred_corners is None:
            skipped += 1
            continue

        gt_rects = parse_grasp_rectangles(f"{folder}/pcd{pcd_id}cpos.txt")

        if len(gt_rects) == 0:
            skipped += 1
            continue

        total += 1
        matched = any(is_correct_grasp(pred_corners, gt) for gt in gt_rects)

        if matched:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    print(f"Total test images attempted: {total}")
    print(f"Skipped (no valid baseline prediction): {skipped}")
    print(f"Correct: {correct}")
    print(f"Baseline accuracy: {accuracy:.2%}")

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
    evaluate_baseline(
        backgrounds_dir="/Users/pranavvenkatraman/Downloads/Cornell Grasp Data/backgrounds"
    )