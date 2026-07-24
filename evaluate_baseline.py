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

#computes IoU between two rectangles
def rectangle_iou(corners_a, corners_b): 
    poly_a = Polygon(corners_a)
    poly_b = Polygon(corners_b)

    #check for bad predictions
    if not poly_a.is_valid or not poly_b.is_valid: 
        return 0.0
    if poly_a.area == 0 or poly_b.area == 0: 
        return 0.0

    intersection = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area

    return intersection / union

#angle difference accounts for wraparound and symmetry
def angle_diff_deg(theta_a, theta_b): 
    diff = abs(theta_a - theta_b)
    return min(diff, 180 - diff)

#checks using IoU > 0.25 and within 30 degrees 
def is_correct_grasp(pred_corners, gt_corners, iou_thresh=0.25, angle_thresh=30):
    iou = rectangle_iou(pred_corners, gt_corners)
    if iou <= iou_thresh:
        return False

    _, _, _, _, theta_pred = convert(pred_corners)
    _, _, _, _, theta_gt = convert(gt_corners)

    diff = angle_diff_deg(theta_pred, theta_gt)
    return diff <= angle_thresh

#runs baseline on one object and returns rectangle corners 
#reuses run_baseline without the plotting 
def predict_one(base_path, pcd_id, backgrounds, min_dist=30, max_dist=150):
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
    predicted_corners = xywh_theta_to_corners(x, y, w, h, theta)

    return predicted_corners

#runs baseline across whole test split computing accuracy
def evaluate_baseline(dataset_root, backgrounds_dir, split_path="dataset_split.json", min_dist = 19, max_dist = 90):
    with open(split_path, "r") as f:
        split = json.load(f)

    test_ids = split["test"]  #each entry is {"id": "...", "folder": "..."}
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

if __name__ == "__main__":
    evaluate_baseline(
        dataset_root="/Users/pranavvenkatraman/Downloads/Cornell Grasp Data",
        backgrounds_dir="/Users/pranavvenkatraman/Downloads/Cornell Grasp Data/backgrounds"
    )