#imports
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data_loading import load_rgb, parse_grasp_rectangles
from baseline import (
    load_backgrounds, create_mask, get_largest_contour,
    estimate_normals, pair_to_grasp_rectangle, xywh_theta_to_corners, get_roi_bounds
)
from evaluate_baseline import is_correct_grasp

#the exact final-tuned parameters locked in for the project
MIN_DIST = 19
MAX_DIST = 90
PLATE_THICKNESS = 30
MARGIN_FRAC = 0.15   # same ROI margin used everywhere else in the pipeline


#a LOCAL copy of the antipodal search, used only for this visualization
#script -- identical logic to baseline.py's find_best_antipodal_pair, but
#ALSO returns the exact normal_a/normal_b used at the winning pair, instead
#of trying to guess them afterward via a nearest-point search (which is
#unreliable on complex/looping contours, like a tangled cord)
def find_best_antipodal_pair_with_normals(contour, normals, min_dist, max_dist, sample_step=4):
    n_points = len(contour)
    best_score = -np.inf
    best_pair = None
    best_normals = None

    indices = range(0, n_points, sample_step)

    for i in indices:
        for j in indices:
            if i >= j:
                continue

            point_a = contour[i].astype(float)
            point_b = contour[j].astype(float)

            dist = np.linalg.norm(point_b - point_a)
            if dist < min_dist or dist > max_dist:
                continue

            direction_ab = (point_b - point_a) / dist
            normal_a = normals[i]
            normal_b = normals[j]

            score_a = np.dot(normal_a, direction_ab)
            score_b = np.dot(normal_b, -direction_ab)
            score = score_a + score_b

            if score > best_score:
                best_score = score
                best_pair = (point_a, point_b)
                best_normals = (normal_a, normal_b)   # <-- captured directly, no guessing later

    return best_pair, best_normals, best_score


#draws one full annotated image: ROI boundary, the correct predicted
#rectangle, and both outward-pointing surface normal arrows
def draw_annotated_image(rgb, roi_bounds, pred_corners, point_a, point_b,
                          normal_a, normal_b, save_path, pcd_id, arrow_length=45):
    x_min, x_max, y_min, y_max = roi_bounds

    fig, ax = plt.subplots(1, figsize=(7, 7))
    ax.imshow(rgb)
    ax.set_title(f"pcd{pcd_id}", fontsize=13, fontweight="bold")

    # ---- 1. ROI boundary: dashed rectangle outline ----
    roi_rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=2.5, edgecolor="#FFB400", facecolor="none", linestyle="--"
    )
    ax.add_patch(roi_rect)

    # ---- 2. the single correct predicted grasp rectangle ----
    closed = np.vstack([pred_corners, pred_corners[0]])
    ax.plot(closed[:, 0], closed[:, 1], color="#39D353", linewidth=3)

    # ---- 3. the two surface normal vectors, drawn pointing OUTWARD ----
    # flip sign since the search internally scores normals pointing TOWARD
    # each other (mechanically meaningful for the algorithm), but a diagram
    # should show them pointing away from the object -- the standard,
    # readable convention for illustrating surface normals
    for point, normal in [(point_a, normal_a), (point_b, normal_b)]:
        end_point = point - normal * arrow_length
        ax.annotate(
            "", xy=(end_point[0], end_point[1]), xytext=(point[0], point[1]),
            arrowprops=dict(arrowstyle="->", color="#FF4D4D", linewidth=2.5)
        )
        ax.plot(point[0], point[1], "o", color="#FF4D4D", markersize=6)

    # clean, slide-ready presentation: no axis ticks/labels
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


#scans the ENTIRE test set, collects every correct prediction, ranks them
#by how clean/textbook their antipodal pair actually is (opposition score
#close to -1 = normals point cleanly opposite each other), and generates
#the slide image using the single BEST one -- not just the first one found
def generate_best_slide_image(backgrounds_dir, split_path="dataset_split.json",
                               output_path="antipodal_slide_image.png", top_n_to_show=5):
    with open(split_path, "r") as f:
        split = json.load(f)

    backgrounds = load_backgrounds(backgrounds_dir)
    candidates = []   # will hold (opposition_score, pcd_id, all the drawing data)

    for entry in split["test"]:
        pcd_id, folder = entry["id"], entry["folder"]

        img_path = f"{folder}/pcd{pcd_id}r.png"
        if not os.path.exists(img_path):
            continue

        rgb = load_rgb(folder, pcd_id)

        mask = create_mask(rgb, backgrounds, margin_frac=MARGIN_FRAC)
        contour = get_largest_contour(mask)
        if contour is None or len(contour) < 10:
            continue

        normals = estimate_normals(contour)
        best_pair, best_normals, score = find_best_antipodal_pair_with_normals(contour, normals, MIN_DIST, MAX_DIST)
        if best_pair is None:
            continue

        point_a, point_b = best_pair
        normal_a, normal_b = best_normals
        x, y, w, h, theta = pair_to_grasp_rectangle(point_a, point_b, plate_thickness=PLATE_THICKNESS)
        pred_corners = xywh_theta_to_corners(x, y, w, h, theta)

        gt_rects = parse_grasp_rectangles(f"{folder}/pcd{pcd_id}cpos.txt")
        if len(gt_rects) == 0:
            continue

        is_correct = any(is_correct_grasp(pred_corners, gt) for gt in gt_rects)
        if not is_correct:
            continue

        # the actual "how clean does this pair look" metric --
        # close to -1 = normals point cleanly opposite each other
        opposition = np.dot(normal_a, normal_b)

        candidates.append((opposition, pcd_id, rgb, pred_corners, point_a, point_b, normal_a, normal_b))

    if not candidates:
        print("No correct examples found at all.")
        return None

    # sort so the MOST NEGATIVE opposition (cleanest antipodal pair) comes first
    candidates.sort(key=lambda c: c[0])

    print(f"Found {len(candidates)} correct examples. Top {min(top_n_to_show, len(candidates))} by opposition quality:")
    for opp, pcd_id, *_ in candidates[:top_n_to_show]:
        print(f"  pcd{pcd_id}: opposition = {opp:.2f}")

    # generate the image for the single BEST one
    best = candidates[0]
    _, pcd_id, rgb, pred_corners, point_a, point_b, normal_a, normal_b = best

    roi_bounds = get_roi_bounds(rgb.shape, margin_frac=MARGIN_FRAC)
    print(f"\nUsing pcd{pcd_id} (best opposition: {best[0]:.2f}) for the slide image.")

    draw_annotated_image(
        rgb, roi_bounds, pred_corners, point_a, point_b,
        normal_a, normal_b, output_path, pcd_id
    )
    return pcd_id


if __name__ == "__main__":
    generate_best_slide_image(
        backgrounds_dir="/Users/pranavvenkatraman/Downloads/Cornell Grasp Data/backgrounds",
        output_path="antipodal_slide_image.png"
    )