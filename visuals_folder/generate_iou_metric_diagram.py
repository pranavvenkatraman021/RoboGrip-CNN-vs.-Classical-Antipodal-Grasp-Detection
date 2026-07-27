#imports
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from data_loading import load_rgb, parse_grasp_rectangles, convert
from evaluate_baseline import rectangle_iou, angle_diff_deg, is_correct_grasp
from baseline import (
    load_backgrounds, create_mask, get_largest_contour,
    estimate_normals, find_best_antipodal_pair,
    pair_to_grasp_rectangle, xywh_theta_to_corners
)
from config import (
    BACKGROUNDS_DIR, MIN_DIST, MAX_DIST,
    PLATE_THICKNESS, WIDTH_MARGIN, TIE_BREAK, TIE_TOL, MASK_CHECK
)


#draws the IoU diagram for one real image:
#  - the actual object photo as the background
#  - one ground-truth rectangle in green
#  - the baseline's predicted rectangle in red
#  - the intersection region shaded in blue
#  - the IoU formula below the image
#  - pcd id, IoU value, and angle difference labelled top-right
def make_iou_figure(rgb, gt_corners, pred_corners, pcd_id, save_path):
    img_h, img_w = rgb.shape[:2]
    aspect = img_w / img_h

    fig_width_in  = 6.5
    img_height_in = fig_width_in / aspect

    title_height_in   = 0.45
    formula_height_in = 0.55   # space below the image for the IoU formula
    fig_height_in = title_height_in + img_height_in + formula_height_in

    fig = plt.figure(figsize=(fig_width_in, fig_height_in))

    #image axes: exact position so no letterboxing gap (same technique as
    #generate_pos_neg_images.py)
    ax_bottom = formula_height_in / fig_height_in
    ax_height = img_height_in    / fig_height_in
    ax = fig.add_axes([0.0, ax_bottom, 1.0, ax_height])

    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    #compute the intersection polygon so we can shade it on the actual image
    gt_poly   = Polygon(gt_corners)
    pred_poly = Polygon(pred_corners)
    inter     = gt_poly.intersection(pred_poly)

    #shade the intersection region directly on the image axes
    if not inter.is_empty:
        inter_xy = np.array(inter.exterior.coords)
        ax.fill(inter_xy[:, 0], inter_xy[:, 1],
                color="#2980B9", alpha=0.45, zorder=2)
        #"Intersection" label at the centroid of the overlap region
        ax.text(inter.centroid.x, inter.centroid.y, "Intersection",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="white", zorder=4)

    #ground truth rectangle in green
    gt_closed = np.vstack([gt_corners, gt_corners[0]])
    ax.plot(gt_closed[:, 0], gt_closed[:, 1],
            color="#27AE60", linewidth=2.8, zorder=3, label="Ground truth")

    #predicted rectangle in red
    pred_closed = np.vstack([pred_corners, pred_corners[0]])
    ax.plot(pred_closed[:, 0], pred_closed[:, 1],
            color="#E74C3C", linewidth=2.8, zorder=3, label="Prediction")

    #legend inside the image axes -- top left, no box
    ax.legend(loc="upper left", fontsize=8.5, frameon=True,
              framealpha=0.75, edgecolor="none",
              handlelength=1.4, handleheight=0.9)

    #pcd label and metric values -- top right corner of the image
    iou  = rectangle_iou(pred_corners, gt_corners)
    *_, theta_pred = convert(pred_corners)
    *_, theta_gt   = convert(gt_corners)
    adiff = angle_diff_deg(theta_pred, theta_gt)
    correct = is_correct_grasp(pred_corners, gt_corners)

    verdict = "CORRECT ✓" if correct else "INCORRECT ✗"
    info_str = (f"pcd{pcd_id}\n"
                f"IoU = {iou:.2f}  (threshold > 0.25)\n"
                f"Δangle = {adiff:.1f}°  (threshold ≤ 30°)\n"
                f"{verdict}")
    ax.text(img_w - 8, 8, info_str,
            ha="right", va="top", fontsize=13, color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#20344F",
                      alpha=0.70, edgecolor="none"),
            linespacing=1.9, zorder=5)

    #title centred in the strip above the image
    title_y = ax_bottom + ax_height + (title_height_in * 0.5) / fig_height_in
    fig.text(0.5, title_y, f"pcd{pcd_id} — IoU metric example",
             ha="center", va="center", fontsize=12, fontweight="bold")

    #IoU formula centred in the strip below the image
    fig.text(0.5, (formula_height_in * 0.52) / fig_height_in,
             r"$\mathrm{IoU}\ =\ \dfrac{\mathrm{Intersection\ area}}{\mathrm{Union\ area}}\ >\ 0.25$"
             r"$\quad\mathrm{AND}\quad\Delta\theta\ \leq\ 30°\ \Rightarrow\ \mathrm{CORRECT}$",
             ha="center", va="center", fontsize=9.5, color="#1A252F")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


#scans the test split for a real image where:
#  - the baseline produces a valid prediction
#  - there is at least one labelled ground-truth rectangle
#  - the IoU is above 0.25 (so there IS an intersection worth shading)
#  - the angle difference is also > 0 (so both conditions are visible)
#prefers a near-miss (IoU passes but angle close to 30 deg, or vice versa)
#because that is more visually informative than a clean pass
def generate_iou_diagram(split_path="dataset_split.json",
                         output_path="iou_metric_diagram.png"):
    with open(split_path, "r") as f:
        split = json.load(f)

    backgrounds = load_backgrounds(BACKGROUNDS_DIR)

    candidates = []   # (interest_score, pcd_id, rgb, gt, pred)

    for entry in split["test"]:
        pcd_id, folder = entry["id"], entry["folder"]

        img_path = f"{folder}/pcd{pcd_id}r.png"
        if not os.path.exists(img_path):
            continue

        gt_rects = parse_grasp_rectangles(f"{folder}/pcd{pcd_id}cpos.txt")
        if len(gt_rects) == 0:
            continue

        rgb     = load_rgb(folder, pcd_id)
        mask    = create_mask(rgb, backgrounds)
        contour = get_largest_contour(mask)
        if contour is None or len(contour) < 10:
            continue

        normals   = estimate_normals(contour)
        best_pair, score = find_best_antipodal_pair(
            contour, normals, MIN_DIST, MAX_DIST,
            tie_break=TIE_BREAK, tie_tol=TIE_TOL,
            mask=mask if MASK_CHECK else None)
        if best_pair is None:
            continue

        point_a, point_b = best_pair
        x, y, w, h, theta = pair_to_grasp_rectangle(
            point_a, point_b, PLATE_THICKNESS, width_margin=WIDTH_MARGIN)
        pred_corners = xywh_theta_to_corners(x, y, w, h, theta)

        #find the GT rectangle with the best IoU against our prediction
        best_iou = -1
        best_gt  = None
        for gt in gt_rects:
            iou = rectangle_iou(pred_corners, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt  = gt

        if best_iou < 0.20:
            continue   # no meaningful overlap -- not illustrative

        *_, tp = convert(pred_corners)
        *_, tg = convert(best_gt)
        adiff = angle_diff_deg(tp, tg)

        #interest score: prefer images where IoU is close to the threshold
        #(either side of 0.25) so the shaded region is neither tiny nor huge
        interest = -abs(best_iou - 0.35)

        candidates.append((interest, pcd_id, rgb, best_gt, pred_corners))

    if not candidates:
        print("No suitable test-set image found.")
        return None

    candidates.sort(key=lambda c: -c[0])   # best interest score first

    print(f"Found {len(candidates)} candidate images. Top 5:")
    for interest, pcd_id, rgb, gt, pred in candidates[:5]:
        iou   = rectangle_iou(pred, gt)
        *_, tp = convert(pred); *_, tg = convert(gt)
        adiff = angle_diff_deg(tp, tg)
        print(f"  pcd{pcd_id}: IoU={iou:.3f}, Δangle={adiff:.1f}°, "
              f"correct={is_correct_grasp(pred, gt)}")

    _, pcd_id, rgb, best_gt, pred_corners = candidates[0]
    print(f"\nUsing pcd{pcd_id} for the IoU diagram.")

    make_iou_figure(rgb, best_gt, pred_corners, pcd_id, output_path)
    return pcd_id


if __name__ == "__main__":
    generate_iou_diagram(output_path="iou_metric_diagram.png")