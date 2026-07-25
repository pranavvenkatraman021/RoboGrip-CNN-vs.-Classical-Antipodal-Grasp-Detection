#imports
import os
import sys
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from data_loading import load_rgb, parse_grasp_rectangles

#loads every background photo once
def load_backgrounds(backgrounds_dir):
    bg_paths = glob.glob(os.path.join(backgrounds_dir, "*r.png"))
    backgrounds = []
    for path in bg_paths:
        bg_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        backgrounds.append(bg_img)
    return backgrounds

#picks whichever background photo best matches this image's outer border
def find_best_matching_background(img, backgrounds, border=20):
    h, w = img.shape[:2]

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True

    best_score = np.inf
    best_bg = None

    for bg in backgrounds:
        if bg.shape != img.shape:
            continue

        diff = np.abs(img.astype(int) - bg.astype(int))
        border_diff = diff[border_mask].mean()

        if border_diff < best_score:
            best_score = border_diff
            best_bg = bg

    return best_bg

#defines a centered rectangular region, excluding a margin from each edge --
#since objects are always placed centrally in this dataset, the board's
#own outer edge (which lives in that margin) never needs to be considered
def get_roi_bounds(img_shape, margin_frac=0.15):
    h, w = img_shape[:2]
    x_margin = int(w * margin_frac)
    y_margin = int(h * margin_frac)
    return x_margin, w - x_margin, y_margin, h - y_margin

#RGB image into a mask (white is object, black is background)
def create_mask(img, backgrounds, margin_frac=0.15):
    best_bg = find_best_matching_background(img, backgrounds)
    if best_bg is None:
        raise ValueError("No matching background found — check image sizes match.")

    diff = np.abs(img.astype(int) - best_bg.astype(int)).sum(axis=2)
    diff = diff.astype(np.uint8)

    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #zero out everything outside a central region, so the board's own
    #outer edge (a common false-positive "biggest contour") can never
    #be picked up -- objects are always placed centrally in this dataset
    x_min, x_max, y_min, y_max = get_roi_bounds(img.shape, margin_frac)
    roi_mask = np.zeros_like(mask)
    roi_mask[y_min:y_max, x_min:x_max] = mask[y_min:y_max, x_min:x_max]

    return roi_mask

#find contour of object in the mask
def get_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)
    return largest.squeeze()

#estimates surface normal at every point along the contour
def estimate_normals(contour, step=8):
    n_points = len(contour)
    normals = []

    for i in range(n_points):
        p_before = contour[(i - step) % n_points]
        p_after = contour[(i + step) % n_points]

        tangent = p_after.astype(float) - p_before.astype(float)
        tangent_len = np.linalg.norm(tangent)

        if tangent_len == 0:
            normals.append(np.array([0.0, -1.0]))
            continue

        tangent = tangent / tangent_len
        normal = np.array([-tangent[1], tangent[0]])
        normals.append(normal)

    return normals

#searches pairs of contour points to find best antipodal pair
def find_best_antipodal_pair(contour, normals, min_dist, max_dist, sample_step=4):
    n_points = len(contour)
    best_score = -np.inf
    best_pair = None

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

    return best_pair, best_score

#converts an antipodal point pair into a grasp rectangle (x, y, w, h, theta)
def pair_to_grasp_rectangle(point_a, point_b, plate_thickness=30):
    center = (point_a + point_b) / 2
    w = np.linalg.norm(point_b - point_a)
    h = plate_thickness

    edge = point_b - point_a
    theta = np.degrees(np.arctan2(edge[1], edge[0]))

    return center[0], center[1], w, h, theta

#converts (x, y, w, h, theta) back into 4 corner points, for plotting
def xywh_theta_to_corners(x, y, w, h, theta_deg):
    theta = np.radians(theta_deg)
    dx_w, dy_w = np.cos(theta) * (w / 2), np.sin(theta) * (w / 2)
    dx_h, dy_h = -np.sin(theta) * (h / 2), np.cos(theta) * (h / 2)

    corners = np.array([
        [x - dx_w - dx_h, y - dy_w - dy_h],
        [x + dx_w - dx_h, y + dy_w - dy_h],
        [x + dx_w + dx_h, y + dy_w + dy_h],
        [x - dx_w + dx_h, y - dy_w + dy_h],
    ])
    return corners

#runs the full baseline pipeline on one object and visualizes the result:
#ground truth (green) vs. the baseline's predicted grasp (red)
def run_baseline(base_path, pcd_id, backgrounds_dir, min_dist=19, max_dist=90):
    img_path = f"{base_path}/pcd{pcd_id}r.png"
    if not os.path.exists(img_path):
        print(f"MISSING FILE: {img_path} — skipping this image.")
        return

    img = load_rgb(base_path, pcd_id)

    backgrounds = load_backgrounds(backgrounds_dir)
    mask = create_mask(img, backgrounds)
    contour = get_largest_contour(mask)

    if contour is None or len(contour) < 10:
        print(f"pcd{pcd_id}: couldn't find a usable contour — skipping.")
        return

    normals = estimate_normals(contour)
    best_pair, score = find_best_antipodal_pair(contour, normals, min_dist, max_dist)

    if best_pair is None:
        print(f"pcd{pcd_id}: no valid antipodal pair found — try adjusting min_dist/max_dist.")
        return

    point_a, point_b = best_pair
    x, y, w, h, theta = pair_to_grasp_rectangle(point_a, point_b)
    predicted_corners = xywh_theta_to_corners(x, y, w, h, theta)

    gt_rects = parse_grasp_rectangles(f"{base_path}/pcd{pcd_id}cpos.txt")

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img)

    for rect in gt_rects:
        closed = np.vstack([rect, rect[0]])
        ax.plot(closed[:, 0], closed[:, 1], 'g-', linewidth=1.5,
                 label="ground truth" if rect is gt_rects[0] else None)

    closed_pred = np.vstack([predicted_corners, predicted_corners[0]])
    ax.plot(closed_pred[:, 0], closed_pred[:, 1], 'r-', linewidth=2, label="baseline prediction")

    ax.set_title(f"pcd{pcd_id} — baseline antipodal score: {score:.2f}")
    ax.legend()
    plt.show()

#lets you inspect any single image: shows the plot AND prints the
#actual metric result against every ground truth rectangle, plus a
#detailed diagnostic on the closest-matching one
if __name__ == "__main__":
    from evaluate_baseline import predict_one, analyze_single_prediction

    pcd_id = sys.argv[1] if len(sys.argv) > 1 else "0100"
    folder_num = pcd_id[:2]
    base_path = f"/Users/pranavvenkatraman/Downloads/Cornell Grasp Data/{folder_num}"
    backgrounds_dir = "/Users/pranavvenkatraman/Downloads/Cornell Grasp Data/backgrounds"

    run_baseline(base_path, pcd_id, backgrounds_dir)

    backgrounds = load_backgrounds(backgrounds_dir)
    pred_corners = predict_one(base_path, pcd_id, backgrounds)

    analyze_single_prediction(base_path, pcd_id, pred_corners)