#imports
import os
import sys
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from data_loading import load_rgb, parse_grasp_rectangles
from config import (DATA_ROOT, BACKGROUNDS_DIR, MIN_DIST, MAX_DIST,
                    PLATE_THICKNESS, WIDTH_MARGIN, TIE_BREAK, TIE_TOL,
                    MASK_CHECK, PLATE_CAP_RATIO)

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

    #building a boolean mask that is true only on the border
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

        #per-pixel absolute difference across color channels
        diff = np.abs(img.astype(int) - bg.astype(int))
        border_diff = diff[border_mask].mean()

        #smallest border difference
        if border_diff < best_score:
            best_score = border_diff
            best_bg = bg

    return best_bg

#returns pixel coords of central (region of interest, ROI)
def get_roi_bounds(img_shape, margin_frac = 0.15):
    h, w = img_shape[:2]
    x_margin = int(w * margin_frac)
    y_margin = int(h * margin_frac)
    return x_margin, w - x_margin, y_margin, h - y_margin

#RGB image into a mask (white is object, black is background)
def create_mask(img, backgrounds, margin_frac=0.15):
    #find closest background photo
    best_bg = find_best_matching_background(img, backgrounds)
    if best_bg is None:
        raise ValueError("No matching background found — check image sizes match.")

    #collapse into single grayscale difference map 
    diff = np.abs(img.astype(int) - best_bg.astype(int)).sum(axis = 2)
    diff = diff.astype(np.uint8)

    #use OTSU to find the treshold value that maximizes 
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    #morphological opening removes tiny white specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #morphological closing finds small holes 
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #create blanck canvas and only copy central ROI region 
    x_min, x_max, y_min, y_max = get_roi_bounds(img.shape, margin_frac)
    roi_mask = np.zeros_like(mask)
    roi_mask[y_min:y_max, x_min:x_max] = mask[y_min:y_max, x_min:x_max]

    return roi_mask

#find contour of object in the mask
def get_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return None

    #taking max to get the object
    largest = max(contours, key = cv2.contourArea)
    #get a clean (N, 2) array of [x, y] points 
    return largest.squeeze()

#estimates surface normal at every point on the contour 
def estimate_normals(contour, step=8):
    n_points = len(contour)
    #centroid is avg position of all contour points (center of object)
    centroid = contour.astype(float).mean(axis=0)
    normals = []

    for i in range(n_points):
        #wrap around at the ends to get valid tangent estimate
        p_before = contour[(i - step) % n_points].astype(float)
        p_after = contour[(i + step) % n_points].astype(float)

        #tangent is direction along contour at point i
        tangent = p_after - p_before
        tangent_len = np.linalg.norm(tangent)

        #is the step-8 neighbors are the same point 
        if tangent_len == 0:
            normals.append(np.array([0.0, -1.0]))
            continue

        #normalize unit length
        tangent = tangent / tangent_len
        #perpendiculat to tangent (could point in or out)
        normal = np.array([-tangent[1], tangent[0]])

        #force outwards using dot product with vector from centroid to point
        if np.dot(normal, contour[i].astype(float) - centroid) < 0:
            normal = -normal

        normals.append(normal)

    return normals

#finds best pair of contact points for gripper to close on 
def find_best_antipodal_pair(contour, normals, min_dist, max_dist, sample_step = 4,
                             tie_break = TIE_BREAK, tie_tol = TIE_TOL, mask = None):

    n_points = len(contour)
    centroid = contour.astype(float).mean(axis=0)

    candidates = []
    #sample every 4th for runtime 
    indices = range(0, n_points, sample_step)

    for i in indices:
        for j in indices:
            if i >= j:
                continue

            point_a = contour[i].astype(float)
            point_b = contour[j].astype(float)

            dist = np.linalg.norm(point_b - point_a)

            #skip pairs that are too close or too far
            if dist < min_dist or dist > max_dist:
                continue

            #reject pairs with midpoint outside of object 
            if mask is not None:
                mid = ((point_a + point_b) / 2).astype(int)
                if not (0 <= mid[1] < mask.shape[0] and 0 <= mid[0] < mask.shape[1]):
                    continue
                if mask[mid[1], mid[0]] == 0:
                    continue

            #unit vector from A toward B
            direction_ab = (point_b - point_a) / dist

            #antipodal score, how well they oppose
            #normal of a vs A to B
            score_a = -np.dot(normals[i], direction_ab)
            #normal of b vs B to A
            score_b = -np.dot(normals[j], -direction_ab)

            #store score and opening width
            candidates.append((score_a + score_b, point_a, point_b, dist))

    if not candidates:
        return None, -np.inf

    #find best score
    top = max(c[0] for c in candidates)
    #collect every pair within tie_tol of it 
    finalists = [c for c in candidates if c[0] >= top - tie_tol]

    #choose amont tied candidates
    if tie_break == "centroid":
        #midpoint nearest the object
        pick = min(finalists, key=lambda c: np.linalg.norm((c[1] + c[2]) / 2 - centroid))
    elif tie_break == "narrow":
        #smallest opening
        pick = min(finalists, key=lambda c: c[3])
    elif tie_break == "wide":
        #largest opening
        pick = max(finalists, key=lambda c: c[3])
    else:            
        #first pair that contour traversal reached 
        pick = finalists[0]

    return (pick[1], pick[2]), float(pick[0])

#counts how many pairs score within tie_tol of the best score
#finding importance of tie-break criteria
def count_near_optimal(contour, normals, min_dist, max_dist, sample_step=4,
                       tie_tol = TIE_TOL):
    scores = []
    n_points = len(contour)
    indices = range(0, n_points, sample_step)
    for i in indices:
        for j in indices:
            if i >= j:
                continue
            a, b = contour[i].astype(float), contour[j].astype(float)
            d = np.linalg.norm(b - a)
            if d < min_dist or d > max_dist:
                continue
            u = (b - a) / d
            #same score formula 
            scores.append(-np.dot(normals[i], u) - np.dot(normals[j], -u))
    if not scores:
        return 0, 0
    scores = np.array(scores)
    #count how many are within tie_tol
    return int((scores >= scores.max() - tie_tol).sum()), len(scores)

#converts two contact points into (x, y, w, h theta) grasp rectangle
def pair_to_grasp_rectangle(point_a, point_b, plate_thickness = PLATE_THICKNESS,
                            width_margin = WIDTH_MARGIN,
                            plate_cap_ratio = PLATE_CAP_RATIO):
    center = (point_a + point_b) / 2
    w = np.linalg.norm(point_b - point_a) * width_margin
    h = plate_thickness

    #keep h below w so the longer edge is always closing direction
    if plate_cap_ratio is not None:
        h = min(h, w * plate_cap_ratio)

    #angle of A to B vector in degrees
    edge = point_b - point_a
    theta = np.degrees(np.arctan2(edge[1], edge[0]))

    return center[0], center[1], w, h, theta

#converts (x, y, w, h, theta) back into 4 corner points for plotting
def xywh_theta_to_corners(x, y, w, h, theta_deg):
    theta = np.radians(theta_deg)
    #vector length w/2 along closing direction
    dx_w, dy_w = np.cos(theta) * (w / 2), np.sin(theta) * (w / 2)
    #vector of length h/2 perp to the closing direction
    dx_h, dy_h = -np.sin(theta) * (h / 2), np.cos(theta) * (h / 2)

    corners = np.array([
        [x - dx_w - dx_h, y - dy_w - dy_h],
        [x + dx_w - dx_h, y + dy_w - dy_h],
        [x + dx_w + dx_h, y + dy_w + dy_h],
        [x - dx_w + dx_h, y - dy_w + dy_h],
    ])
    return corners

#passes min_dist through unchanged when plate_cap_ratio is set 
def enforce_min_dist(min_dist, plate_thickness, warn = False,
                     plate_cap_ratio = PLATE_CAP_RATIO):
    #no need to restrict min_dist
    if plate_cap_ratio is not None:
        return min_dist
    if min_dist < plate_thickness:
        if warn:
            print(f"min_dist={min_dist} < plate_thickness={plate_thickness}; "
                  f"clamping min_dist to {plate_thickness} to avoid the 90-degree theta flip.")
        return plate_thickness
    return min_dist

#runs the full baseline pipeline on one object and visualizes the result:
#ground truth (green) vs. the baseline's predicted grasp (red), yellow dots = two points
def run_baseline(base_path, pcd_id, backgrounds_dir,
                 min_dist = MIN_DIST, max_dist = MAX_DIST, plate_thickness = PLATE_THICKNESS):
    img_path = f"{base_path}/pcd{pcd_id}r.png"
    if not os.path.exists(img_path):
        print(f"MISSING FILE: {img_path} — skipping this image.")
        return

    #passthrough when plate_cap ratio is set
    min_dist = enforce_min_dist(min_dist, plate_thickness, warn = True)

    img = load_rgb(base_path, pcd_id)

    backgrounds = load_backgrounds(backgrounds_dir)
    mask = create_mask(img, backgrounds)
    contour = get_largest_contour(mask)

    if contour is None or len(contour) < 10:
        print(f"pcd{pcd_id}: couldn't find a usable contour — skipping.")
        return

    normals = estimate_normals(contour)

    #pass the mask only if mask check is on 
    best_pair, score = find_best_antipodal_pair(contour, normals, min_dist, max_dist,
                                                mask = mask if MASK_CHECK else None)

    #count how many pairs are tied
    n_tied, n_valid = count_near_optimal(contour, normals, min_dist, max_dist)

    if best_pair is None:
        #explain why no pair was found (easier to diagnose)
        n_free, _ = count_near_optimal(contour, normals, min_dist, max_dist)
        print(f"pcd{pcd_id}: no valid antipodal pair found.")
        print(f"  min_dist={min_dist} (clamped from config), max_dist={max_dist}, "
              f"MASK_CHECK={MASK_CHECK}")
        print(f"  ignoring the mask check, {n_free} pairs would be available — so")
        print(f"  {'the mask check is what removed them' if n_free else 'the distance window is what removed them'}.")
        return

    print(f"pcd{pcd_id}: {n_tied} of {n_valid} valid pairs are tied within "
          f"{TIE_TOL} of the top score — tie_break='{TIE_BREAK}' chose among them.")

    if score < 1.0:
        print(f"pcd{pcd_id}: weak antipodal score {score:.2f} (max 2.0) — "
              f"no well-opposed pair of contour points was available.")

    point_a, point_b = best_pair
    x, y, w, h, theta = pair_to_grasp_rectangle(point_a, point_b, plate_thickness)
    predicted_corners = xywh_theta_to_corners(x, y, w, h, theta)

    gt_rects = parse_grasp_rectangles(f"{base_path}/pcd{pcd_id}cpos.txt")

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img)

    #draw ground-truth grasps in green 
    for rect in gt_rects:
        closed = np.vstack([rect, rect[0]])
        ax.plot(closed[:, 0], closed[:, 1], 'g-', linewidth=1.5,
                label="ground truth" if rect is gt_rects[0] else None)

    #draw our prediction in red
    closed_pred = np.vstack([predicted_corners, predicted_corners[0]])
    ax.plot(closed_pred[:, 0], closed_pred[:, 1], 'r-', linewidth=2, label="baseline prediction")

    #draw the two contact points
    ax.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], 'yo', markersize=5)

    ax.set_title(f"pcd{pcd_id} — score {score:.2f}/2.00, w {w:.1f}px, {n_tied} tied")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    from evaluate_baseline import predict_one, analyze_single_prediction

    pcd_id = sys.argv[1] if len(sys.argv) > 1 else "0100"
    folder_num = pcd_id[:2]
    base_path = os.path.join(DATA_ROOT, folder_num)

    run_baseline(base_path, pcd_id, BACKGROUNDS_DIR)

    #print the IoU and angle numbers against every ground-truth rectangle
    backgrounds = load_backgrounds(BACKGROUNDS_DIR)
    pred_corners = predict_one(base_path, pcd_id, backgrounds)
    analyze_single_prediction(base_path, pcd_id, pred_corners)