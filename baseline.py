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
#
#FIX (finding B1): perpendicular-to-tangent gives *a* normal, but not
#necessarily an OUTWARD one -- the sign depended entirely on whichever winding
#direction cv2.findContours happened to return. The antipodal score below is
#only meaningful if every normal points outward, so we now force that
#explicitly by checking each normal against the contour centroid. The
#convention can no longer silently invert.
def estimate_normals(contour, step=8):
    n_points = len(contour)
    centroid = contour.astype(float).mean(axis=0)
    normals = []

    for i in range(n_points):
        p_before = contour[(i - step) % n_points].astype(float)
        p_after = contour[(i + step) % n_points].astype(float)

        tangent = p_after - p_before
        tangent_len = np.linalg.norm(tangent)

        if tangent_len == 0:
            normals.append(np.array([0.0, -1.0]))
            continue

        tangent = tangent / tangent_len
        normal = np.array([-tangent[1], tangent[0]])

        #force OUTWARD: the normal must point away from the object's centre
        if np.dot(normal, contour[i].astype(float) - centroid) < 0:
            normal = -normal

        normals.append(normal)

    return normals

#searches pairs of contour points to find best antipodal pair
#
#FIX (finding B1): the score had its sign inverted. With OUTWARD normals and a
#genuine antipodal pair, direction_ab points from A into the object and out the
#far side -- so normal_a opposes it (dot ~ -1) and normal_b aligns with it
#(dot(normal_b, -direction_ab) ~ -1). The old score summed those to -2 and then
#MAXIMISED, i.e. it searched for the least antipodal pair available. Negating
#both terms makes a perfect antipodal pair score +2.0, which is what we want to
#maximise. Verified on synthetic shapes: the corrected search picks the minor
#axis of an ellipse and the short side of a rectangle, while the old one
#collapsed to pairs of nearby points roughly min_dist apart on the same edge.
def find_best_antipodal_pair(contour, normals, min_dist, max_dist, sample_step=4,
                             tie_break=TIE_BREAK, tie_tol=TIE_TOL, mask=None):
    """
    FIX (finding B4): once B1 made the score correct, near-perfect scores became
    common -- on an elongated object every pair of points on the two parallel
    edges scores ~2.0. The old `if score > best_score` then kept whichever one
    contour traversal happened to reach first, so the grasp's position ALONG the
    object was arbitrary. That produces the exact failure signature we measured:
    the angle is right most of the time, but IoU fails because the rectangle is
    in the wrong place.

    So: collect every pair within tie_tol of the best score, then choose among
    them on a physical criterion.

        tie_break="first"     old behaviour (contour order) -- for comparison
        tie_break="centroid"  midpoint nearest the object centroid, a rough
                              stand-in for centre of mass, which is roughly
                              where a person places a grasp
        tie_break="narrow"    smallest opening -- grippers close more reliably
                              on a narrow feature
        tie_break="wide"      largest opening

    If `mask` is supplied, pairs whose midpoint falls outside the object are
    rejected. That matters on concave shapes: the chord between two contour
    points of a mug can cross the hole, which is not a grasp at all.
    """
    n_points = len(contour)
    centroid = contour.astype(float).mean(axis=0)

    candidates = []
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

            if mask is not None:
                mid = ((point_a + point_b) / 2).astype(int)
                if not (0 <= mid[1] < mask.shape[0] and 0 <= mid[0] < mask.shape[1]):
                    continue
                if mask[mid[1], mid[0]] == 0:
                    continue

            direction_ab = (point_b - point_a) / dist

            #antipodal: each OUTWARD normal opposes the closing direction.
            #a perfect pair scores +2.0
            score_a = -np.dot(normals[i], direction_ab)
            score_b = -np.dot(normals[j], -direction_ab)
            candidates.append((score_a + score_b, point_a, point_b, dist))

    if not candidates:
        return None, -np.inf

    top = max(c[0] for c in candidates)
    finalists = [c for c in candidates if c[0] >= top - tie_tol]

    if tie_break == "centroid":
        pick = min(finalists,
                   key=lambda c: np.linalg.norm((c[1] + c[2]) / 2 - centroid))
    elif tie_break == "narrow":
        pick = min(finalists, key=lambda c: c[3])
    elif tie_break == "wide":
        pick = max(finalists, key=lambda c: c[3])
    else:                                   # "first" -- original behaviour
        pick = finalists[0]

    return (pick[1], pick[2]), float(pick[0])

#counts how many pairs are effectively tied at the top score. If this is large,
#the tie-break above is doing more work than the antipodal score itself.
def count_near_optimal(contour, normals, min_dist, max_dist, sample_step=4,
                       tie_tol=TIE_TOL):
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
            scores.append(-np.dot(normals[i], u) - np.dot(normals[j], -u))
    if not scores:
        return 0, 0
    scores = np.array(scores)
    return int((scores >= scores.max() - tie_tol).sum()), len(scores)

#converts an antipodal point pair into a grasp rectangle (x, y, w, h, theta)
#
#width_margin (finding: predicted rectangles measured ~15% smaller than ground
#truth in BOTH dimensions). The chord between the two contact points is the
#OBJECT's width there; a gripper's labelled opening is wider than the object it
#closes on, so a margin factor is physically motivated rather than a fudge.
#Set it from the measured ratio of GT width to predicted width, and check the
#result against measure_gt_stats.py rather than just maximising a score.
def pair_to_grasp_rectangle(point_a, point_b, plate_thickness=PLATE_THICKNESS,
                            width_margin=WIDTH_MARGIN,
                            plate_cap_ratio=PLATE_CAP_RATIO):
    center = (point_a + point_b) / 2
    w = np.linalg.norm(point_b - point_a) * width_margin
    h = plate_thickness

    #B2, better version: keep the plate shorter than the opening so convert()'s
    #longer-edge rule always lands on the closing direction. Only binds when the
    #object is genuinely narrower than the plate.
    if plate_cap_ratio is not None:
        h = min(h, w * plate_cap_ratio)

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

#FIX (finding B2): is_correct_grasp runs BOTH the prediction and the ground
#truth back through convert(), which measures theta from whichever edge is
#LONGER. If a predicted gripper opening is narrower than the plate thickness,
#the longer edge is the plate, not the closing direction, and the reported
#theta jumps 90 degrees -- an automatic failure of the 30-degree angle check no
#matter how good the position is. Clamping min_dist to plate_thickness closes
#that window. Doing it here, rather than changing convert() or
#is_correct_grasp, keeps the shared metric code byte-identical for the CNN.
def enforce_min_dist(min_dist, plate_thickness, warn=False,
                     plate_cap_ratio=PLATE_CAP_RATIO):
    #with the plate capped below the opening the flip cannot happen, so there is
    #no reason to forbid narrow grasps -- a skipped image is a guaranteed failure
    if plate_cap_ratio is not None:
        return min_dist
    if min_dist < plate_thickness:
        if warn:
            print(f"[B2] min_dist={min_dist} < plate_thickness={plate_thickness}; "
                  f"clamping min_dist to {plate_thickness} to avoid the 90-degree theta flip.")
        return plate_thickness
    return min_dist

#runs the full baseline pipeline on one object and visualizes the result:
#ground truth (green) vs. the baseline's predicted grasp (red)
def run_baseline(base_path, pcd_id, backgrounds_dir,
                 min_dist=MIN_DIST, max_dist=MAX_DIST, plate_thickness=PLATE_THICKNESS):
    img_path = f"{base_path}/pcd{pcd_id}r.png"
    if not os.path.exists(img_path):
        print(f"MISSING FILE: {img_path} — skipping this image.")
        return

    min_dist = enforce_min_dist(min_dist, plate_thickness, warn=True)

    img = load_rgb(base_path, pcd_id)

    backgrounds = load_backgrounds(backgrounds_dir)
    mask = create_mask(img, backgrounds)
    contour = get_largest_contour(mask)

    if contour is None or len(contour) < 10:
        print(f"pcd{pcd_id}: couldn't find a usable contour — skipping.")
        return

    normals = estimate_normals(contour)
    best_pair, score = find_best_antipodal_pair(contour, normals, min_dist, max_dist,
                                                mask=mask if MASK_CHECK else None)
    n_tied, n_valid = count_near_optimal(contour, normals, min_dist, max_dist)

    if best_pair is None:
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

    for rect in gt_rects:
        closed = np.vstack([rect, rect[0]])
        ax.plot(closed[:, 0], closed[:, 1], 'g-', linewidth=1.5,
                label="ground truth" if rect is gt_rects[0] else None)

    closed_pred = np.vstack([predicted_corners, predicted_corners[0]])
    ax.plot(closed_pred[:, 0], closed_pred[:, 1], 'r-', linewidth=2, label="baseline prediction")

    #draw the two contact points, so it's obvious where the "fingers" landed
    ax.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], 'yo', markersize=5)

    ax.set_title(f"pcd{pcd_id} — score {score:.2f}/2.00, w {w:.1f}px, {n_tied} tied")
    ax.legend()
    plt.show()

#lets you inspect any single image: shows the plot AND prints the
#actual metric result against every ground truth rectangle, plus a
#detailed diagnostic on the closest-matching one
if __name__ == "__main__":
    from evaluate_baseline import predict_one, analyze_single_prediction

    pcd_id = sys.argv[1] if len(sys.argv) > 1 else "0100"
    folder_num = pcd_id[:2]
    base_path = os.path.join(DATA_ROOT, folder_num)

    run_baseline(base_path, pcd_id, BACKGROUNDS_DIR)

    backgrounds = load_backgrounds(BACKGROUNDS_DIR)
    pred_corners = predict_one(base_path, pcd_id, backgrounds)

    analyze_single_prediction(base_path, pcd_id, pred_corners)