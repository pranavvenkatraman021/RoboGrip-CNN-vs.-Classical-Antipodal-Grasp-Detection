#imports
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from data_loading import load_rgb, load_depth, parse_grasp_rectangles
from baseline import load_backgrounds
from evaluate_baseline import predict_one, is_correct_grasp, rectangle_iou
from evaluate_cnn import load_trained_model, preprocess, output_to_corners
from config import (BACKGROUNDS_DIR, MIN_DIST, MAX_DIST, PLATE_THICKNESS,
                    WIDTH_MARGIN, TIE_BREAK, TIE_TOL, MASK_CHECK)

CNN_WEIGHTS    = "grasp_model_multigt_best.pth"
SPLIT_KEY      = "test"
TARGET_FOLDERS = ["04", "06"]   #scissors/pliers and staplers


def draw_rect(ax, corners, color, lw=2, label=None):
    closed = np.vstack([corners, corners[0]])
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=lw, label=label)


def find_best_image(entries, backgrounds, model, device):
    """
    find an image where:
      - the baseline prediction lands ON the object (IoU > 0 with any GT)
        but is still wrong — this avoids the floating-box case
      - the CNN is correct
      - the gap between CNN IoU and baseline IoU is large
    """
    candidates = []

    for entry in entries:
        if entry["id"][:2] not in TARGET_FOLDERS:
            continue

        pcd_id, folder = entry["id"], entry["folder"]
        gt_rects = parse_grasp_rectangles(f"{folder}/pcd{pcd_id}cpos.txt")
        if not gt_rects:
            continue

        base_pred = predict_one(
            folder, pcd_id, backgrounds,
            min_dist=MIN_DIST, max_dist=MAX_DIST,
            plate_thickness=PLATE_THICKNESS, width_margin=WIDTH_MARGIN,
            tie_break=TIE_BREAK, tie_tol=TIE_TOL, mask_check=MASK_CHECK,
        )

        #baseline must have a prediction that overlaps the object at least a little
        #but still be incorrect -- avoids the floating-box embarrassment
        if base_pred is None:
            continue
        best_base_iou = max(rectangle_iou(base_pred, gt) for gt in gt_rects)
        baseline_correct = any(is_correct_grasp(base_pred, gt) for gt in gt_rects)
        if baseline_correct or best_base_iou < 0.05:
            continue   #skip: either baseline passes, or box is nowhere near the object

        try:
            rgb   = load_rgb(folder, pcd_id)
            depth = load_depth(folder, pcd_id)
        except Exception:
            continue
        if rgb is None or depth is None:
            continue

        tensor, scale, pad_x, pad_y, x_off, y_off = preprocess(rgb, depth)
        with torch.no_grad():
            output = model(tensor.to(device)).squeeze(0).cpu().numpy()
        cnn_pred = output_to_corners(output, scale, pad_x, pad_y, x_off, y_off)

        cnn_correct  = any(is_correct_grasp(cnn_pred, gt) for gt in gt_rects)
        best_cnn_iou = max(rectangle_iou(cnn_pred, gt) for gt in gt_rects)

        if not cnn_correct:
            continue

        gap = best_cnn_iou - best_base_iou
        candidates.append((gap, pcd_id, folder, rgb, gt_rects,
                           base_pred, cnn_pred, best_base_iou, best_cnn_iou))

    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[0])
    print(f"Found {len(candidates)} qualifying images. Top 5:")
    for gap, pid, *_, b_iou, c_iou in candidates[:5]:
        print(f"  pcd{pid}: CNN IoU {c_iou:.2f}  baseline IoU {b_iou:.2f}  gap {gap:.2f}")
    return candidates[0]


def make_figure(pcd_id, rgb, gt_rects, base_pred, cnn_pred,
                base_iou, cnn_iou, output_path="visual_evidence.png"):
    img_h, img_w = rgb.shape[:2]
    aspect       = img_w / img_h

    #three equal panels, tight layout, no custom fonts
    fig, axes = plt.subplots(1, 3, figsize=(14, 14 / (3 * aspect) + 1.4))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.13,
                        wspace=0.04)

    best_gt = max(gt_rects, key=lambda g: rectangle_iou(cnn_pred, g))

    panels = [
        ("Ground Truth",        gt_rects,  None,      None,          None),
        ("Baseline Prediction", [best_gt], base_pred, base_iou,      False),
        ("CNN Prediction",      [best_gt], cnn_pred,  cnn_iou,       True),
    ]

    for ax, (title, gts, pred, iou, correct) in zip(axes, panels):
        ax.imshow(rgb)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")

        for i, gt in enumerate(gts):
            closed = np.vstack([gt, gt[0]])
            ax.plot(closed[:, 0], closed[:, 1], color="green", linewidth=2,
                    label="Ground truth" if i == 0 else None)

        if pred is not None:
            closed = np.vstack([pred, pred[0]])
            ax.plot(closed[:, 0], closed[:, 1], color="red", linewidth=2.2,
                    label="Prediction")

        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)

        if pred is not None:
            ax.legend(loc="upper left", fontsize=8, frameon=True,
                      framealpha=0.8, edgecolor="none", handlelength=1.2)

    #metric labels below each panel -- placed in figure space so they never overlap the image
    for i, (_, _, pred, iou, correct) in enumerate(panels):
        if iou is None:
            continue
        label  = f"IoU = {iou:.2f}   {'✓ CORRECT' if correct else '✗ INCORRECT'}"
        color  = "green" if correct else "red"
        #evenly spaced x positions matching the three axes
        x_pos = (i + 0.5) / 3
        fig.text(x_pos, 0.07, label,
                 ha="center", va="center", fontsize=11,
                 fontweight="bold", color=color)

    #pcd id
    fig.text(0.5, 0.03, f"pcd{pcd_id}",
             ha="center", va="center", fontsize=8.5, color="#666666")

    #caption
    fig.text(0.5, 0.005,
             "Baseline selects the wrong contact axis — no clean antipodal pair on "
             "asymmetric objects.  CNN learned from examples.",
             ha="center", va="bottom", fontsize=9, color="#333333", style="italic")

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    with open("dataset_split.json") as f:
        split = json.load(f)

    backgrounds = load_backgrounds(BACKGROUNDS_DIR)

    if not os.path.exists(CNN_WEIGHTS):
        print(f"ERROR: {CNN_WEIGHTS} not found.")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_trained_model(CNN_WEIGHTS).to(device)
    model.eval()

    print("Scanning test images...")
    result = find_best_image(split[SPLIT_KEY], backgrounds, model, device)

    if result is None:
        print("No qualifying image found.")
        exit(1)

    gap, pcd_id, folder, rgb, gt_rects, base_pred, cnn_pred, base_iou, cnn_iou = result
    print(f"\nUsing pcd{pcd_id}")
    make_figure(pcd_id, rgb, gt_rects, base_pred, cnn_pred,
                base_iou, cnn_iou, output_path="visual_evidence.png")