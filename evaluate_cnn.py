#imports 
import json
import numpy as np
import cv2 
import torch

from model import GraspNet
from data_loading import load_rgb, load_depth, parse_grasp_rectangles
from baseline import xywh_theta_to_corners
from evaluate_baseline import rectangle_iou, is_correct_grasp

IMG_SIZE = 224

#loads trained model weights
def load_trained_model(weights_path="grasp_model_best.pth"):
    model = GraspNet()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()  
    return model

#preprocesses one image 
def preprocess(rgb, depth):
    orig_h, orig_w = rgb.shape[:2]

    rgb_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    rgb_resized = rgb_resized.astype(np.float32) / 255.0

    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0)
    depth_resized = cv2.resize(depth, (IMG_SIZE, IMG_SIZE))
    max_depth = depth_resized.max() if depth_resized.max() > 0 else 1.0
    depth_resized = depth_resized / max_depth

    depth_channel = depth_resized[:, :, np.newaxis]
    combined = np.concatenate([rgb_resized, depth_channel], axis=2)
    combined = combined.transpose(2, 0, 1)

    tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)  #add batch dimension
    return tensor, orig_w, orig_h

#converts model's output to corners, undoes the normalization and resizing 
def output_to_corners(output, orig_w, orig_h):
    x_norm, y_norm, w_norm, h_norm, sin2t, cos2t = output

    #undo normalization
    x_resized = x_norm * IMG_SIZE
    y_resized = y_norm * IMG_SIZE
    w_resized = w_norm * IMG_SIZE
    h_resized = h_norm * IMG_SIZE

    #undo resize
    scale_x = IMG_SIZE / orig_w
    scale_y = IMG_SIZE / orig_h
    x = x_resized / scale_x
    y = y_resized / scale_y
    w = w_resized / scale_x
    h = h_resized / scale_y

    #get theta
    theta = np.degrees(0.5 * np.arctan2(sin2t, cos2t))

    return xywh_theta_to_corners(x, y, w, h, theta)

#runs CNN across test split and computes accuracy
def evaluate_cnn(split_path="dataset_split.json", weights_path="grasp_model_best.pth"):
    model = load_trained_model(weights_path)

    with open(split_path, "r") as f:
        split = json.load(f)

    test_ids = split["test"]

    total = 0
    correct = 0
    skipped = 0

    for entry in test_ids:
        pcd_id = entry["id"]
        folder = entry["folder"]

        try:
            rgb = load_rgb(folder, pcd_id)
            depth = load_depth(folder, pcd_id)
        except Exception:
            skipped += 1
            continue

        if rgb is None or depth is None:
            skipped += 1
            continue

        gt_rects = parse_grasp_rectangles(f"{folder}/pcd{pcd_id}cpos.txt")
        if len(gt_rects) == 0:
            skipped += 1
            continue

        tensor, orig_w, orig_h = preprocess(rgb, depth)

        with torch.no_grad():
            output = model(tensor).squeeze(0).numpy()

        pred_corners = output_to_corners(output, orig_w, orig_h)

        total += 1
        matched = any(is_correct_grasp(pred_corners, gt) for gt in gt_rects)
        if matched:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    print(f"Total test images attempted: {total}")
    print(f"Skipped: {skipped}")
    print(f"Correct: {correct}")
    print(f"CNN accuracy: {accuracy:.2%}")

    return accuracy

if __name__ == "__main__":
    evaluate_cnn()