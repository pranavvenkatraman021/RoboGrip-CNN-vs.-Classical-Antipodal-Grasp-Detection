#imports 
import json
import numpy as np
import torch


from model import GraspNet
from data_loading import (
    load_rgb, load_depth, parse_grasp_rectangles, crop_to_roi,
    letterbox_to_square
)
from baseline import xywh_theta_to_corners
from evaluate_baseline import rectangle_iou, is_correct_grasp


IMG_SIZE = 224




#loads trained model weights
def load_trained_model(weights_path="grasp_model_best.pth"):
    model = GraspNet()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()  
    return model


#preprocesses one image with square letterboxing
def preprocess(rgb, depth):
    rgb_crop, x_off, y_off = crop_to_roi(rgb)
    depth_crop, _, _ = crop_to_roi(depth)


    rgb_resized, scale, pad_x, pad_y = letterbox_to_square(rgb_crop, IMG_SIZE)
    rgb_resized = rgb_resized.astype(np.float32) / 255.0


    depth_crop = np.nan_to_num(depth_crop.astype(np.float32), nan=0.0)
    depth_resized, depth_scale, depth_pad_x, depth_pad_y = letterbox_to_square(
        depth_crop, IMG_SIZE
    )
    if (
        depth_scale != scale
        or depth_pad_x != pad_x
        or depth_pad_y != pad_y
    ):
        raise ValueError("RGB and depth letterbox transforms do not match")
    max_depth = depth_resized.max() if depth_resized.max() > 0 else 1.0
    depth_resized = depth_resized / max_depth


    depth_channel = depth_resized[:, :, np.newaxis]
    combined = np.concatenate([rgb_resized, depth_channel], axis=2)
    combined = combined.transpose(2, 0, 1)


    tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)
    return tensor, scale, pad_x, pad_y, x_off, y_off


#converts model's output back to real image-coordinate corners
def output_to_corners(output, scale, pad_x, pad_y, x_off, y_off):
    x_norm, y_norm, w_norm, h_norm, sin2t, cos2t = output


    x_square = x_norm * IMG_SIZE
    y_square = y_norm * IMG_SIZE
    w_square = w_norm * IMG_SIZE
    h_square = h_norm * IMG_SIZE


    x = ((x_square - pad_x) / scale) + x_off
    y = ((y_square - pad_y) / scale) + y_off
    w = w_square / scale
    h = h_square / scale


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


        tensor, scale, pad_x, pad_y, x_off, y_off = preprocess(rgb, depth)


        with torch.no_grad():
            output = model(tensor).squeeze(0).numpy()


        pred_corners = output_to_corners(
            output, scale, pad_x, pad_y, x_off, y_off
        )


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