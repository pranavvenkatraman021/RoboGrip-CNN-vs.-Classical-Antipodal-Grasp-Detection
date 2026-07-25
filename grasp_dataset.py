#imports
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import os
import random

from data_loading import load_rgb, load_depth, parse_grasp_rectangles, convert, crop_to_roi

IMG_SIZE = 224

#rotates rgb, depth, and the rectangle corners by the SAME transform,
#so image and label stay consistent
def rotate_image_and_rect(rgb, depth, rect_corners, angle_deg):
    h, w = rgb.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    rgb_rot = cv2.warpAffine(rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    depth_rot = cv2.warpAffine(depth, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    ones = np.ones((rect_corners.shape[0], 1))
    homogeneous = np.hstack([rect_corners, ones])
    rotated_corners = (M @ homogeneous.T).T

    return rgb_rot, depth_rot, rotated_corners

#randomly jitters brightness/contrast -- RGB only, simulates lighting variation
def jitter_brightness_contrast(rgb, brightness_range=0.2, contrast_range=0.2):
    brightness_factor = 1.0 + random.uniform(-brightness_range, brightness_range)
    contrast_factor = 1.0 + random.uniform(-contrast_range, contrast_range)

    rgb = rgb.astype(np.float32)
    mean = rgb.mean()
    rgb = (rgb - mean) * contrast_factor + mean
    rgb = rgb * brightness_factor
    return np.clip(rgb, 0, 255).astype(np.uint8)

#loads one (image, grasp label) pair 
class GraspDataset(Dataset): 
    def __init__(self, entries, augment=False):
        self.augment = augment

        valid_entries = []
        skipped = 0

        for entry in entries:
            pcd_id = entry["id"]
            folder = entry["folder"]

            rgb_path = f"{folder}/pcd{pcd_id}r.png"
            depth_path = f"{folder}/pcd{pcd_id}d.tiff"

            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                skipped += 1
                continue

            rgb_check = cv2.imread(rgb_path)
            depth_check = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if rgb_check is None or depth_check is None:
                skipped += 1
                continue

            valid_entries.append(entry)

        print(f"GraspDataset: {len(valid_entries)} valid, {skipped} skipped (unreadable files), augment={augment}")
        self.entries = valid_entries

    def __len__(self): 
        return len(self.entries)

    def __getitem__(self, idx): 
        entry = self.entries[idx]
        pcd_id = entry["id"]
        folder = entry["folder"]

        rgb = load_rgb(folder, pcd_id)
        depth = load_depth(folder, pcd_id)

        gt_rects = parse_grasp_rectangles(f"{folder}/pcd{pcd_id}cpos.txt")
        chosen_rect = gt_rects[0].copy()

        #crop to the central ROI FIRST, before anything else.
        rgb, x_off, y_off = crop_to_roi(rgb)
        depth, _, _ = crop_to_roi(depth)

        chosen_rect[:, 0] -= x_off
        chosen_rect[:, 1] -= y_off

        crop_h, crop_w = rgb.shape[:2]

        #random rotation, applied to image AND label together
        if self.augment and random.random() < 0.5:
            angle = random.uniform(-15, 15)
            rgb, depth, chosen_rect = rotate_image_and_rect(rgb, depth, chosen_rect, angle)

        #brightness/contrast jitter -- RGB only, doesn't touch the label
        if self.augment and random.random() < 0.5:
            rgb = jitter_brightness_contrast(rgb)

        #flip -- FIXED: this block now appears only ONCE (a duplicate
        #copy of this same block was removed here)
        if self.augment and random.random() < 0.5:
            rgb = np.fliplr(rgb).copy()
            depth = np.fliplr(depth).copy()
            chosen_rect = chosen_rect.copy()
            chosen_rect[:, 0] = crop_w - chosen_rect[:, 0]

        #resize the CROPPED rgb to the fixed size resnet expects 
        rgb_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        rgb_resized = rgb_resized.astype(np.float32) / 255.0

        #resize the CROPPED depth, normalize 0-1
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0)
        depth_resized = cv2.resize(depth, (IMG_SIZE, IMG_SIZE))
        max_depth = depth_resized.max() if depth_resized.max() > 0 else 1.0
        depth_resized = depth_resized / max_depth

        depth_channel = depth_resized[:, :, np.newaxis]
        combined = np.concatenate([rgb_resized, depth_channel], axis=2)
        combined = combined.transpose(2, 0, 1)
        image_tensor = torch.tensor(combined, dtype=torch.float32)

        x, y, w, h, theta_deg = convert(chosen_rect)

        scale_x = IMG_SIZE / crop_w
        scale_y = IMG_SIZE / crop_h
        x = x * scale_x
        y = y * scale_y
        w = w * scale_x
        h = h * scale_y

        x_norm = x / IMG_SIZE
        y_norm = y / IMG_SIZE
        w_norm = w / IMG_SIZE
        h_norm = h / IMG_SIZE

        theta_rad = np.radians(theta_deg)
        sin2t = np.sin(2 * theta_rad)
        cos2t = np.cos(2 * theta_rad)

        target = torch.tensor([x_norm, y_norm, w_norm, h_norm, sin2t, cos2t], dtype=torch.float32)

        return image_tensor, target

if __name__ == "__main__":
    with open("dataset_split.json", "r") as f:
        split = json.load(f)

    dataset = GraspDataset(split["train"], augment=True)
    print(f"Training set size: {len(dataset)}")

    image, target = dataset[0]
    print("Image tensor shape:", image.shape)   
    print("Target:", target)