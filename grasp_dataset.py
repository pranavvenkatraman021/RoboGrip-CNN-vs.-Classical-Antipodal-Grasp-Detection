#imports
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import os
import random

from data_loading import load_rgb, load_depth, parse_grasp_rectangles, convert

IMG_SIZE = 224

#loads one (image, grasp label) pair 
class GraspDataset(Dataset): 
    #CHANGED: added augment parameter, defaults to False so nothing
    #breaks for existing calls that don't pass it
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

        orig_h, orig_w = rgb.shape[:2]

        #load ground truth grasp rectangles and pick the first valid one
        gt_rects = parse_grasp_rectangles(f"{folder}/pcd{pcd_id}cpos.txt")
        chosen_rect = gt_rects[0]

        #NEW: augmentation happens here — BEFORE resizing, while everything
        #is still in original pixel coordinates, and only for training data.
        #Flipping rgb/depth and the rectangle's x-coordinates together
        #keeps the image and label consistent with each other.
        if self.augment and random.random() < 0.5:
            rgb = np.fliplr(rgb).copy()
            depth = np.fliplr(depth).copy()
            chosen_rect = chosen_rect.copy()
            chosen_rect[:, 0] = orig_w - chosen_rect[:, 0]

        #resize RGB to fixed size resnet expects 
        rgb_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        rgb_resized = rgb_resized.astype(np.float32) / 255.0

        #resize depth values and normalize from 0-1
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0)
        depth_resized = cv2.resize(depth, (IMG_SIZE, IMG_SIZE))
        max_depth = depth_resized.max() if depth_resized.max() > 0 else 1.0
        depth_resized = depth_resized / max_depth

        #stack RGB channels and depth channel into 4 channel image 
        depth_channel = depth_resized[:, :, np.newaxis]
        combined = np.concatenate([rgb_resized, depth_channel], axis=2)
        combined = combined.transpose(2, 0, 1)
        image_tensor = torch.tensor(combined, dtype=torch.float32)

        #CHANGED: convert() now runs on chosen_rect AFTER the possible flip above
        x, y, w, h, theta_deg = convert(chosen_rect)

        scale_x = IMG_SIZE / orig_w
        scale_y = IMG_SIZE / orig_h
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