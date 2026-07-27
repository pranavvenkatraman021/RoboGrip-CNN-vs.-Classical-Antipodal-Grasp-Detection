#imports
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import os
import random


from data_loading import (
    load_rgb, load_depth, parse_grasp_rectangles, convert, crop_to_roi,
    letterbox_to_square, points_to_letterbox
)


IMG_SIZE = 224


#rotates rgb, depth, and the rectangle corners by the SAME transform,
#so image and label stay consistent
def rotate_image_and_rects(rgb, depth, rects, angle_deg):
    h, w = rgb.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)


    rgb_rot = cv2.warpAffine(rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    depth_rot = cv2.warpAffine(depth, M, (w, h), borderMode=cv2.BORDER_REFLECT)


    flat_points = rects.reshape(-1, 2)
    ones = np.ones((flat_points.shape[0], 1))
    homogeneous = np.hstack([flat_points, ones])
    rotated_points = (M @ homogeneous.T).T
    rotated_rects = rotated_points.reshape(rects.shape)


    return rgb_rot, depth_rot, rotated_rects


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
        all_rects = np.stack(gt_rects).astype(np.float32)


        #crop to the central ROI FIRST, before anything else.
        rgb, x_off, y_off = crop_to_roi(rgb)
        depth, _, _ = crop_to_roi(depth)


        all_rects[:, :, 0] -= x_off
        all_rects[:, :, 1] -= y_off


        crop_h, crop_w = rgb.shape[:2]


        #random rotation, applied to image AND label together
        if self.augment and random.random() < 0.5:
            angle = random.uniform(-15, 15)
            rgb, depth, all_rects = rotate_image_and_rects(
                rgb, depth, all_rects, angle
            )


        #brightness/contrast jitter -- RGB only, doesn't touch the label
        if self.augment and random.random() < 0.5:
            rgb = jitter_brightness_contrast(rgb)


        #flip -- FIXED: this block now appears only ONCE (a duplicate
        #copy of this same block was removed here)
        if self.augment and random.random() < 0.5:
            rgb = np.fliplr(rgb).copy()
            depth = np.fliplr(depth).copy()
            all_rects = all_rects.copy()
            all_rects[:, :, 0] = crop_w - all_rects[:, :, 0]


        #resize without changing the visible grasp angle
        rgb_resized, scale, pad_x, pad_y = letterbox_to_square(rgb, IMG_SIZE)
        rgb_resized = rgb_resized.astype(np.float32) / 255.0


        #resize depth with the same scale and padding
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0)
        depth_resized, depth_scale, depth_pad_x, depth_pad_y = letterbox_to_square(
            depth, IMG_SIZE
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
        image_tensor = torch.tensor(combined, dtype=torch.float32)


        #moves every label through the same letterbox transform
        targets = []
        for rect in all_rects:
            rect = points_to_letterbox(rect, scale, pad_x, pad_y)
            x, y, w, h, theta_deg = convert(rect)


            theta_rad = np.radians(theta_deg)
            targets.append([
                x / IMG_SIZE,
                y / IMG_SIZE,
                w / IMG_SIZE,
                h / IMG_SIZE,
                np.sin(2 * theta_rad),
                np.cos(2 * theta_rad)
            ])


        target_tensor = torch.tensor(targets, dtype=torch.float32)
        return image_tensor, target_tensor


if __name__ == "__main__":
    with open("dataset_split.json", "r") as f:
        split = json.load(f)


    dataset = GraspDataset(split["train"], augment=True)
    print(f"Training set size: {len(dataset)}")


    image, targets = dataset[0]
    print("Image tensor shape:", image.shape)   
    print("Target tensor shape:", targets.shape)