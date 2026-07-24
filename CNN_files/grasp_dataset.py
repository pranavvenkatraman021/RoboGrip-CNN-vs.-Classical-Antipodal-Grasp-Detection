#imports
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from data_handling.data_loading import load_rgb, load_depth, parse_grasp_rectangles, convert

IMG_SIZE = 224

#loads one (image, grasp label) pair 
class GraspDataset(Dataset): 
    def __init__(self, entries): 
        #entries are a list of {"id":, "folder": dicts}
        self.entries = entries

    def __len__(self): 
        return len(self.entries)

    def __getitem__(self, idx): 
        entry = self.entries[idx]
        pcd_id = entry["id"]
        folder = entry["folder"]

        #load RGB and depth
        rgb = load_rgb(folder, pcd_id)
        depth = load_depth(folder, pcd_id)

        orig_h, orig_w = rgb.shape[:2]

        #resize RGB to fixed size resnet expects 
        rgb_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        #scaled pixel values to 0-1
        rgb_resized = rgb_resized.astype(np.float32) / 255.0

        #resize depth values and normalize from 0-1
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0)
        depth_resized = cv2.resize(depth, (IMG_SIZE, IMG_SIZE))
        max_depth = depth_resized.max() if depth_resized.max() > 0 else 1.0
        depth_resized = depth_resized / max_depth

        #stack RGB channels and depth channel into 4 channel image 
        depth_channel = depth_resized[:, :, np.newaxis]
        combined = np.concatenate([rgb_resized, depth_channel], axis=2)

        #pytorch expects channels first 
        combined = combined.transpose(2, 0, 1)
        image_tensor = torch.tensor(combined, dtype=torch.float32)

        #load ground truth grasp rectangles and pick the first valid one 
        #this is a simple starting point 
        gt_rects = parse_grasp_rectangles(f"{folder}/pcd{pcd_id}cpos.txt")
        chosen_rect = gt_rects[0]

        x, y, w, h, theta_deg = convert(chosen_rect)

        #rescale values to match resized image 
        scale_x = IMG_SIZE / orig_w
        scale_y = IMG_SIZE / orig_h
        x = x * scale_x
        y = y * scale_y
        w = w * scale_x
        h = h * scale_y

        #normalize to roughly 0-1 range 
        x_norm = x / IMG_SIZE
        y_norm = y / IMG_SIZE
        w_norm = w / IMG_SIZE
        h_norm = h / IMG_SIZE

        #avoiding degree wraparound 
        theta_rad = np.radians(theta_deg)
        sin2t = np.sin(2 * theta_rad)
        cos2t = np.cos(2 * theta_rad)

        target = torch.tensor([x_norm, y_norm, w_norm, h_norm, sin2t, cos2t], dtype=torch.float32)

        return image_tensor, target

if __name__ == "__main__":
    #test
    with open("dataset_split.json", "r") as f:
        split = json.load(f)

    dataset = GraspDataset(split["train"])
    print(f"Training set size: {len(dataset)}")

    image, target = dataset[0]
    #should be torch.Size([4, 224, 224])
    print("Image tensor shape:", image.shape)   
    print("Target:", target)       