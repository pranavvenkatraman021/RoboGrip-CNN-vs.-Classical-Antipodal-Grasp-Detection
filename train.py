#imports 
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


from model import GraspNet
from grasp_dataset import GraspDataset
from evaluate_baseline import is_correct_grasp
from baseline import xywh_theta_to_corners
from data_loading import load_rgb, load_depth, parse_grasp_rectangles, crop_to_roi

IMG_SIZE = 224

#loss function: MSE on x,y + weighted MSE on w,h + MSE on sin/cos angle terms
#w,h is weighted higher since it's what actually drives IoU/correctness --
#x,y naturally has larger values and was dominating the total loss otherwise
def grasp_loss(pred, target):
    xy_loss = nn.functional.mse_loss(pred[:, :2], target[:, :2])
    wh_loss = nn.functional.mse_loss(pred[:, 2:4], target[:, 2:4])
    angle_loss = nn.functional.mse_loss(pred[:, 4:], target[:, 4:])
    return xy_loss + 3.0 * wh_loss + angle_loss

#computes REAL validation accuracy using the same IoU+angle metric used
#for the baseline and final evaluation -- not a loss proxy, the actual metric.
#Uses the SAME ROI-crop preprocessing as GraspDataset and evaluate_cnn.py,
#so checkpoint selection stays consistent with how the model was trained
#and how it's finally evaluated.
def compute_val_accuracy(model, val_dataset, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for entry in val_dataset.entries:
            pcd_id, folder = entry["id"], entry["folder"]
            rgb = load_rgb(folder, pcd_id)
            depth = load_depth(folder, pcd_id)
            gt_rects = parse_grasp_rectangles(f"{folder}/pcd{pcd_id}cpos.txt")
            if len(gt_rects) == 0:
                continue

            rgb_crop, x_off, y_off = crop_to_roi(rgb)
            depth_crop, _, _ = crop_to_roi(depth)
            crop_h, crop_w = rgb_crop.shape[:2]

            rgb_r = cv2.resize(rgb_crop, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
            d = np.nan_to_num(depth_crop.astype(np.float32), nan=0.0)
            d_r = cv2.resize(d, (IMG_SIZE, IMG_SIZE))
            d_r = d_r / (d_r.max() if d_r.max() > 0 else 1.0)
            combined = np.concatenate([rgb_r, d_r[:, :, np.newaxis]], axis=2).transpose(2, 0, 1)
            tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(device)

            output = model(tensor).squeeze(0).cpu().numpy()
            x_n, y_n, w_n, h_n, s2t, c2t = output

            #map normalized output directly back to crop-local pixels,
            #then shift by the crop offset to get real image coordinates
            x_pred = (x_n * crop_w) + x_off
            y_pred = (y_n * crop_h) + y_off
            w_pred = w_n * crop_w
            h_pred = h_n * crop_h
            theta = np.degrees(0.5 * np.arctan2(s2t, c2t))
            pred_corners = xywh_theta_to_corners(x_pred, y_pred, w_pred, h_pred, theta)

            total += 1
            if any(is_correct_grasp(pred_corners, gt) for gt in gt_rects):
                correct += 1

    model.train()
    return correct / total if total > 0 else 0.0

def train(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("dataset_split.json", "r") as f: 
        split = json.load(f)

    train_dataset = GraspDataset(split["train"], augment=True)
    val_dataset = GraspDataset(split["val"], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = GraspNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 40
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_accuracy = 0.0
    patience = 1000  # effectively disabled -- confirmed this was cutting off w/h convergence too early
    epochs_without_improvement = 0

    for epoch in range(num_epochs): 
        model.train()
        running_train_loss = 0.0

        for images, targets in train_loader: 
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = grasp_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad(): 
            for images, targets in val_loader: 
                images, targets = images.to(device), targets.to(device)
                predictions = model(images)
                loss = grasp_loss(predictions, targets)
                running_val_loss += loss.item()
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        #NEW: checkpoint on REAL accuracy, not loss
        val_accuracy = compute_val_accuracy(model, val_dataset, device)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs} — train loss: {avg_train_loss:.4f} — val loss: {avg_val_loss:.4f} — val accuracy: {val_accuracy:.2%}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "grasp_model_best.pth")
            print(f"  → New best val accuracy ({val_accuracy:.2%}), saved checkpoint")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    torch.save(model.state_dict(), "grasp_model_final.pth")
    print("Saved final-epoch model to grasp_model_final.pth")
    print(f"Best val accuracy achieved: {best_val_accuracy:.2%} (saved as grasp_model_best.pth)")

    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Training loss")
    plt.savefig("loss_curve.png")
    plt.show()

    plt.figure()
    plt.plot(val_accuracies, label="val accuracy", color="green")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Validation accuracy (real IoU+angle metric)")
    plt.savefig("accuracy_curve.png")
    plt.show()

if __name__ == "__main__":
    train()