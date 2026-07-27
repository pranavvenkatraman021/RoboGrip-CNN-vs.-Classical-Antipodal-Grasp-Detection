#imports 
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt




from model import GraspNet
from grasp_dataset import GraspDataset
from evaluate_baseline import is_correct_grasp
from evaluate_cnn import preprocess, output_to_corners
from data_loading import load_rgb, load_depth, parse_grasp_rectangles


IMG_SIZE = 224


#pads each image's different number of valid grasps
def grasp_collate_fn(batch):
    images, target_lists = zip(*batch)
    max_targets = max(targets.shape[0] for targets in target_lists)


    padded_targets = torch.zeros(len(batch), max_targets, 6)
    target_mask = torch.zeros(len(batch), max_targets, dtype=torch.bool)


    for i, targets in enumerate(target_lists):
        count = targets.shape[0]
        padded_targets[i, :count] = targets
        target_mask[i, :count] = True


    return torch.stack(images), padded_targets, target_mask


#uses the closest valid grasp instead of only the first annotation
def grasp_loss(pred, targets, target_mask):
    expanded_pred = pred.unsqueeze(1)


    xy_loss = ((expanded_pred[:, :, :2] - targets[:, :, :2]) ** 2).mean(dim=2)
    wh_loss = ((expanded_pred[:, :, 2:4] - targets[:, :, 2:4]) ** 2).mean(dim=2)
    angle_loss = ((expanded_pred[:, :, 4:] - targets[:, :, 4:]) ** 2).mean(dim=2)


    all_losses = xy_loss + 3.0 * wh_loss + angle_loss
    all_losses = all_losses.masked_fill(~target_mask, float("inf"))
    return all_losses.min(dim=1).values.mean()


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


            tensor, scale, pad_x, pad_y, x_off, y_off = preprocess(rgb, depth)
            tensor = tensor.to(device)


            output = model(tensor).squeeze(0).cpu().numpy()
            pred_corners = output_to_corners(
                output, scale, pad_x, pad_y, x_off, y_off
            )


            total += 1
            if any(is_correct_grasp(pred_corners, gt) for gt in gt_rects):
                correct += 1


    model.train()
    return correct / total if total > 0 else 0.0


def train(): 
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    with open("dataset_split.json", "r") as f: 
        split = json.load(f)


    train_dataset = GraspDataset(split["train"], augment=True)
    val_dataset = GraspDataset(split["val"], augment=False)


    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=grasp_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=grasp_collate_fn
    )


    model = GraspNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 40
    train_losses = []
    val_losses = []
    val_accuracies = []


    best_val_accuracy = 0.0
    patience = 1000  #effectively disabled
    epochs_without_improvement = 0


    for epoch in range(num_epochs): 
        model.train()
        running_train_loss = 0.0


        for images, targets, target_mask in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_mask = target_mask.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = grasp_loss(predictions, targets, target_mask)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()


        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)


        model.eval()
        running_val_loss = 0.0
        with torch.no_grad(): 
            for images, targets, target_mask in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                target_mask = target_mask.to(device)
                predictions = model(images)
                loss = grasp_loss(predictions, targets, target_mask)
                running_val_loss += loss.item()
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)


        #checkpoints on real accuracy
        val_accuracy = compute_val_accuracy(model, val_dataset, device)
        val_accuracies.append(val_accuracy)


        print(f"Epoch {epoch+1}/{num_epochs} — train loss: {avg_train_loss:.4f} — val loss: {avg_val_loss:.4f} — val accuracy: {val_accuracy:.2%}")


        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "grasp_model_multigt_best.pth")
            print(f"  → New best val accuracy ({val_accuracy:.2%}), saved multigt checkpoint")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


    torch.save(model.state_dict(), "grasp_model_multigt_final.pth")
    print("Saved final-epoch model to grasp_model_multigt_final.pth")
    print(f"Best val accuracy achieved: {best_val_accuracy:.2%} (saved as grasp_model_multigt_best.pth)")


    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Training loss")
    plt.savefig("multigt_loss_curve.png")
    plt.show()


    plt.figure()
    plt.plot(val_accuracies, label="val accuracy", color="green")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Validation accuracy (real IoU+angle metric)")
    plt.savefig("multigt_accuracy_curve.png")
    plt.show()


if __name__ == "__main__":
    train()