#imports 
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import GraspNet
from grasp_dataset import GraspDataset

#loss function: MSE on x, y, w, h + MSE on sin/cos angle terms 
def grasp_loss(pred, target): 
    xywh_loss = nn.functional.mse_loss(pred[:, :4], target[:, :4])
    angle_loss = nn.functional.mse_loss(pred[:, 4:], target[:, 4:])
    return xywh_loss + angle_loss

def train(): 
    #GPU if available 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load train/val split
    with open("dataset_split.json", "r") as f: 
        split = json.load(f)

    train_dataset = GraspDataset(split["train"])
    val_dataset = GraspDataset(split["val"])

    #batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 16, shuffle = False)

    model = GraspNet().to(device)

    #Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    num_epochs = 20
    train_losses = []
    val_losses = []

    #FIX: initialize these ONCE, before the loop, not inside it
    best_val_loss = float("inf")
    patience = 5
    epochs_without_improvement = 0

    for epoch in range(num_epochs): 
        #training phase
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

        #validation phase
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

        print(f"Epoch {epoch+1}/{num_epochs} — train loss: {avg_train_loss:.4f} — val loss: {avg_val_loss:.4f}")

        #FIX: ONE combined check instead of two duplicated ones
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "grasp_model_best.pth")
            print(f"  → New best val loss, saved checkpoint")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    #save final-epoch weights too, for reference/comparison against the best checkpoint
    torch.save(model.state_dict(), "grasp_model_final.pth")
    print("Saved final-epoch model to grasp_model_final.pth")
    print(f"Best val loss achieved: {best_val_loss:.4f} (saved as grasp_model_best.pth)")

    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Training progress")
    plt.savefig("loss_curve.png")
    plt.show()

if __name__ == "__main__":
    train()