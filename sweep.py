#sweep.py
import json
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import GraspNet
from grasp_dataset import GraspDataset
from train import compute_val_accuracy  # reuse the real accuracy metric

def make_loss_fn(wh_weight):
    def loss_fn(pred, target):
        xy = nn.functional.mse_loss(pred[:, :2], target[:, :2])
        wh = nn.functional.mse_loss(pred[:, 2:4], target[:, 2:4])
        angle = nn.functional.mse_loss(pred[:, 4:], target[:, 4:])
        return xy + wh_weight * wh + angle
    return loss_fn

def run_trial(wh_weight, lr, train_dataset, val_dataset, device, epochs=15):
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = GraspNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = make_loss_fn(wh_weight)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), targets)
            loss.backward()
            optimizer.step()

        acc = compute_val_accuracy(model, val_dataset, device)
        if acc > best_acc:
            best_acc = acc

    return best_acc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("dataset_split.json") as f:
        split = json.load(f)

    train_dataset = GraspDataset(split["train"], augment=True)
    val_dataset = GraspDataset(split["val"], augment=False)

    wh_weights = [2.0, 3.0, 4.0, 5.0]
    learning_rates = [5e-5, 1e-4, 2e-4]

    results = []
    for wh_w, lr in itertools.product(wh_weights, learning_rates):
        print(f"\n--- Trying wh_weight={wh_w}, lr={lr} ---")
        best_acc = run_trial(wh_w, lr, train_dataset, val_dataset, device, epochs=15)
        print(f"Best val accuracy (15 epochs): {best_acc:.2%}")
        results.append((wh_w, lr, best_acc))

    print("\n==== Summary ====")
    for wh_w, lr, acc in sorted(results, key=lambda r: -r[2]):
        print(f"wh_weight={wh_w}, lr={lr} -> {acc:.2%}")