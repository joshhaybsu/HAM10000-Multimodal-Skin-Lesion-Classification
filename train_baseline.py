import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import csv

from models.baseline_cnn import BaselineCNN
from utils.dataloaders import get_dataloaders

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 20

IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2"
]

SPLIT_DIR = "data/splits"
SAVE_PATH = "results/baseline_cnn_best.pth"
CSV_LOG_PATH = "results/baseline_training_log.csv"

os.makedirs("results", exist_ok=True)

# -------------------------------------------------------
# Metric function
# -------------------------------------------------------
def compute_metrics(labels, probs):
    labels = np.array(labels)
    probs = np.array(probs).astype(float)
    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs)
    }

# -------------------------------------------------------
# Training loop
# -------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0

    print("\nStarting training batches...")
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=DEVICE):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)

# -------------------------------------------------------
# Validation loop
# -------------------------------------------------------
def validate(model, loader, criterion):
    model.eval()
    losses = []
    all_labels = []
    all_probs = []

    print("\nEvaluating validation set...")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            losses.append(loss.item())

    metrics = compute_metrics(all_labels, all_probs)
    return sum(losses) / len(losses), metrics

# -------------------------------------------------------
# Main routine
# -------------------------------------------------------
def main():
    print("Loading dataloaders...")
    train_loader, val_loader = get_dataloaders(
        split_dir=SPLIT_DIR,
        image_dirs=IMAGE_DIRS,
        batch_size=BATCH_SIZE
    )

    print("Initialising model...")
    model = BaselineCNN().to(DEVICE)

    train_labels = train_loader.dataset.df["label_binary"].values
    class_counts = torch.tensor([
        (train_labels == 0).sum(),
        (train_labels == 1).sum()
    ], dtype=torch.float32)
    class_weights = 1.0 / class_counts
    pos_weight = class_weights[1] / class_weights[0]

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)


    scaler = torch.amp.GradScaler(device=DEVICE)
    best_val_auc = 0

    # Prepare CSV log
    with open(CSV_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "accuracy", "precision", "recall", "f1", "auc"])

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"===== Epoch {epoch}/{EPOCHS} =====")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_metrics = validate(model, val_loader, criterion)

        # Log to CSV
        with open(CSV_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                val_metrics["accuracy"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
                val_metrics["auc"]
            ])

        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_metrics['accuracy']:.4f}")
        print(f"  Val F1:     {val_metrics['f1']:.4f}")
        print(f"  Val AUC:    {val_metrics['auc']:.4f}")

        # Save best model
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved new best model (AUC={best_val_auc:.4f})")

        # LR Scheduler step
        scheduler.step(val_metrics["auc"])
        
        for param_group in optimizer.param_groups:
          print(f"  Current LR: {param_group['lr']:.6f}")

    print("\nTraining complete!")
    print(f"Best model saved to: {SAVE_PATH}")
    print(f"Training log saved to: {CSV_LOG_PATH}")

if __name__ == "__main__":
    main()
