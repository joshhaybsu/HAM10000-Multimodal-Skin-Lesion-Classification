import os
import csv
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from models.multimodal_cnn import MultimodalModel
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

# -------------------------------------------------------
# RESUME SUPPORT
# -------------------------------------------------------
RESUME = False   # <-- manually switch to True if resuming training
RESUME_PATH = "checkpoints/multimodal/multimodal_best_XXXX.pth"  # <-- set manually


# -------------------------------------------------------
# Timestamp & directory setup
# -------------------------------------------------------
RUN_ID = time.strftime("%Y%m%d_%H%M%S")

CKPT_DIR = "checkpoints/multimodal"
RES_DIR = "results/multimodal"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

CSV_LOG_PATH = os.path.join(RES_DIR, f"multimodal_results_{RUN_ID}.csv")
BEST_MODEL_PATH = os.path.join(CKPT_DIR, f"multimodal_best_{RUN_ID}.pth")


# -------------------------------------------------------
# Metrics
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
# Training step
# -------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0

    for images, metadata, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE)
        metadata = metadata.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=DEVICE):
            logits = model(images, metadata)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------------------------------------
# Validation step
# -------------------------------------------------------
def validate(model, loader, criterion):
    model.eval()
    losses = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, metadata, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(DEVICE)
            metadata = metadata.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images, metadata)
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
    print(f"\n🔵 MULTIMODAL RUN ID: {RUN_ID}")
    print("Loading dataloaders...")

    train_loader, val_loader = get_dataloaders(
        split_dir=SPLIT_DIR,
        image_dirs=IMAGE_DIRS,
        batch_size=BATCH_SIZE,
        include_metadata=True     # NOTE: enables metadata tensors
    )

    print("Initialising multimodal model...")
    model = MultimodalModel().to(DEVICE)

    # Resume training if enabled
    if RESUME and os.path.exists(RESUME_PATH):
        model.load_state_dict(torch.load(RESUME_PATH, map_location=DEVICE))
        print(f"Resumed training from checkpoint: {RESUME_PATH}")

    # Class imbalance weights
    train_labels = train_loader.dataset.df["label_binary"].values
    counts = torch.tensor([
        (train_labels == 0).sum(),
        (train_labels == 1).sum()
    ], dtype=torch.float32)

    class_weights = 1.0 / counts
    pos_weight = class_weights[1] / class_weights[0]

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    scaler = torch.amp.GradScaler(device=DEVICE)
    best_auc = 0

    # CSV header
    with open(CSV_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss",
                         "accuracy", "precision", "recall", "f1", "auc"])

    print("\nStarting multimodal training...\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_metrics = validate(model, val_loader, criterion)

        # Log results
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

        print(f"Train Loss: {train_loss:.4f} | Val AUC: {val_metrics['auc']:.4f}")

        # Save best checkpoint
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved new best model → {BEST_MODEL_PATH}")

        scheduler.step(val_metrics["auc"])

    print("\nMultimodal training complete!")
    print(f"Results saved to: {CSV_LOG_PATH}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
