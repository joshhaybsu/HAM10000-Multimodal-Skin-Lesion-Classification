import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from tqdm import tqdm
import csv

from models.multimodal_cnn import MultimodalCNN
from utils.dataloaders import get_dataloaders


# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 30

IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2",
]

SPLIT_DIR = "data/splits"

# Resume feature
RESUME = False  # <-- set to True if you want to resume training
RESUME_PATH = "checkpoints/multimodal/multimodal_best_XXXXX.pth" # <-- edit manually when needed

# Run ID & Directory Setup
RUN_ID = time.strftime("%Y%m%d_%H%M%S")

CKPT_DIR = "checkpoints/multimodal"
RES_DIR = "results/multimodal"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

CSV_LOG_PATH = os.path.join(RES_DIR, f"multimodal_results_{RUN_ID}.csv")
BEST_MODEL_PATH = os.path.join(CKPT_DIR, f"multimodal_best_{RUN_ID}.pth")

# Metrics 
def compute_metrics(labels, probs):
    labels = np.array(labels)
    probs = np.array(probs).astype(float)
    preds = (probs >= 0.5).astype(int)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = None

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": auc,
    }

# Training loop
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0

    for images, metadata, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        age = metadata["age"].to(DEVICE)
        sex = metadata["sex"].to(DEVICE)
        loc = metadata["loc"].to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=DEVICE):
            logits = model(images, age, sex, loc)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)

# Validation loop
def validate(model, loader, criterion):
    model.eval()
    losses = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, metadata, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            age = metadata["age"].to(DEVICE)
            sex = metadata["sex"].to(DEVICE)
            loc = metadata["loc"].to(DEVICE)

            logits = model(images, age, sex, loc)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            losses.append(loss.item())

    metrics = compute_metrics(all_labels, all_probs)
    return sum(losses) / len(losses), metrics

# Main
def main():
    print(f"\nRUN ID: {RUN_ID}")
    print("Loading dataloaders...")

    train_loader, val_loader, test_loader = get_dataloaders(
        split_dir=SPLIT_DIR,
        image_dirs=IMAGE_DIRS,
        batch_size=BATCH_SIZE,
    )

    # Get vocab sizes from the dataset
    train_ds = train_loader.dataset
    num_sex_categories = len(train_ds.sex_vocab)
    num_loc_categories = len(train_ds.loc_vocab)

    print(f"Sex categories: {num_sex_categories}")
    print(f"Location categories: {num_loc_categories}")

    print("Initialising multimodal model...")
    model = MultimodalCNN(
        num_sex_categories=num_sex_categories,
        num_loc_categories=num_loc_categories,
    ).to(DEVICE)

    # Optional resume
    if RESUME and os.path.exists(RESUME_PATH):
        model.load_state_dict(torch.load(RESUME_PATH, map_location=DEVICE))
        print(f"Resumed training from checkpoint: {RESUME_PATH}")

    # Class imbalance handling
    train_labels = train_ds.df["label_binary"].values
    class_counts = torch.tensor(
        [
            (train_labels == 0).sum(),
            (train_labels == 1).sum(),
        ],
        dtype=torch.float32,
    )
    class_weights = 1.0 / class_counts
    pos_weight = class_weights[1] / class_weights[0]

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    scaler = torch.amp.GradScaler(device=DEVICE)
    best_val_auc = 0.0

    # CSV header
    with open(CSV_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "accuracy", "precision", "recall", "f1", "auc"]
        )

    print("\nStarting multimodal training...\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_metrics = validate(model, val_loader, criterion)

        auc = val_metrics["auc"]
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"

        print(f"Train Loss: {train_loss:.4f}\n"
          f"Val Loss:   {val_loss:.4f}\n"
          f"AUC:        {auc_str}\n"
          f"Accuracy:   {val_metrics['accuracy']:.4f}\n"
          f"Precision:  {val_metrics['precision']:.4f}\n"
          f"Recall:     {val_metrics['recall']:.4f}\n"
          f"F1 Score:   {val_metrics['f1']:.4f}")

        # Write results
        with open(CSV_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    val_loss,
                    val_metrics["accuracy"],
                    val_metrics["precision"],
                    val_metrics["recall"],
                    val_metrics["f1"],
                    auc,
                ]
            )

        # Save best model by AUC
        if auc is not None and auc > best_val_auc:
            best_val_auc = auc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved new best multimodal model to: {BEST_MODEL_PATH}")

        scheduler.step(auc if auc is not None else 0.0)

    print("\nTraining complete.")
    print(f"Results saved to: {CSV_LOG_PATH}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")

if __name__ == "__main__":
    main()
