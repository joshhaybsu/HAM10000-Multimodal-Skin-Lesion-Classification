import os
import time
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

from utils.dataloaders import get_dataloaders
from models.baseline_cnn import BaselineCNN
from models.multimodal_cnn import MultimodalCNN

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASELINE_CKPT = "checkpoints/baseline/baseline_best_20251208_161156.pth"
MULTIMODAL_CKPT = "checkpoints/multimodal/multimodal_best_20251208_163850.pth"

IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2"
]
SPLIT_DIR = "data/splits"

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RESULTS_PATH = f"results/test_results_{RUN_ID}.csv"

os.makedirs("results", exist_ok=True)

# Metrics
def compute_metrics(labels, probs):
    preds = (np.array(probs) >= 0.5).astype(int)

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
        "confusion_matrix": confusion_matrix(labels, preds),
    }

# Plot ROC curve
def plot_roc(labels, probs, title, save_path):
    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_predictions(labels, probs)
    plt.title(title)
    plt.savefig(save_path, dpi=300)
    plt.close()

# Confusion matrix heatmap
def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# PR Curve
def plot_pr_curve(labels, probs, title, save_path):
    PrecisionRecallDisplay.from_predictions(labels, probs)
    plt.title(title)
    plt.savefig(save_path, dpi=300)
    plt.close()


# Threshold sweep (Recall & F1 vs threshold)
def plot_threshold_sweep(labels, probs, title, save_path):
    thresholds = np.linspace(0, 1, 200)
    recalls = []
    f1s = []

    labels_arr = np.array(labels)

    for t in thresholds:
        preds = (np.array(probs) >= t).astype(int)
        recalls.append(recall_score(labels_arr, preds, zero_division=0))
        f1s.append(f1_score(labels_arr, preds, zero_division=0))

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, f1s, label="F1-score")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()

# Model evaluation
def evaluate(model, loader, multimodal=False):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if multimodal:
                images, meta_dict, labels = batch
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                age = meta_dict["age"].to(DEVICE)
                sex = meta_dict["sex"].to(DEVICE)
                loc = meta_dict["loc"].to(DEVICE)

                logits = model(images, age, sex, loc)

            else:
                images, _, labels = batch
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model(images)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    return compute_metrics(all_labels, all_probs), all_labels, all_probs


# Main
def main():
    print("Loading test loader...")
    _, _, test_loader = get_dataloaders(
        split_dir=SPLIT_DIR,
        image_dirs=IMAGE_DIRS,
        batch_size=64,
    )

    results = []

    # Baseline
    print("\nLoading BASELINE checkpoint...")
    baseline = BaselineCNN().to(DEVICE)
    baseline.load_state_dict(torch.load(BASELINE_CKPT, map_location=DEVICE))

    print("Evaluating Baseline...")
    base_metrics, base_labels, base_probs = evaluate(baseline, test_loader, multimodal=False)

    plot_roc(base_labels, base_probs, "Baseline ROC Curve", "results/roc_baseline.png")
    plot_confusion_matrix(base_metrics["confusion_matrix"], "Baseline Confusion Matrix", "results/cm_baseline.png")
    plot_pr_curve(base_labels, base_probs, "Baseline PR Curve", "results/pr_baseline.png")
    plot_threshold_sweep(base_labels, base_probs, "Baseline Threshold Sweep", "results/threshold_baseline.png")

    results.append({
        "model": "baseline",
        **base_metrics
    })

    # Multimodal
    print("\nLoading MULTIMODAL checkpoint...")

    # vocab sizes from dataset
    test_ds = test_loader.dataset
    num_sex = len(test_ds.sex_vocab)
    num_loc = len(test_ds.loc_vocab)

    multimodal = MultimodalCNN(num_sex, num_loc).to(DEVICE)
    multimodal.load_state_dict(torch.load(MULTIMODAL_CKPT, map_location=DEVICE))

    print("Evaluating Multimodal...")
    multi_metrics, multi_labels, multi_probs = evaluate(multimodal, test_loader, multimodal=True)

    plot_roc(multi_labels, multi_probs, "Multimodal ROC Curve", "results/roc_multimodal.png")
    plot_confusion_matrix(multi_metrics["confusion_matrix"], "Multimodal Confusion Matrix", "results/cm_multimodal.png")
    plot_pr_curve(multi_labels, multi_probs, "Multimodal PR Curve", "results/pr_multimodal.png")
    plot_threshold_sweep(multi_labels, multi_probs, "Multimodal Threshold Sweep", "results/threshold_multimodal.png")

    results.append({
        "model": "multimodal",
        **multi_metrics
    })

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved test results → {RESULTS_PATH}\n")

    print("\n===== COMPARISON =====")
    print(df)

if __name__ == "__main__":
    main()
