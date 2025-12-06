import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


# =====================================================
# 1. CONFIGURATION
# =====================================================

DATA_ROOT = "data"

METADATA_CSV = os.path.join(DATA_ROOT, "HAM10000_metadata.csv")

IMAGES_DIRS = [
    os.path.join(DATA_ROOT, "HAM10000_images_part_1"),
    os.path.join(DATA_ROOT, "HAM10000_images_part_2"),
]

SPLIT_DIR = os.path.join(DATA_ROOT, "splits")
os.makedirs(SPLIT_DIR, exist_ok=True)

RANDOM_STATE = 42
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15


# =====================================================
# 2. LOAD METADATA
# =====================================================

df = pd.read_csv(METADATA_CSV)
print("Loaded metadata:", df.shape)


# =====================================================
# 3. MAP dx → BINARY LABEL
# =====================================================

malignant_set = {"mel", "bcc", "akiec"}  # malignant

df["label_binary"] = df["dx"].apply(
    lambda x: 1 if x in malignant_set else 0
)

print("\nClass distribution:")
print(df["label_binary"].value_counts())


# =====================================================
# 4. BUILD IMAGE LOOKUP
# =====================================================

def build_lookup(image_dirs):
    lookup = {}
    for d in image_dirs:
        for path in glob.glob(os.path.join(d, "*.jpg")):
            stem = os.path.splitext(os.path.basename(path))[0]
            lookup[stem] = path
    return lookup

image_lookup = build_lookup(IMAGES_DIRS)
print("\nImages found:", len(image_lookup))

df["image_path"] = df["image_id"].map(image_lookup)
missing = df["image_path"].isna().sum()
if missing > 0:
    print(f"WARNING: {missing} missing images")
    df = df.dropna(subset=["image_path"])


# =====================================================
# 5. CLEAN METADATA (NO ONE-HOT ENCODING)
# =====================================================

df["age"] = df["age"].fillna(df["age"].median())
df["sex"] = df["sex"].fillna("unknown")
df["localization"] = df["localization"].fillna("unknown")


# =====================================================
# 6. STRATIFIED LESION-LEVEL SPLITTING
# =====================================================

groups = df["lesion_id"]
y = df["label_binary"]

gss = GroupShuffleSplit(n_splits=1, train_size=TRAIN_FRACTION, random_state=RANDOM_STATE)
train_idx, temp_idx = next(gss.split(df, y, groups))

train_df = df.iloc[train_idx].copy()
temp_df  = df.iloc[temp_idx].copy()

val_size = VAL_FRACTION / (1 - TRAIN_FRACTION)

val_idx, test_idx = train_test_split(
    temp_df.index,
    test_size=1 - val_size,
    stratify=temp_df["label_binary"],
    random_state=RANDOM_STATE
)

val_df = temp_df.loc[val_idx].copy()
test_df = temp_df.loc[test_idx].copy()

print("\nSplit sizes:")
print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)


# =====================================================
# 7. SAVE SPLITS
# =====================================================

train_df.to_csv(os.path.join(SPLIT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(SPLIT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(SPLIT_DIR, "test.csv"), index=False)

print("\nSaved split CSVs.\n")


# =====================================================
# 8. COMPUTE CLASS WEIGHTS FOR IMBALANCE HANDLING
# =====================================================

train_counts = train_df["label_binary"].value_counts().sort_index()
class_weights = 1.0 / train_counts
class_weights = class_weights / class_weights.sum()

loss_weights = torch.tensor(class_weights.values, dtype=torch.float32)

print("Class weights:", class_weights.to_dict())


# =====================================================
# 9. WEIGHTED RANDOM SAMPLER
# =====================================================

sample_weights = train_df["label_binary"].map(class_weights).values
sample_weights = torch.DoubleTensor(sample_weights)

sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# =====================================================
# 10. DATASET CLASS (RETURNS RAW METADATA)
# =====================================================

class HAM10000Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Build categorical vocabularies
        self.sex_vocab = {cat: i for i, cat in enumerate(df["sex"].unique())}
        self.loc_vocab = {cat: i for i, cat in enumerate(df["localization"].unique())}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)

        # RAW metadata values
        age = torch.tensor([row["age"]], dtype=torch.float32)
        sex = torch.tensor(self.sex_vocab[row["sex"]], dtype=torch.long)
        loc = torch.tensor(self.loc_vocab[row["localization"]], dtype=torch.long)

        # Label
        label = torch.tensor(row["label_binary"], dtype=torch.long)

        return img, age, sex, loc, label


# =====================================================
# 11. BUILD DATALOADERS
# =====================================================

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

train_dataset = HAM10000Dataset(train_df, transform=img_transform)
val_dataset   = HAM10000Dataset(val_df, transform=img_transform)
test_dataset  = HAM10000Dataset(test_df, transform=img_transform)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print("Dataloaders ready.")


# =====================================================
# 12. EVALUATION METRICS
# =====================================================

def evaluate_model(model, dataloader, device="cpu"):
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for img, age, sex, loc, labels in dataloader:
            img = img.to(device)
            age = age.to(device)
            sex = sex.to(device)
            loc = loc.to(device)
            labels = labels.to(device)

            outputs = model(img, age, sex, loc)

            if outputs.shape[-1] == 1:  # BCEWithLogits
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds = (probs >= 0.5).astype(int)
            else:  # CrossEntropy
                probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs),
        "confusion_matrix": confusion_matrix(all_labels, all_preds)
    }


print("\nPipeline finished successfully!")
