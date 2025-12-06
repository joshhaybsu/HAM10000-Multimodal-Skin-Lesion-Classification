"""
HAM10000_generate_splits.py
Creates reproducible train/val/test splits:
- lesion-level GroupShuffleSplit to prevent leakage
- stratified by malignant vs benign
"""

import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

DATA_ROOT = "data"
INPUT_CSV = os.path.join(DATA_ROOT, "HAM10000_processed.csv")

# --- NEW: Write splits to data/splits/ ---
SPLIT_DIR = os.path.join(DATA_ROOT, "splits")
os.makedirs(SPLIT_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(SPLIT_DIR, "train.csv")
VAL_CSV   = os.path.join(SPLIT_DIR, "val.csv")
TEST_CSV  = os.path.join(SPLIT_DIR, "test.csv")

TRAIN_FRACTION = 0.70
VAL_FRACTION   = 0.15
RANDOM_STATE   = 42

# ---------------------------------------------
# 1. Load preprocessed metadata
# ---------------------------------------------
df = pd.read_csv(INPUT_CSV)
print("Loaded cleaned metadata:", df.shape)

# ---------------------------------------------
# 2. Grouped train split (lesion-level)
# ---------------------------------------------
groups = df["lesion_id"]
labels = df["label_binary"]

gss = GroupShuffleSplit(
    n_splits=1,
    train_size=TRAIN_FRACTION,
    random_state=RANDOM_STATE
)

train_idx, temp_idx = next(gss.split(df, labels, groups))
train_df = df.iloc[train_idx].copy()
temp_df  = df.iloc[temp_idx].copy()

# ---------------------------------------------
# 3. Stratified val/test split
# ---------------------------------------------
val_ratio_of_temp = VAL_FRACTION / (1 - TRAIN_FRACTION)

val_idx, test_idx = train_test_split(
    temp_df.index,
    test_size=1 - val_ratio_of_temp,
    stratify=temp_df["label_binary"],
    random_state=RANDOM_STATE
)

val_df = temp_df.loc[val_idx].copy()
test_df = temp_df.loc[test_idx].copy()

# ---------------------------------------------
# 4. Save splits
# ---------------------------------------------
train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print("\nSplit sizes:")
print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

print("\nSaved splits to:", SPLIT_DIR)
