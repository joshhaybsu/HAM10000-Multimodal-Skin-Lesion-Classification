"""
HAM10000_preprocess.py
Preprocesses the HAM10000 metadata:
- loads metadata
- cleans missing values
- creates binary label (malignant vs benign)
- outputs a cleaned CSV: HAM10000_processed.csv
"""

import os
import pandas as pd

DATA_ROOT = "data"
RAW_METADATA = os.path.join(DATA_ROOT, "HAM10000_metadata.csv")
OUTPUT_CSV = os.path.join(DATA_ROOT, "HAM10000_processed.csv")

# ---------------------------------------------
# 1. Load metadata
# ---------------------------------------------
df = pd.read_csv(RAW_METADATA)
print("Loaded metadata:", df.shape)

# ---------------------------------------------
# 2. Create binary malignant label
# ---------------------------------------------
malignant_classes = {"mel", "bcc", "akiec"}

df["label_binary"] = df["dx"].apply(lambda x: 1 if x in malignant_classes else 0)

print("\nMalignant vs Benign distribution:")
print(df["label_binary"].value_counts())

# ---------------------------------------------
# 3. Clean metadata fields
# ---------------------------------------------
df["age"] = df["age"].fillna(df["age"].median())
df["sex"] = df["sex"].fillna("unknown")
df["localization"] = df["localization"].fillna("unknown")

# ---------------------------------------------
# 4. Save cleaned metadata
# ---------------------------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved cleaned metadata → {OUTPUT_CSV}")
