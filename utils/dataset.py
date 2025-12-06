import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HAM10000Dataset(Dataset):
    """
    Dataset returning:
        image_tensor, metadata_tensor, label_tensor
    """

    def __init__(self, csv_path, image_dirs, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_dirs = image_dirs

        # Default transform
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        # ---- Metadata processing ----
        # Normalise age to [0,1]
        self.df["age"] = self.df["age"].fillna(self.df["age"].median())
        self.df["age_norm"] = self.df["age"] / 100.0

        # Encode sex → {0, 1}
        self.df["sex"] = self.df["sex"].fillna("unknown")
        self.sex_map = {"male": 1, "female": 0, "unknown": 0.5}
        self.df["sex_encoded"] = self.df["sex"].map(self.sex_map)

        # Encode location as integer, model will embed later
        self.df["localization"] = self.df["localization"].fillna("unknown")
        self.loc_vocab = {loc: i for i, loc in enumerate(self.df["localization"].unique())}
        self.df["location_encoded"] = self.df["localization"].map(self.loc_vocab)

    def __len__(self):
        return len(self.df)

    def _find_image(self, image_id):
        """Search both directories for matching PNG/JPG."""
        possible_exts = [".jpg", ".jpeg", ".png"]
        for directory in self.image_dirs:
            for ext in possible_exts:
                path = os.path.join(directory, image_id + ext)
                if os.path.exists(path):
                    return path
        raise FileNotFoundError(f"Image not found for ID {image_id}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Load image ---
        image_path = self._find_image(row["image_id"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # --- Metadata tensor ---
        metadata = torch.tensor([
            row["age_norm"],
            row["sex_encoded"],
            row["location_encoded"]
        ], dtype=torch.float32)

        # --- Label ---
        label = torch.tensor(row["label_binary"], dtype=torch.float32)

        return image, metadata, label
