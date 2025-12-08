import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HAM10000Dataset(Dataset):
    """
    Dataset returning:
        baseline:   image, metadata_tensor, label
        multimodal: image, metadata_dict(age, sex, loc), label
    """

    def __init__(self, csv_path, image_dirs, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_dirs = image_dirs

        # Image transform
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        # -----------------------------------------
        # METADATA PROCESSING
        # -----------------------------------------

        # Age normalization
        self.df["age"] = self.df["age"].fillna(self.df["age"].median())
        self.df["age_norm"] = self.df["age"] / 100.0

        # Sex vocab + encoding
        self.df["sex"] = self.df["sex"].fillna("unknown")
        self.sex_vocab = {sex: i for i, sex in enumerate(sorted(self.df["sex"].unique()))}
        self.df["sex_id"] = self.df["sex"].map(self.sex_vocab)

        # Location vocab + encoding
        self.df["localization"] = self.df["localization"].fillna("unknown")
        self.loc_vocab = {loc: i for i, loc in enumerate(sorted(self.df["localization"].unique()))}
        self.df["loc_id"] = self.df["localization"].map(self.loc_vocab)

    def __len__(self):
        return len(self.df)

    def _find_image(self, image_id):
        """Search directories for matching image file."""
        for directory in self.image_dirs:
            for ext in [".jpg", ".jpeg", ".png"]:
                path = os.path.join(directory, image_id + ext)
                if os.path.exists(path):
                    return path
        raise FileNotFoundError(f"Image not found for ID {image_id}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -----------------------------------------
        # IMAGE
        # -----------------------------------------
        img_path = self._find_image(row["image_id"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # -----------------------------------------
        # LABEL
        # -----------------------------------------
        label = torch.tensor(row["label_binary"], dtype=torch.float32)

        # -----------------------------------------
        # BASELINE METADATA (tensor)
        # -----------------------------------------
        metadata_tensor = torch.tensor([
            row["age_norm"],
            float(row["sex_id"]),     # still numeric, preserves old behavior
            float(row["loc_id"])
        ], dtype=torch.float32)

        # -----------------------------------------
        # MULTIMODAL METADATA (dict)
        # -----------------------------------------
        metadata_dict = {
            "age": torch.tensor([row["age_norm"]], dtype=torch.float32),
            "sex": torch.tensor(row["sex_id"], dtype=torch.long),
            "loc": torch.tensor(row["loc_id"], dtype=torch.long)
        }

        # Both models expect: image, metadata, label
        # Baseline uses metadata_tensor
        # Multimodal uses metadata_dict
        return img, {"tensor": metadata_tensor, **metadata_dict}, label
