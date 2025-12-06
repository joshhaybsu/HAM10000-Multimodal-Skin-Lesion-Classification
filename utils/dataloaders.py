import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from utils.dataset import HAM10000Dataset


def get_transforms():
    """Default image transforms."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


def create_sampler(labels):
    """Weighted sampler for handling class imbalance."""
    labels = np.array(labels)
    class_counts = np.bincount(labels.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels.astype(int)]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def get_dataloaders(split_dir, image_dirs, batch_size=32):
    """Build train, val, test dataloaders."""

    train_csv = f"{split_dir}/train.csv"
    val_csv = f"{split_dir}/val.csv"
    test_csv = f"{split_dir}/test.csv"

    transform = get_transforms()

    train_ds = HAM10000Dataset(train_csv, image_dirs, transform)
    val_ds   = HAM10000Dataset(val_csv,  image_dirs, transform)
    test_ds  = HAM10000Dataset(test_csv, image_dirs, transform)

    sampler = create_sampler(train_ds.df["label_binary"].values)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, test_loader
