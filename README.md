# HAM10000 Multimodal Skin Lesion Classification

## Research Question
> **Does incorporating patient metadata (age, sex, and lesion location) into a deep learning classifier improve malignant skin lesion classification performance compared to an image-only convolutional neural network on the HAM10000 dataset?**

## Overview
This project investigates whether incorporating patient metadata*(age, sex, and lesion location) alongside dermoscopic images improves malignant skin lesion classification. Using the HAM10000 dataset, comparing a baseline **image-only convolutional neural network (CNN)** against a **multimodal deep learning model** that fuses image features with metadata.

### Using the Dataset
After downloading HAM10000 from **Kaggle**, place the following into the `data/` folder:

- `HAM10000_metadata.csv`
- `HAM10000_images_part_1/`
- `HAM10000_images_part_2/`

### 1. Preprocess metadata and generate dataset splits
```bash
python data/HAM10000_preprocess.py
python data/HAM10000_gen_splits.py
```

### 2. Train both models
```bash
python train_baseline.py
python train_multimodal.py
```
### 3. Evaluate trained models
```bash
python evaluate_models.py
```

### Outputs
- Model checkpoints: `checkpoints/`
- Numerical results table: `results/test_resultsYYYYMMDD_HHMMSS.csv`
- Generated plots:
  - `results/baseline/`
  - `results/multimodal/`

## Dataset Attribution
The HAM10000 dataset was created by Philipp Tschandl, Cliff Rosendahl, and Harald Kittler.

> Tschandl, P., Rosendahl, C. and Kittler, H., 2018. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data, 5, p.180161. Available at: https://doi.org/10.1038/sdata.2018.161

The dataset is licensed under **CC BY-NC-SA 4.0:**  
https://creativecommons.org/licenses/by-nc-sa/4.0/

This repository does **not** distribute the dataset.

All analysis in this project is based on locally stored copies that are not redistributed.

## Dataset Source **(IMPORTANT)**
This project requires the **Kaggle version** of HAM10000:

https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

The ISIC Archive version has a different folder structure and metadata schema that is not compatible with this preprocessing pipeline.

## Recommended Requirements
Training is GPU-accelerated.

A CUDA-compatible GPU (≥ 8 GB VRAM) is recommended for reasonable training times.