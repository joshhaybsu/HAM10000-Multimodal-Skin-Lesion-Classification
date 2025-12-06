# HAM10000 Multimodal Skin Lesion Classification

## Research Question
> **Does incorporating patient metadata (age, sex, and lesion location) into a deep learning classifier improve malignant skin lesion classification performance compared to an image-only convolutional neural network on the HAM10000 dataset?**

## Overview
This project investigates whether incorporating patient metadata*(age, sex, and lesion location) alongside dermoscopic images improves malignant skin lesion classification. Using the HAM10000 dataset, comparing a baseline **image-only convolutional neural network (CNN)** against a **multimodal deep learning model** that fuses image features with metadata.

### Using the Dataset
After downloading HAM10000, place the following files/directories into the `data/` folder:

- `HAM10000_metadata.csv`
- `HAM10000_images_part_1/`
- `HAM10000_images_part_2/`

Then run the preprocessing pipeline:
```bash
python data/HAM10000_pipeline.py
```

## Dataset Attribution
The HAM10000 dataset was created by Philipp Tschandl, Cliff Rosendahl, and Harald Kittler.

> Tschandl, P., Rosendahl, C. and Kittler, H., 2018. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data, 5, p.180161. Available at: https://doi.org/10.1038/sdata.2018.161

The dataset is licensed under **CC BY-NC-SA 4.0:**  
https://creativecommons.org/licenses/by-nc-sa/4.0/

This repository does **not** distribute the dataset.

All analysis in this project is based on locally stored copies that are not redistributed.

### **Dataset Source (IMPORTANT)**
This project requires the **Kaggle version** of HAM10000:

https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Do **not** use the ISIC Archive version, as its metadata format and folder structure are different and incompatible with the preprocessing pipeline used.
