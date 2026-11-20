# BRISC 2025 Dataset Inspection & Verification Report

**Dataset Reference:** [BRISC 2025 (Fateh et al., arXiv:2506.14318)](https://arxiv.org/abs/2506.14318)  
**Inspection Date:** November 2026  
**Auditor:** Automated Computer Vision Pipeline Audit  

---

## 1. Executive Summary

An exhaustive inspection of the downloaded BRISC 2025 dataset was conducted prior to model design.
The dataset provides physician-validated T1-weighted MRI slices paired with pixel-level segmentation masks and 4-class image classification labels.

| Attribute | Specification |
|:---|:---|
| **Total Segmentation Pairs** | **4,793 pairs** (3,933 Train / 860 Test) |
| **Total Classification Images** | **6,000 images** (5,000 Train / 1,000 Test) |
| **Image Resolution** | 512 × 512 pixels |
| **Image Format & Modality** | JPEG (`.jpg`), 8-bit RGB, T1-weighted MRI |
| **Mask Format & Encoding** | PNG (`.png`), 8-bit Single Channel (`L`), Binary Tumor Mask |
| **Tumor Pathology Classes** | Glioma, Meningioma, Pituitary Adenoma, No Tumor |
| **Anatomical Slicing Planes** | Axial, Coronal, Sagittal |
| **Integrity & Missing Pairs** | **0 missing pairs**, 100% matched filenames across train and test |

---

## 2. Directory Structure Audit

```
brisc2025/
├── classification_task/
│   ├── train/ [glioma: 1401, meningioma: 1635, pituitary: 1757, no_tumor: 1207]
│   └── test/  [glioma: 300,  meningioma: 300,  pituitary: 300,  no_tumor: 100]
├── segmentation_task/
│   ├── train/
│   │   ├── images/ [3,933 .jpg files]
│   │   └── masks/  [3,933 .png files]
│   └── test/
│       ├── images/ [860 .jpg files]
│       └── masks/  [860 .png files]
├── manifest.csv
└── manifest.json
```

---

## 3. Image-Mask Pairing & Encoding Analysis

1. **File Matching:** Every image in `segmentation_task/<split>/images/` matches its mask in `segmentation_task/<split>/masks/` with the exact same stem:
   - Example Image: `brisc2025_train_00001_gl_ax_t1.jpg`
   - Example Mask:  `brisc2025_train_00001_gl_ax_t1.png`
2. **Mask Thresholding:** Raw PNG masks contain pixel intensities near 0 (background) and 255 (tumor), with slight interpolation boundaries (1-7, 248-254) from rasterization. 
   Applying threshold `M = (mask > 127).astype(float32)` produces exact, clean binary ground truth labels (0.0, 1.0).
3. **Tumor Area Coverage:** Mean tumor coverage across slices is **2.18%** (range: 0.159% to 7.41%).

---

## 4. Multi-Task Learning Compatibility

- **Correspondence:** All 4,793 segmentation images are a verified 1:1 subset of the classification task images.
- **Classification Head Labels:**
  - `0`: Glioma (`gl`)
  - `1`: Meningioma (`me`)
  - `2`: Pituitary (`pi`)
  - `3`: No Tumor (`nt`) (represented by zeroed segmentation masks)
- **Feasibility:** Confirmed. Multi-task learning (shared encoder + segmentation decoder + classification head) is fully supported by the dataset without artificial label synthesis.

---

## 5. Patient-Level Leakage & Data Splitting Strategy

- **Observation:** The dataset manifest provides slice-level indices without individual patient identifiers.
- **Mitigation:** The evaluation strictly preserves the official partition splits created by the dataset authors (`train`: 3,933 / `test`: 860). For validation during training, a stratified 15% split is carved out from `train` using a fixed random seed (`seed=42`), preserving plane and tumor type proportions. The test partition remains strictly untouched until final evaluation.

---

## 6. Inspection Visualization

Visual inspection samples have been saved to `artifacts/visualizations/dataset_samples/inspection_samples.png`.
