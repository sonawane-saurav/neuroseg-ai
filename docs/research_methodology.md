# Research Methodology & Experimental Contributions

**Project:** Deep Learning-Based Brain Tumor Segmentation & Clinical Diagnostic Intelligence  
**Dataset Reference:** BRISC 2025 Brain MRI Collection (Fateh et al., arXiv:2506.14318)  
**Authorship & Academic Attribution:** Research Prototype Developed for Academic Evaluation  

---

## 1. Problem Statement & Research Objectives

Brain tumors (such as gliomas, meningiomas, and pituitary adenomas) represent diverse pathologies requiring accurate localization, volumetric measurement, and surgical margin delineation on Magnetic Resonance Imaging (MRI). Manual contouring is time-consuming and subject to significant inter-observer variability.

While convolutional neural networks (e.g., U-Net) have demonstrated remarkable efficacy in medical image segmentation, conventional models suffer from three fundamental clinical limitations:
1. **Uncalibrated Confidence:** Standard networks output uncalibrated sigmoid probabilities that do not reflect true posterior uncertainty, especially along infiltrative tumor margins.
2. **Vulnerability to Scanner Hazards:** Medical imaging frequently encounters artifacts such as patient motion, low tissue contrast, thermal/sensor electronic noise, and reduced scanner resolution.
3. **Task Isolation:** Standard segmentation systems ignore high-level pathology classification, missing opportunities for joint multi-task representation learning.

To address these challenges, this project investigates three distinct research questions:

---

## 2. Research Questions & Experimental Formulations

### RQ1: Segmentation Accuracy & Class Imbalance
> *How accurately can a deep learning encoder-decoder model segment brain tumors from T1-weighted MRI slices under severe foreground-background pixel imbalance?*

**Formulation:**
Tumor lesions in the BRISC 2025 cohort occupy between 0.05% and 18.2% of cranial pixels (mean coverage: 3.2%). A standard Cross-Entropy loss leads to trivial background bias. We implement a hybrid loss:
$$\mathcal{L}_{\text{seg}} = w_{\text{BCE}} \cdot \mathcal{L}_{\text{BCE}} + w_{\text{Dice}} \cdot \mathcal{L}_{\text{Dice}}$$
where the soft Dice loss is formulated as:
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i} p_i g_i + \epsilon}{\sum_{i} p_i + \sum_{i} g_i + \epsilon}$$
with Laplace smoothing $\epsilon = 1.0$.

---

### RQ2: Epistemic Uncertainty via Monte Carlo Dropout
> *How reliable are segmentation predictions, and can test-time Monte Carlo Dropout identify clinically ambiguous tumor margins?*

**Theoretical Basis (Gal & Ghahramani, 2016):**
Standard neural network inference uses deterministic weights. By viewing dropout as variational inference in deep Gaussian processes, we retain dropout layers ($p = 0.2$) in training mode during inference while keeping batch normalization frozen in evaluation mode.

For an MRI slice $x$, we execute $T$ stochastic forward passes $\{f_{\hat{W}_t}(x)\}_{t=1}^T$ with randomly sampled weight configurations $\hat{W}_t \sim q(W)$:
1. **Predictive Mean Probability:**
   $$\mu(x) = \frac{1}{T} \sum_{t=1}^T \text{Sigmoid}(f_{\hat{W}_t}(x))$$
2. **Epistemic Predictive Variance (Uncertainty):**
   $$\sigma^2(x) = \frac{1}{T} \sum_{t=1}^T \left( \text{Sigmoid}(f_{\hat{W}_t}(x)) - \mu(x) \right)^2$$
3. **Predictive Shannon Entropy:**
   $$H(x) = -\mu(x) \log_2(\mu(x) + \epsilon) - (1 - \mu(x)) \log_2(1 - \mu(x) + \epsilon)$$

**Findings:**
High variance concentrates sharply along the **perimeter margin** of the tumor, corresponding to partial volume voxels and infiltrative glioma margins where radiologist contours exhibit the greatest uncertainty.

---

### RQ3: Robustness Under Image Quality Degradation & Multi-Task Synergy
> *How severely does synthetic image quality degradation impair segmentation, and can joint multi-task learning improve representation quality?*

**1. Controlled Synthetic Degradation Pipeline:**
We simulate four physical scanner hazards across five discrete severity levels ($s \in \{1, 2, 3, 4, 5\}$):
- **Gaussian Electronic Noise:** $\mathcal{N}(0, \sigma_s^2)$ with $\sigma \in [0.03, 0.22]$
- **Low Contrast Dynamic Range Compression:** $I' = \alpha_s (I - I_{\text{mid}}) + I_{\text{mid}}$ with $\alpha \in [0.85, 0.20]$
- **Motion Blur Artifacts:** Linear convolution kernel of dimension $k_s \in [3, 21]$ pixels at 45° angle.
- **Low-Resolution Scanner Downsampling:** Spatial downsampling by factors $[1.5\times, 6.0\times]$ followed by bicubic upsampling.

**2. Multi-Task Learning Architecture:**
We construct a shared encoder with two task-specific heads:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda \cdot \mathcal{L}_{\text{cls}}$$
where $\mathcal{L}_{\text{cls}}$ is Cross-Entropy loss over 4 pathological categories (Glioma, Meningioma, Pituitary Adenoma, Healthy Control) and $\lambda = 0.5$.

**Findings:**
Multi-task learning encourages the shared encoder to capture both global semantic features (tumor type and gross anatomical distortion) and local boundary details, yielding a positive transfer effect for both segmentation and classification.

---

## 3. Dataset & Split Integrity

The BRISC 2025 dataset provides 6,000 T1 MRI slices (4,793 with expert tumor segmentations).
- **Patient Identifier Limitation:** Patient identifiers are not provided in the manifest; slices are individually indexed.
- **Mitigation:** The held-out test partition (860 segmentation pairs / 1,000 classification images) is structurally isolated and never seen during training or hyperparameter selection. Validation sets are carved out using deterministic stratified random seeds.

---

## 4. Academic Disclaimer

This system is an academic research prototype designed for computational vision and deep learning experimentation. It is not clinically approved or validated for medical diagnostics, surgical guidance, or patient treatment decisions.
