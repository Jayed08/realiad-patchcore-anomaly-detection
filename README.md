# 🏭 Real-IAD Industrial Defect Detection & Localization (PatchCore-Style)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://realiad-patchcore-anomaly-detection-jayed.streamlit.app/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10.0-ee4c2c.svg)](https://pytorch.org/)
[![Torchvision 0.25](https://img.shields.io/badge/Torchvision-0.25.0-orange.svg)](https://pytorch.org/vision/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end normal-only industrial anomaly detection and localization system built on the **Real-IAD** dataset. The system uses a **PatchCore-style** nearest-neighbor approach with a pretrained **Wide ResNet-50-2** backbone and category-specific normal memory banks.

---

## 🌐 Live Demo

Experience the model directly in your browser:  
👉 **[Open Real-IAD Anomaly Detection App](https://realiad-patchcore-anomaly-detection-jayed.streamlit.app/)**

---

## 🔍 Overview

In industrial manufacturing, anomalies (scratches, dents, cracks, impurities) are rare and unpredictable, making supervised defect classification unfeasible. This project implements **PatchCore-style normal-only anomaly detection**:
1. **Train on defect-free nominal images only**: The model learns representations of standard parts without ever seeing defects during training.
2. **Detect anomalies at inference**: Image patches that deviate from the normal representation memory bank are flagged.
3. **Generate pixel-level defect localization maps**: Generates high-resolution anomaly heatmaps and visual inspection overlays without requiring manual pixel annotations during training.

---

## ✨ Key Features

- **Multi-Scale Deep Feature Hierarchy**: Concatenates mid-level (`Layer 2`) and high-level semantics (`Layer 3`) from a pretrained `Wide ResNet-50-2` to capture both micro-textures and structural layout ($1,536\text{-dim}$ representations).
- **High-Resolution Inspection ($512 \times 512$)**: Preserves critical surface details across $4,096$ spatial patches ($64 \times 64$ grid).
- **Memory-Efficient Memory Bank Sampling**: Streams extracted patch features and applies 1% random subsampling to construct category-specific normal memory banks while keeping memory usage manageable.
- **30 Industrial Categories**: Comprehensive evaluation across all 30 Real-IAD classes (electronics, mechanical parts, consumer goods).
- **Interactive Streamlit Web Dashboard**: Real-time image upload, category-specific threshold-based prediction, and 3-panel visualization (Original, Heatmap, Localization Overlay).
- **Dockerized Deployment**: Fully containerized CPU-based deployment with Python 3.12 for local and cloud hosting.

---

## 🏗️ Methodology & Architecture

Input images are resized to $512 \times 512$ and normalized using ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`) before feature extraction.

```text
                  Input Image (512 × 512 × 3)
                              │
                              ▼
                     Wide ResNet-50-2
               ┌──────────────┴──────────────┐
               ▼                             ▼
        Layer 2 Features              Layer 3 Features
        [512 × 64 × 64]               [1024 × 32 × 32]
               │                             │
               │ (AvgPool 3x3)               │ (AvgPool 3x3 + Bilinear 2x)
               └──────────────┬──────────────┘
                              │
                              ▼
                Concatenated Patch Features
                       [1536 × 64 × 64]
                              │
                              ▼
                  4,096 Patch Vectors (1536-d)
                              │
                              ▼
          Nearest-Neighbor Search vs Memory Bank (M)
                 min || patch_i - m_j ||_2
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       Image Anomaly Score           Anomaly Map (64 × 64)
     S(x) = max_i (min_dist)                 │ (Bilinear 8x to 512 × 512)
               │                             ▼
               ▼                     Gaussian Smoothing (σ=4.0)
     Threshold Decision                      │
    Score ≥ Threshold?                       ▼
    ├── Yes ➔ DEFECTIVE               Heatmap & Overlay
    └── No  ➔ NORMAL
```

### Anomaly Scoring Formulation

For an image $x$, patch feature vectors $\mathcal{P}(x) = \{p_1, p_2, \dots, p_N\} \subset \mathbb{R}^{1536}$ are extracted. Given the nominal memory bank $\mathcal{M} \subset \mathbb{R}^{1536}$, the anomaly score $s(p_i)$ for patch $p_i$ is its distance to the closest normal patch:

$$s(p_i) = \min_{m \in \mathcal{M}} \|p_i - m\|_2$$

The image-level anomaly score $S(x)$ is defined as the maximum patch distance:

$$S(x) = \max_{p \in \mathcal{P}(x)} s(p)$$

A sample is classified as **DEFECTIVE** if $S(x) \ge \tau_{\text{category}}$, where $\tau$ is determined by the 99th-percentile score of validation normal samples.

---

## 📊 Benchmark Performance

Evaluated on the official **Real-IAD (Single-View UIAD)** test set across all 30 object categories.

> **Note:** The model uses only normal training images to construct the category-specific memory banks. Test defect labels and ground-truth masks are used exclusively for evaluation.

### Aggregate Test Metrics

| Metric Level | Metric | Mean Score | Description |
| :--- | :--- | :---: | :--- |
| **Sample-Level** | **S-AUROC** | **90.56%** | Area Under Receiver Operating Characteristic (Sample) |
| **Sample-Level** | **Accuracy** | **79.18%** | Binary classification accuracy using category-specific validation-derived thresholds |
| **Sample-Level** | **Precision** | **93.12%** | Precision of flagged defective components |
| **Sample-Level** | **Recall** | **58.13%** | Detection sensitivity on test defect samples |
| **Sample-Level** | **F1-Score** | **69.86%** | Harmonic mean of precision and recall |
| **Pixel-Level** | **I-AUROC** | **99.06%** | Pixel-level Area Under ROC Curve |
| **Pixel-Level** | **P-AUPRO** | **88.03%** | Per-Region Overlap AUC (integrated up to 0.30 FPR) |

<details>
<summary><b>📈 Click to expand complete 30-category benchmark breakdown</b></summary>

| Category | S-AUROC | Accuracy | Precision | Recall | F1-Score | I-AUROC (Pixel) | P-AUPRO (0.30) | Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `audiojack` | 0.8288 | 0.8723 | 0.9655 | 0.6573 | 0.7821 | 0.9944 | 0.9451 | 21.4030 |
| `bottle_cap` | 0.9618 | 0.8278 | 0.9814 | 0.5985 | 0.7435 | 0.9966 | 0.8929 | 18.1935 |
| `button_battery` | 0.8479 | 0.7161 | 0.9615 | 0.5396 | 0.6912 | 0.9947 | 0.9163 | 20.8063 |
| `end_cap` | 0.8329 | 0.6732 | 0.8631 | 0.5354 | 0.6608 | 0.9883 | 0.8100 | 19.3390 |
| `eraser` | 0.9379 | 0.8638 | 0.9870 | 0.6468 | 0.7815 | 0.9978 | 0.8465 | 19.8173 |
| `fire_hood` | 0.9596 | 0.9114 | 0.9034 | 0.7751 | 0.8344 | 0.9962 | 0.8639 | 22.4464 |
| `mint` | 0.7268 | 0.5762 | 0.8963 | 0.3267 | 0.4788 | 0.9857 | 0.7043 | 18.7771 |
| `mounts` | 0.9264 | 0.7920 | 0.9661 | 0.4750 | 0.6369 | 0.9970 | 0.8771 | 22.7317 |
| `pcb` | 0.9023 | 0.6224 | 0.9674 | 0.4000 | 0.5660 | 0.9974 | 0.8749 | 21.3793 |
| `phone_battery` | 0.9362 | 0.8015 | 0.9943 | 0.5748 | 0.7284 | 0.9968 | 0.8633 | 22.0916 |
| `plastic_nut` | 0.8607 | 0.8339 | 0.7551 | 0.3136 | 0.4431 | 0.9975 | 0.8821 | 21.0721 |
| `plastic_plug` | 0.9573 | 0.8889 | 0.9800 | 0.7481 | 0.8485 | 0.9969 | 0.7802 | 20.1564 |
| `porcelain_doll` | 0.8674 | 0.8227 | 0.8750 | 0.5357 | 0.6646 | 0.9799 | 0.8552 | 18.3277 |
| `regulator` | 0.8646 | 0.9079 | 0.6538 | 0.5152 | 0.5763 | 0.9974 | 0.9166 | 18.4501 |
| `rolled_strip_base` | 0.9861 | 0.9381 | 0.9812 | 0.9253 | 0.9525 | 0.9985 | 0.9765 | 19.4087 |
| `sim_card_set` | 0.9780 | 0.9073 | 0.9742 | 0.8586 | 0.9128 | 0.9982 | 0.9557 | 20.1280 |
| `switch` | 0.9304 | 0.6581 | 0.9665 | 0.4843 | 0.6453 | 0.9713 | 0.9171 | 21.9950 |
| `tape` | 0.9840 | 0.8842 | 1.0000 | 0.6800 | 0.8095 | 0.9986 | 0.9886 | 20.8826 |
| `terminalblock` | 0.9701 | 0.9016 | 0.9765 | 0.8448 | 0.9059 | 0.9985 | 0.9576 | 19.0404 |
| `toothbrush` | 0.8894 | 0.6250 | 0.9893 | 0.4057 | 0.5754 | 0.9929 | 0.9433 | 25.2239 |
| `toy` | 0.8652 | 0.5703 | 0.9592 | 0.3730 | 0.5371 | 0.9319 | 0.8768 | 21.0941 |
| `toy_brick` | 0.8664 | 0.8336 | 0.9333 | 0.6437 | 0.7619 | 0.9895 | 0.8850 | 22.0600 |
| `transistor1` | 0.8999 | 0.4523 | 0.9589 | 0.1493 | 0.2583 | 0.9903 | 0.8194 | 22.5007 |
| `u_block` | 0.9209 | 0.8879 | 0.8987 | 0.5635 | 0.6927 | 0.9950 | 0.9584 | 17.5335 |
| `usb` | 0.9558 | 0.8979 | 0.9683 | 0.8053 | 0.8793 | 0.9980 | 0.9803 | 19.6187 |
| `usb_adaptor` | 0.8617 | 0.7660 | 0.9779 | 0.4750 | 0.6394 | 0.9952 | 0.8479 | 19.1885 |
| `vcpill` | 0.8794 | 0.8102 | 0.8908 | 0.5096 | 0.6483 | 0.9864 | 0.7087 | 19.5500 |
| `wooden_beads` | 0.8725 | 0.6911 | 0.9733 | 0.4643 | 0.6287 | 0.9854 | 0.8479 | 20.9424 |
| `woodstick` | 0.9194 | 0.8925 | 0.7642 | 0.6983 | 0.7297 | 0.9803 | 0.8299 | 21.0806 |
| `zipper` | 0.9770 | 0.9280 | 0.9745 | 0.9160 | 0.9443 | 0.9909 | 0.8876 | 17.7488 |
| **MEAN** | **0.9056** | **0.7918** | **0.9312** | **0.5813** | **0.6986** | **0.9906** | **0.8803** | **—** |

</details>

---

## 📁 Project Structure

```text
├── app.py                         # Streamlit web application for visual inspection
├── real-iad-anomaly-detection.ipynb         # Full Jupyter training, feature extraction & evaluation pipeline
├── requirements.txt               # Python package dependencies (Python 3.12)
├── Dockerfile                     # Containerization setup for Python 3.12 runtime
├── realiad_patchcore_model.pt     # Serialized artifact (backbone state, memory banks, thresholds)
└── README.md                      # Project documentation & benchmark report
```

---

## 💻 Installation & Setup

### Prerequisites
- Python **3.12**
- Git

### 1. Clone the repository
```bash
git clone https://github.com/Jayed08/realiad-patchcore-anomaly-detection.git
cd realiad-patchcore-anomaly-detection
```

### 2. Create and activate a virtual environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Running Streamlit Web App

Launch the local inspection web interface:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

#### Interface Workflow:
1. **Sidebar**: Inspect model architecture parameters, feature dimensions, and test set benchmark statistics.
2. **Category Selection**: Choose your target industrial object class (e.g., `zipper`, `pcb`, `bottle_cap`).
3. **Upload Image**: Drag and drop inspection images (`.png`, `.jpg`, `.jpeg`).
4. **Analyze**: Click **Analyze** for instant `NORMAL` vs `DEFECTIVE` prediction, anomaly score, decision threshold, and defect heatmap overlay.

---

### 2. Python Inference API

Run inference programmatically in Python:

```python
from PIL import Image
from app import load_model, predict

# 1. Load detector model
feature_extractor, memory_banks, thresholds, transform, config, device, _ = load_model("realiad_patchcore_model.pt")

# 2. Open an inspection image
img = Image.open("sample_inspection.png").convert("RGB")

# 3. Perform defect detection
result = predict(
    image=img,
    category="zipper",
    feature_extractor=feature_extractor,
    memory_banks=memory_banks,
    thresholds=thresholds,
    transform=transform,
    config=config,
    device=device
)

print(f"Status:        {'DEFECTIVE' if result['is_anomaly'] else 'NORMAL'}")
print(f"Anomaly Score: {result['score']:.2f}")
print(f"Threshold:     {result['threshold']:.2f}")
```

---

### 3. Training & Notebook Exploration

For end-to-end dataset downloading, feature extraction, memory bank subsampling, and metric evaluation, open the Jupyter Notebook:

```bash
jupyter lab real-iad-anomaly-detection.ipynb
```

---

## 📦 Dataset (Real-IAD)

The **Real-IAD** dataset is a large-scale benchmark for industrial anomaly detection containing:
- **30 object categories** spanning metal, plastic, fabric, wood, and electronic components.
- Complex anomalies: scratches, pits, deformations, contamination, component missing, and surface stains.
- Multi-view / single-view imaging setups with pixel-level ground-truth defect annotations.

Dataset link: [Hugging Face Real-IAD](https://huggingface.co/datasets/Real-IAD/Real-IAD)

---

## 🖼️ Results Visualization

The inspection interface produces three synchronized inspection outputs:
1. **Original Image**: Raw $512 \times 512$ industrial surface.
2. **Anomaly Heatmap**: Continuous Jet-colormap intensity gradient showing localized patch anomaly distance.
3. **Localization Overlay**: $50\% / 50\%$ blended alpha overlay highlighting the exact anomalous region for quality control operators.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
