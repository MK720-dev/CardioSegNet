# 📘 **CardioSegNet: Left Ventricle Segmentation Using U-Net (Phase 1 + Phase 2)**

CardioSegNet is a research-grade cardiac MRI segmentation project using U-Net architectures.
The project is organized into **two phases**:

* **Phase 1** – Lightweight baseline U-Net (LV-only segmentation)
* **Phase 2** – Enhanced U-Net with residual blocks, attention gates, and AutoML-ready design

This README provides a detailed technical overview of the architectural choices, training pipeline, evaluation metrics, and planned extensions.

## Visual Demo

Quick demo available here:  
👉 [Download / Watch Demo](https://github.com/MK720-dev/CardioSegNet/CardioSegNet1.0_demo.mp4)

---

# ----------------------------------------------------------

# 🩺 **PHASE 1 — Baseline U-Net Model**

# ----------------------------------------------------------

## 🎯 **Goal**

Develop a simple, fast, and reliable **baseline** model for **binary segmentation** of the **Left Ventricle (LV)** on single 2D MRI slices.
This serves as the foundation for more advanced modeling in Phase 2.

---

# 1. 🧱 **Baseline Architecture Overview**

### ✔ **Input**

* Preprocessed 2D MRI slice
* Grayscale
* **128 × 128 resolution**
* Shape: `(128,128,1)`

Downsampling to 128 × 128 drastically reduces training time while preserving LV shape.

---

## 1.1 **Why a Baseline U-Net?**

The classical U-Net is ideal for medical image segmentation because:

* Works well with small datasets
* Maintains fine localization (via skip connections)
* Captures global structure via encoder bottleneck
* Provides pixel-level predictions efficiently

Phase 1 uses a **shallow 2-level encoder/decoder**, optimized for speed and clarity.

---

## 1.2 **Baseline U-Net Architecture Summary**

### **Encoder (contracting path)**

```
Conv(32) → BN → ReLU
Conv(32) → BN → ReLU
MaxPool(2)

Conv(64) → BN → ReLU
Conv(64) → BN → ReLU
MaxPool(2)
```

Captures:

* Local LV edges
* Low/mid-level structural features
* Basic texture and contrast patterns

---

### **Bottleneck**

```
Conv(128) → BN → ReLU  
Conv(128) → BN → ReLU
```

Captures:

* LV global shape
* Robust semantic representation

---

### **Decoder (expanding path)**

Uses non-learnable upsampling:

```
UpSampling(2)
Concat(skip from encoder)
Conv(64) → BN → ReLU
Conv(64) → BN → ReLU

UpSampling(2)
Concat(skip)
Conv(32) → BN → ReLU
Conv(32) → BN → ReLU
```

---

### **Output Layer**

```
Conv(1×1, 1 filter, activation = sigmoid)
```

Outputs **a single probability map** representing LV presence.

---

## 1.3 **Why These Design Choices?**

### ✔ **Shallow depth (2 encoder levels)**

* Fast prototyping
* Low VRAM usage
* Easy debugging
* Sufficient for LV segmentation

### ✔ **Batch Normalization**

Stabilizes training and accelerates convergence.

### ✔ **UpSampling over Transposed Convolution**

* Reduces checkerboard artifacts
* Faster for lightweight trials

### ✔ **1-channel Sigmoid output**

Perfect for **binary LV-only segmentation**.

---

# 2. 🧪 **Training Pipeline Overview**

### **Loss Function: BCE + Dice**

This hybrid loss prevents model collapse and handles LV class imbalance.

* **BCE** encourages pixel-level correctness
* **Dice** encourages correct shape and overlap

### **Optimizer**

```
Adam(lr=1e-4)
```

Chosen for stable convergence on segmentation tasks.

### **Dataset**

* ACDC MRI slices (HDF5 format)
* LV labeled as class “3”
* Preprocessed into 128×128 tensors
* TF dataset generator with:

  * parallel loading
  * caching
  * shuffling
  * prefetching

---

# 3. 📊 **Performance Evaluation**

Model performance is evaluated using three components:

---

## 3.1 **Dice Coefficient (Primary Metric)**

[
\mathrm{Dice} = \frac{2|P∩G|}{|P| + |G|}
]

Dice captures **overlap quality**, which is essential for medical segmentation.

It is robust to class imbalance and reflects:

* LV completeness
* Contour accuracy
* Boundary consistency

---

## 3.2 **Qualitative Evaluation (Visual)**

Slices are inspected using a **custom-built Dash viewer**, which overlays:

* **Predicted mask**
* **Ground truth**
* **Differences**

Visualization reveals:

* Apex/base failure modes
* Over-smoothing of boundaries
* Slice-to-slice inconsistencies

---

## 3.3 **Full-inference Evaluation**

The trained 2D U-Net is applied slice-by-slice to 3D volumes.
We evaluate:

* Per-slice Dice
* Per-volume Dice
* False positive & false negative regions
* Overlays with raw MRI

This provides a full anatomical assessment.

---

# 4. 🎥 **Video Demo (Coming Soon)**

A full video demonstration will be added showing:

* Viewer usage
* Slice scrolling
* GT vs prediction
* Failure modes
* Model comparison

---

# ----------------------------------------------------------

# 🚀 **PHASE 2 — Enhanced U-Net + AutoML Vision**

# ----------------------------------------------------------

Phase 2 expands the system into a **research-grade** segmentation framework with flexibility, performance, and extensibility in mind.

---

# 1. ⚙️ **Architectural Enhancements**

### **Phase 2 U-Net adds:**

### ✔ **Deeper Encoder + Decoder (4 levels)**

Allows the model to learn:

* richer anatomical context
* better apex/base handling
* more expressive features

### ✔ **Residual Blocks**

Improve gradient flow and stabilize deeper models.

### ✔ **Attention Gates (optional)**

Highlight relevant anatomical regions and suppress noise.

### ✔ **Transposed Convolutions**

Learn optimal upsampling patterns for sharper boundaries.

### ✔ **Flexible Output**

* LV-only (1 channel)
* or Multi-class (background, RV, Myo, LV)

By setting:

```
NUM_CLASSES = 1 or 4
```

Phase 2 allows easy switching.

---

# 2. 🧠 **Potential Improvements for Phase 2 and Beyond**

### **A. Architecture Improvements**

* Residual U-Net
* Attention U-Net
* UNet++ (nested skip connections)
* UNet3+
* Swin-UNet (transformer-based)
* 2.5D U-Net (use slice neighbors)
* Full 3D U-Net for volumetric consistency
* Use pretrained encoders (ResNet / EfficientNet backbones)

---

### **B. Advanced Optimizers & Schedulers**

* Ranger (Rectified Adam + Lookahead)
* AdamW (decoupled weight decay)
* SGD warm restarts
* Cosine Annealing LR
* One-cycle policy
* ReduceLROnPlateau

These can dramatically impact convergence and stability.

---

### **C. Activation Function Exploration**

* ReLU6
* LeakyReLU
* GELU
* Swish / SiLU
* Mish

These nonlinearities can help gradient flow in deeper U-Nets.

---

### **D. Performance Metrics**

Possible additional metrics:

* **Hausdorff Distance** (boundary quality)
* **95th percentile HD** (robustness to outliers)
* **Precision / Recall on LV boundary**
* **Volumetric Dice** (per volume)
* **Surface Dice**

These provide richer clinical insight than Dice alone.

---

### **E. Data Improvements**

* Elastic deformation
* Spatial dropout
* Synthetic augmentation
* Intensity normalization across scanners
* Noise injection
* Patch-based training for high resolution

---

# 3. 🤖 **Towards AutoML-Driven Architecture Search**

Phase 2 lays the groundwork for future AutoML integration.

### Potential AutoML components:

---

## ✔ 1. Architecture Advisor (Meta-Network)

A neural network that predicts:

* which architectural change is most beneficial (depth, filters, attention, etc.)
* expected Dice improvement
* when to stop training (tolerance threshold)

---

## ✔ 2. RL-based Hyperparameter Controller

An agent that proposes:

* learning rates
* batch sizes
* regularization strengths
* optimizer types

trained to maximize Dice under compute constraints.

---

## ✔ 3. Experiment Logger for Meta-Learning

Every experiment becomes a datapoint:

* architecture
* data properties
* training curves
* performance lift after changes

This produces a dataset that enables training of a higher-level AutoML advisor.

---

# 4. 🧪 **Phase 2 Training Goals**

* Achieve >0.87 Dice for LV
* Improve apex/base segmentation
* Stabilize contours
* Introduce multi-class support
* Set foundation for RL-driven architecture search

---

# 5. 🎥 **Phase 2 Demo Video (Coming Soon)**

A future demo will compare:

* Phase 1 U-Net
* Phase 2 U-Net (residual + attention)
* Multi-class capability
* Viewer enhancements (model comparison mode)

---

# ----------------------------------------------------------

# 📦 **Project Structure**

# ----------------------------------------------------------

```
CardioSegNet/
│
├── src/
│     ├── model_unet.py        # Phase I + Phase II architectures
│     ├── data_loader.py       # Slice loader (LV / multi-class)
│     ├── losses.py            # BCE+Dice, multi-class Dice/Tversky
│     ├── train.py             # Training script
│
├── viewer/
│     ├── app.py               # Interactive Dash viewer
│     ├── utils.py             # Volume loading + overlays
│
├── results/
│     ├── models/              # Saved weights / .keras models
│     ├── plots/               # Training curves (optional)
│
├── config.py                  # Global config parameters
├── requirements.txt
└── README.md
```

---




