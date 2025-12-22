# **CardioSegNet: Left Ventricle Segmentation Using U-Net (Phase 1 + Phase 2)**

CardioSegNet is a research-grade cardiac MRI segmentation project using U-Net architectures.
The project is organized into **two phases**:

* **Phase 1** – Lightweight baseline U-Net (LV-only segmentation)
* **Phase 2** – Enhanced U-Net with residual blocks, attention gates, and AutoML-ready design

This README provides a detailed technical overview of the architectural choices, training pipeline, evaluation metrics, and planned extensions.

## Visual Demo

Quick Phase 1 demo available here (click **view raw** to download):  
👉 [Download / Watch Demo](./CardioSegNet1.0_demo.mp4)

Phase 2 demo will be available soon.
👉 [Download / Watch Demo](./CardioSegNet2.0_demo.mp4)

---
# **PHASE 1 — Baseline U-Net Model**

## **Goal**

Develop a simple, fast, and reliable **baseline** model for **binary segmentation** of the **Left Ventricle (LV)** on single 2D MRI slices.  
This phase establishes a stable end-to-end pipeline that serves as the foundation for deeper architectures and multi-class segmentation in Phase 2.

---

# **Baseline Architecture Overview**

### **Input**

* Preprocessed 2D MRI slice  
* Grayscale  
* **128 × 128 resolution**  
* Shape: `(128, 128, 1)`

Downsampling slices to 128 × 128 significantly reduces computational cost while preserving the geometric structure of the LV, enabling rapid experimentation and debugging.

---

## **Why a Baseline U-Net?**

The classical U-Net architecture is particularly well-suited for medical image segmentation because it:

* Performs well with limited labeled data  
* Preserves spatial localization through skip connections  
* Learns global context via a bottleneck representation  
* Produces dense, pixel-level predictions efficiently  

In Phase 1, a **shallow two-level U-Net** is used to prioritize training speed, architectural clarity, and interpretability while remaining expressive enough for LV segmentation.

---

## **Baseline U-Net Architecture Summary**

The baseline architecture follows an **encoder–bottleneck–decoder** structure with symmetric skip connections.

---

### **Encoder (contracting path)**

```

Conv(32) → BN → ReLU
Conv(32) → BN → ReLU
MaxPool(2)

Conv(64) → BN → ReLU
Conv(64) → BN → ReLU
MaxPool(2)

```

The encoder progressively reduces spatial resolution while increasing feature dimensionality.

It learns:
* Low-level intensity gradients and edges (first level)
* Local LV boundary fragments and texture patterns
* Mid-level representations capturing partial chamber geometry

---

### **Bottleneck**

```

Conv(128) → BN → ReLU
Conv(128) → BN → ReLU

```

The bottleneck operates at the lowest spatial resolution and captures:

* Global LV shape  
* Coarse anatomical context  
* Robust semantic representation invariant to small spatial perturbations  

This representation provides the decoder with a holistic view of the cardiac structure.

---

### **Decoder (expanding path)**

The decoder progressively restores spatial resolution while refining segmentation boundaries using skip connections.

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

The decoder:
* Integrates global context from the bottleneck  
* Recovers fine spatial detail from encoder features  
* Sharpens LV contours and boundary transitions  

Non-learnable upsampling is used for stability and to avoid checkerboard artifacts.

---

### **Output Layer**

```

Conv(1×1, 1 filter, activation = sigmoid)

```

Produces a **single probability map** indicating LV presence at each pixel.

Each pixel value represents:

$$
P(\text{LV} \mid \text{pixel})
$$

---

## **Why These Design Choices?**

### ✔ **Shallow Depth (2 Encoder Levels)**

* Enables fast prototyping  
* Reduces overfitting risk  
* Low GPU memory usage  
* Sufficient representational capacity for LV-only segmentation  

---

### ✔ **Batch Normalization**

Batch normalization stabilizes gradient flow and accelerates convergence, which is especially beneficial for MRI data with varying intensity distributions.

---

### ✔ **UpSampling instead of Transposed Convolution**

* Avoids checkerboard artifacts  
* Faster and more stable for baseline experiments  
* Simplifies architectural reasoning  

---

### ✔ **Single-Channel Sigmoid Output**

A sigmoid output is mathematically aligned with binary segmentation and pairs naturally with Dice-based losses.

---

# **Training Pipeline Overview**

### **Loss Function: Binary Cross-Entropy + Dice**

Training uses a **hybrid loss function** that combines **Binary Cross-Entropy (BCE)** and **Dice Loss**.  
This formulation balances pixel-wise classification accuracy with global shape consistency, which is essential for medical segmentation tasks with class imbalance.

---

#### **Binary Cross-Entropy (BCE)**

Binary Cross-Entropy measures pixel-level classification error between the predicted LV probability map \( \hat{p} \) and the ground-truth mask \( y \in \{0,1\} \):

$$
\mathcal{L}_{\text{BCE}} = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i) \right]
$$

BCE encourages accurate per-pixel predictions but does not explicitly enforce spatial or shape-level consistency.

---

#### **Dice Loss**

The Dice coefficient measures overlap between the predicted segmentation \( P \) and the ground truth \( G \):

$$
\mathrm{Dice} = \frac{2 |P \cap G|}{|P| + |G|}
$$

The corresponding Dice loss is defined as:

$$
\mathcal{L}_{\text{Dice}} = 1 - \mathrm{Dice}
$$

Dice loss is robust to class imbalance and directly optimizes for region-level overlap and anatomical completeness.

---

#### **Combined Hybrid Loss**

The final training objective is defined as:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}
$$

This hybrid formulation prevents trivial background-dominant solutions and encourages the model to learn accurate LV shape, boundaries, and spatial extent.

---

### **Optimizer**

```

Adam(lr = 1e-4)

```

Adam is chosen for its stable convergence properties and effectiveness in dense prediction tasks.

---

### **Dataset**

* ACDC cardiac MRI dataset  
* Slice-based HDF5 format  
* LV encoded as class label `3`  
* Preprocessed into `128 × 128` tensors  

The TensorFlow data pipeline includes:
* parallel file loading  
* on-the-fly preprocessing  
* dataset shuffling  
* batching  
* prefetching for GPU utilization  

---

# **Performance Evaluation**

Model performance is assessed using a combination of **quantitative metrics** and **qualitative visual inspection**.

---

## **Dice Coefficient (Primary Metric)**

$$
\mathrm{Dice} = \frac{2 |P \cap G|}{|P| + |G|}
$$

The Dice coefficient measures overlap between prediction \(P\) and ground truth \(G\).

It captures:
* LV completeness  
* Boundary accuracy  
* Shape consistency  

Dice is robust to class imbalance and is therefore the primary evaluation metric.

---

## **Qualitative Evaluation (Visual Inspection)**

Predictions are inspected using a **custom-built Dash-based slice viewer**, which overlays:

* Predicted segmentation mask  
* Ground truth annotation  
* Raw MRI slice  

Visual inspection reveals:
* Apex and base failure modes  
* Boundary smoothness  
* Over-segmentation or under-segmentation  
* Slice-to-slice consistency  

This qualitative analysis is critical for clinical interpretability.

---

## **Full-Inference Evaluation on Volumes**

Although the model is trained on 2D slices, inference is performed **slice-by-slice on full 3D MRI volumes**.

Evaluation includes:
* Per-slice Dice scores  
* Per-volume Dice aggregation  
* False positive and false negative analysis  
* Overlay inspection across anatomical planes  

This provides a complete anatomical assessment and validates slice-based training for volumetric inference.

---

# Phase 2: Deep Multi-Class U-Net for Cardiac MRI Segmentation

In Phase 2, CardioSegNet was extended from a lightweight binary segmentation model to a **deep multi-class U-Net** capable of segmenting the **left ventricle (LV), right ventricle (RV), and myocardium (MYO)** from cardiac MRI slices. This phase preserves the original slice-based pipeline while increasing model depth, input resolution, and label complexity.

## **Goal**
Phase 2 expands the system into a **research-grade** segmentation framework with flexibility, performance, and extensibility in mind.

---

## Model Architecture

The Phase 2 model is a **four-level U-Net** operating on **256×256 grayscale MRI slices**. It follows a symmetric encoder–decoder design with skip connections, allowing the network to jointly learn **high-level anatomical semantics** and **fine-grained spatial detail**.

### Input

* Shape: `256 × 256 × 1`
* Single MRI slice
* Intensity-normalized prior to training

---

## Encoder (Contracting Path)

The encoder progressively reduces spatial resolution while increasing the number of learned feature channels.
Each encoder level consists of:

* two `3×3` convolutions (padding = same),
* Batch Normalization,
* ReLU activation,
* followed by `2×2` max pooling.

### Encoder Level 1

* Resolution: `256 × 256`
* Filters: `32`
* Learns:

  * local intensity gradients,
  * low-level edges,
  * early boundary cues for LV and myocardium.

---

### Encoder Level 2

* Resolution: `128 × 128`
* Filters: `64`
* Learns:

  * larger contour fragments,
  * partial chamber shapes,
  * contrast differences between blood pool and muscle tissue.

---

### Encoder Level 3

* Resolution: `64 × 64`
* Filters: `128`
* Learns:

  * mid-level anatomical structures,
  * curved myocardial wall segments,
  * relative spatial arrangements of LV and RV regions.

---

### Encoder Level 4

* Resolution: `32 × 32`
* Filters: `256`
* Learns:

  * high-level regional cardiac semantics,
  * chamber-scale geometry,
  * contextual relationships between cardiac structures.

---

## Bottleneck

* Resolution: `16 × 16`
* Filters: `512`

The bottleneck captures **global cardiac context**, integrating information across the entire slice.
At this stage, each spatial location represents a large portion of the heart, encoding coarse topology and overall anatomical configuration.

---

## Decoder (Expanding Path)

The decoder mirrors the encoder structure, progressively increasing spatial resolution while reducing feature dimensionality.
Each decoder level performs:

* upsampling,
* concatenation with corresponding encoder features via skip connections,
* two `3×3` convolutions with BatchNorm and ReLU.

### Decoder Level 4

* Resolution: `32 × 32`
* Filters: `256`
* Combines:

  * global context from the bottleneck,
  * high-level encoder features.
* Refines coarse anatomical regions.

---

### Decoder Level 3

* Resolution: `64 × 64`
* Filters: `128`
* Refines:

  * chamber boundaries,
  * myocardium thickness,
  * spatial consistency of cardiac structures.

---

### Decoder Level 2

* Resolution: `128 × 128`
* Filters: `64`
* Recovers:

  * finer anatomical details,
  * sharper transitions between tissues.

---

### Decoder Level 1

* Resolution: `256 × 256`
* Filters: `32`
* Produces:

  * pixel-accurate boundary localization,
  * high-resolution feature representations for final classification.

---

## Output Layer

* `1×1` convolution with **4 channels**
* `softmax` activation
* Output shape: `256 × 256 × 4`

Each channel corresponds to:

1. Background
2. Right ventricle (RV)
3. Myocardium (MYO)
4. Left ventricle (LV)

Each pixel is assigned a probability distribution over these **mutually exclusive classes**.

---

## Loss Function

Training uses a **hybrid loss** combining **Sparse Categorical Cross-Entropy** and **Multi-Class Dice Loss**, balancing pixel-level accuracy with region-level overlap.

### Sparse Categorical Cross-Entropy

For a pixel with ground-truth class label $$ y_i \in \{0, \dots, C-1\} $$ and predicted class probabilities $$ \hat{p}_{i,c} $$, the sparse categorical cross-entropy loss is defined as:

$$
\mathcal{L}_{\text{SCCE}}
= - \frac{1}{N} \sum_{i=1}^{N}
\log\\left( \hat{p}_{i,\,y_i} \right)
$$

This loss penalizes incorrect per-pixel class predictions and encourages well-calibrated probability estimates.

---

### Generalized Multi-Class Dice

For each foreground class ( c ), the Dice coefficient is defined as:

$$
\text{Dice}*c = \frac{2 \sum_i y*{i,c},\hat{y}*{i,c}}{\sum_i y*{i,c} + \sum_i \hat{y}_{i,c} + \epsilon}
$$

The multi-class Dice coefficient is computed by averaging Dice scores across all **foreground classes** (background excluded):

$$
\text{Dice}*{\text{multi}} = \frac{1}{C-1} \sum*{c=1}^{C-1} \text{Dice}_c
$$

The corresponding Dice loss is:

$$
\mathcal{L}*{\text{Dice}} = 1 - \text{Dice}*{\text{multi}}
$$

---

### Final Training Objective

The total loss optimized during training is:

$$
\mathcal{L}*{\text{total}} = \mathcal{L}*{\text{SCCE}} + \mathcal{L}_{\text{Dice}}
$$

This formulation ensures robust optimization in the presence of class imbalance and anatomical variability.

---

# **Performance Evaluation (Phase 2)**

Model performance in Phase 2 is evaluated using a combination of **quantitative metrics** and **qualitative visual inspection**, extending the evaluation strategy used in Phase 1 to the multi-class setting.

---

## **Multi-Class Dice Coefficient (Primary Metric)**

The primary quantitative metric for Phase 2 is the **multi-class Dice coefficient**, which measures region-level overlap between predicted and ground-truth segmentations across cardiac structures.

For each foreground class \( c \in \{\text{RV}, \text{MYO}, \text{LV}\} \), the Dice coefficient is defined as:

$$
\mathrm{Dice}_c = \frac{2 |P_c \cap G_c|}{|P_c| + |G_c|}
$$

The overall multi-class Dice score is computed by averaging Dice values across all foreground classes:

$$
\mathrm{Dice}_{\text{multi}} = \frac{1}{3} \sum_{c=1}^{3} \mathrm{Dice}_c
$$

This metric emphasizes:
* Anatomical overlap quality
* Shape consistency across regions
* Robustness to class imbalance

Background pixels are excluded from the Dice computation.

---

## **Pixel-wise Accuracy (Secondary Metric)**

Pixel-wise accuracy is reported as a complementary metric:

$$
\mathrm{Accuracy} = \frac{\text{Number of correctly classified pixels}}{\text{Total number of pixels}}
$$

While accuracy provides a general sense of classification performance, it is not used as the primary metric due to class imbalance between background and cardiac structures.

---

## **Qualitative Evaluation (Visual Inspection)**

Qualitative evaluation is performed using an interactive **slice-based visualization tool**, which allows inspection of:

* Raw MRI slices
* Ground-truth segmentation masks
* Predicted segmentation outputs
* Per-class overlay comparisons

This inspection enables identification of:
* Boundary inaccuracies between adjacent structures (e.g., LV vs MYO)
* Under- or over-segmentation of thin myocardial regions
* Failure cases at the apex and base of the heart
* Slice-to-slice spatial consistency

---

## **Full-Inference Evaluation on Volumes**

Although the Phase 2 model is trained on individual 2D slices, inference is performed **slice-by-slice on full 3D cardiac MRI volumes**.

Evaluation includes:
* Per-slice Dice scores for each class
* Per-volume aggregated Dice metrics
* Class-wise false positive and false negative analysis
* Visual overlays across the full cardiac cycle

This evaluation validates the effectiveness of slice-based training for volumetric segmentation and provides anatomical context for model predictions.

---

## **Comparison with Phase 1**

Compared to Phase 1, Phase 2 evaluation:

* Extends Dice analysis from binary LV segmentation to multi-class cardiac structures
* Introduces class-wise performance assessment
* Enables direct qualitative comparison between multiple anatomical regions
* Preserves a consistent evaluation methodology across phases

This ensures that performance improvements are attributable to architectural and training enhancements rather than changes in evaluation protocol.

--- 

# Future Work

Potential extensions include exploring **Focal Loss** and **Tversky-based losses** to further address class imbalance, particularly for myocardium segmentation. Architecturally, attention mechanisms or residual connections could improve boundary refinement. Beyond manual experimentation, the modular design of CardioSegNet naturally supports integration into an **AutoML framework**, enabling automated exploration of architectural depth, loss composition, and optimization strategies based on performance feedback.

---

# **Project Structure**

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




