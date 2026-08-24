# CS776-Kerbeus

## Detecting And Preventing Modality Collapse in Multimodal — Kerbeus

A multimodal deep learning pipeline for skin lesion diagnosis using the **Derm7pt** dataset. The model — **Kerbeus** — fuses dual-image (dermoscopy + clinical) features with structured tabular metadata through cross-modal attention, CLIP-style alignment, and a learnable reliability gate that dynamically re-weights modalities and prevents modality collapse, ensuring results don't rely on a single modality.

**Course:** CS776 (Deep Learning for Computer Vision), IIT Kanpur

**Group:** 9

---

## Table of Contents

* [Quick Results](https://www.google.com/search?q=%23quick-results)
* [Overview](https://www.google.com/search?q=%23overview)
* [Dataset](https://www.google.com/search?q=%23dataset)
* [Architecture](https://www.google.com/search?q=%23architecture)
* [Training Strategy](https://www.google.com/search?q=%23training-strategy)
* [Ablation Study](https://www.google.com/search?q=%23ablation-study)
* [Project Structure](https://www.google.com/search?q=%23project-structure)
* [Requirements](https://www.google.com/search?q=%23requirements)
* [Usage](https://www.google.com/search?q=%23usage)
* [Outputs & Checkpoints](https://www.google.com/search?q=%23outputs--checkpoints)

---

## Quick Results

**Kerbeus vs. Baseline** on Derm7pt Test Set:

| Metric | Baseline | Kerbeus | Improvement |
| --- | --- | --- | --- |
| **Accuracy** | 67.85% | **83.04%** | **+15.19%** |
| **Macro-F1** | 0.4757 | **0.7311** | **+0.2554** |
| **Weighted-F1** | 0.6653 | **0.8253** | **+0.1600** |
| **Modality Balance (Image/Tabular)** | 83% / 17% | **61% / 39%** | **Balanced** |

**Key insight:** The baseline suffers from severe **modality collapse** — relying almost entirely on image features while ignoring tabular metadata. Kerbeus solves this through adaptive fusion and gradient balancing, achieving both higher accuracy and balanced multimodal contribution.

See the full report in the `docs/` folder for detailed class-wise performance, ablation studies, and robustness analysis.

---

## Overview

Kerbeus addresses two core failure modes common in multimodal medical imaging models:

1. **Modality collapse** — the model ignores tabular features and relies almost entirely on image features.


2. **Fragility under degradation** — performance degrades sharply when one modality is noisy or missing at inference time.



It does so by combining cross-modal attention fusion, a contrastive CLIP alignment head, and a curriculum-trained **reliability gate** that learns to down-weight unreliable modalities at inference time.

---

## Dataset

**Derm7pt** (7-Point Checklist Dermoscopy Dataset)

| Split | Images |
| --- | --- |
| Train | 376 |
| Validation | 181 |
| Test | 355 |

### Target Classes (merged)

| Label | Original Diagnoses |
| --- | --- |
| `MEL` | Melanoma (all subtypes + metastasis) |
| `NEV` | Clark / Combined / Congenital / Dermal / Recurrent / Reed / Blue Nevus |
| `BCC` | Basal Cell Carcinoma |
| `SK` | Seborrheic Keratosis |
| `MISC` | Dermatofibroma, Lentigo, Melanosis, Vascular Lesion, Miscellaneous |

Note: The table above reflects the merged target classes used for the project.

### Tabular Features

* **Categorical (11):** `vascular_structures`, `blue_whitish_veil`, `pigment_network`, `management`, `streaks`, `dots_and_globules`, `elevation`, `regression_structures`, `pigmentation`, `level_of_diagnostic_difficulty`, `location`.


* **Numerical (1):** `seven_point_score`.



---

## Architecture

See **`docs/report.pdf`** (Figure 2) for the full architecture diagram with all component connections.

* **`InceptionBase`**: Shared InceptionV3 feature extractor (aux logits disabled). Two instances — one for dermoscopy, one for clinical images.


* **`DiagnosisMultimodalNet`**: Dual-backbone image branch.


* **`FTTransformerEncoder`**: Feature Tokenisation Transformer for tabular metadata.


* **`CrossModalAttentionFusion`**: Projects image and tabular features to `D_MODEL=256`, performs 2-token self-attention, and concatenates them.


* **`TripleCLIPHead`**: Contrastive alignment that pulls image and tabular representations together.


* **`ReliabilityHead`**: Classifies each modality as clean / ID-perturbed / OOD-perturbed and outputs soft gate weights.



---

## Training Strategy

Kerbeus is trained in two stages (**Stage A** and **Stage B**), totalling four phases.

### Stage A — Clean Training (Phases 1–3)

* **Phase 1 (Epochs 1-5):** Fusion + FT-Transformer only (Image backbone frozen).


* **Phase 2 (Epochs 6-30):** Full model (All components unfrozen).


* **Phase 3 (Epochs 31-50):** Full model (continued) with Early stopping patience = 7.



### Stage B — Curriculum Training for Reliability head (Phase 4)

The reliability head is introduced and trained using the `PerturbedCurriculumDataset`.

* **Clean:** No perturbations.


* **ID:** Gaussian blur σ ∈ [0.05, 0.20], Gaussian noise, random occlusion for images; Random feature masking 10–40% for tabular data.


* **OOD:** Strong blur σ=7, noise σ=0.40, occlusion 30% for images; Full mask (100%), feature shuffle for tabular data.



---

## Ablation Study

Component-wise ablation shows removing individual components consistently degrades performance, highlighting that each mechanism contributes meaningfully. The **most critical** is the cross-attention fusion combined with gradient balancing.

| Model Variant | Macro-F1 |
| --- | --- |
| Kerbeus (Full Model) | **0.7311** |
| No Cross-Attention | 0.6827 |
| No CLIP Alignment | 0.6586 |
| CLIP Only | 0.5041 |
| No Modality Dropout | 0.6471 |
| No Fragility Sampling | 0.6353 |
| No Gradient Balancing | 0.6797 |

---

## Project Structure

```text
.
├── docs/
│   └── report.pdf
├── notebooks/
│   └── kerbeus.ipynb
├── src/
│   ├── app.py
│   ├── tabular_preprocessor.pkl
│   ├── model_full.pt
│   ├── model.pt
│   ├── meta.csv
│   ├── test_indexes.csv
│   └── test_images/
│       ├── derm/
│       └── clinical/
├── requirements.txt
└── README.md

```

* `notebooks/kerbeus.ipynb`: Contains the original model definition, data pipelines, training loops, and curriculum evaluation.


* `docs/report.pdf`: The detailed academic report.


* `src/app.py`: A Streamlit web application providing a user interface for multimodal inference, OOD testing (blur/masking), and GradCAM interpretability.

---

## Requirements

Ensure all dependencies listed in `requirements.txt` are installed.

**Core ML Dependencies:**

* `torch >= 2.0`, `torchvision`
* `numpy`, `pandas`, `Pillow`, `scikit-learn`, `tqdm`

**Web Application Dependencies:**

* `streamlit`, `joblib`
* *(Optional for visual explanations)* `grad-cam`, `opencv-python-headless`

---

## Usage

### 1. Interactive Application (Streamlit)

To run the interactive UI, ensure your model weights (`model_full.pt`, `model.pt`), preprocessors, and test images are placed inside the `src/` directory (or update the paths in `app.py`). Then, run:

```bash
streamlit run src/app.py

```

**Features included in the app:**

* **Modality Configuration:** Test the model using Full Modalities, or easily exclude Tabular Data or Images.
* **Ablation / OOD Testing:** Dynamically apply Gaussian Blur (σ) to images or mask tabular features (in %) to test model reliability.
* **Interpretability:** If `pytorch_grad_cam` is installed, you can generate feature explanations (heatmaps) for both dermoscopy and clinical images to see where the model focuses.

### 2. Training Pipeline (Jupyter Notebook)

To run the training pipeline, open `notebooks/kerbeus.ipynb` and execute the cells sequentially. Running the `main()` function in the final cell will trigger:

1. Baseline training and evaluation.


2. Kerbeus Stage A (clean) and Stage B (curriculum) training.


3. Inference ablations.


4. Generation of the final report.



---

## Outputs & Checkpoints

During training, the following checkpoints are generated:

* **`kerbeus_v5_best.pt`** — best model from Baseline++.


* **`kerbeus_best.pt`** — final Kerbeus model after all 4 phases. *(Note: In the app, this is referenced as `model.pt` and `model_full.pt`)*.
