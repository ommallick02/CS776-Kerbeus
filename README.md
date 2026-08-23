# CS776-Kerbeus

## Detecting And Preventing Modality Collapse in Multimodal — Kerbeus

A multimodal deep learning pipeline for skin lesion diagnosis using the **Derm7pt** dataset. The model — **Kerbeus** — fuses dual-image (dermoscopy + clinical) features with structured tabular metadata through cross-modal attention, CLIP-style alignment, and a learnable reliability gate that dynamically re-weights modalities and prevents modality collapse, ensuring results don't rely on a single modality.

**Course:** CS776 (Deep Learning for Computer Vision), IIT Kanpur  
**Group:** 9

---

## Table of Contents

- [Quick Results](#quick-results)
- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Training Strategy](#training-strategy)
- [Ablation Study](#ablation-study)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Configuration](#configuration)
- [Outputs & Checkpoints](#outputs--checkpoints)
- [Evaluation & Reporting](#evaluation--reporting)

---

## Quick Results

**Kerbeus vs. Baseline** on Derm7pt Test Set:

| Metric | Baseline | Kerbeus | Improvement |
|--------|----------|---------|-------------|
| **Accuracy** | 67.85% | **83.04%** | **+15.19%** |
| **Macro-F1** | 0.4757 | **0.7311** | **+0.2554** |
| **Weighted-F1** | 0.6653 | **0.8253** | **+0.1600** |
| **Modality Balance (Image/Tabular)** | 83% / 17% | **61% / 39%** | **Balanced** |

**Key insight:** The baseline suffers from severe **modality collapse** — relying almost entirely on image features while ignoring tabular metadata. Kerbeus solves this through adaptive fusion and gradient balancing, achieving both higher accuracy and balanced multimodal contribution.

See the full report for detailed class-wise performance, ablation studies, and robustness analysis.

---

## Overview

Kerbeus addresses two core failure modes common in multimodal medical imaging models:

1. **Modality collapse** — the model ignores tabular features and relies almost entirely on image features.
2. **Fragility under degradation** — performance degrades sharply when one modality is noisy or missing at inference time.

It does so by combining cross-modal attention fusion, a contrastive CLIP alignment head, and a curriculum-trained **reliability gate** that learns to down-weight unreliable modalities at inference time.

---

## Dataset

**Derm7pt** (7-Point Checklist Dermoscopy Dataset)

| Split      | Images       |
|------------|--------------|
| Train      |     376      |
| Validation |     181      |
| Test       |     355      |

**Expected directory layout:**
```
release_v0/
├── images/
├── meta/
│   ├── meta.csv
│   ├── train_indexes.csv
│   ├── valid_indexes.csv
│   └── test_indexes.csv
```

> On Kaggle: `/kaggle/input/datasets/menakamohanakumar/derm7pt/release_v0`

### Target Classes (merged)

| Label | Original Diagnoses |
|-------|--------------------|
| `MEL`  | Melanoma (all subtypes + metastasis) |
| `NEV`  | Clark / Combined / Congenital / Dermal / Recurrent / Reed / Blue Nevus |
| `BCC`  | Basal Cell Carcinoma |
| `SK`   | Seborrheic Keratosis |
| `MISC` | Dermatofibroma, Lentigo, Melanosis, Vascular Lesion, Miscellaneous |

### Tabular Features

- **Categorical (11):** `vascular_structures`, `blue_whitish_veil`, `pigment_network`, `management`, `streaks`, `dots_and_globules`, `elevation`, `regression_structures`, `pigmentation`, `level_of_diagnostic_difficulty`, `location`
- **Numerical (1):** `seven_point_score`

---

## Architecture

### Kerbeus Model

See **`report.pdf`** (Figure 2) for the full architecture diagram with all component connections.

### Components

| Component | Description |
|-----------|-------------|
| `InceptionBase` | Shared InceptionV3 feature extractor (aux logits disabled). Two instances — one for dermoscopy, one for clinical images. |
| `DiagnosisMultimodalNet` | Dual-backbone image branch. Produces per-image logits (`out_d`, `out_c`), a combined image feature (`img_feat`), and per-stream GAP vectors. |
| `FTTransformerEncoder` | Feature Tokenisation Transformer. Each tabular feature becomes a learned token; positional biases and a TransformerEncoder with `norm_first=True` produce `tab_feat`. |
| `CrossModalAttentionFusion` | Projects image and tabular features to `D_MODEL=256`, performs 2-token self-attention, and concatenates to form the fused representation. |
| `TripleCLIPHead` | Contrastive alignment that pulls image and tabular representations together in embedding space. |
| `ReliabilityHead` | Trained in Phase 4. Classifies each modality as clean / ID-perturbed / OOD-perturbed and outputs soft gate weights `gate_w_img`, `gate_w_tab`. |

---

## Training Strategy

Kerbeus is trained in two stages (**Stage A** and **Stage B**), totalling four phases.

### Stage A — Clean Training (Phases 1–3)

| Phase | Epochs | What is trained | Notes |
|-------|--------|-----------------|-------|
| 1 | 1–5 | Fusion + FT-Transformer only | Image backbone frozen |
| 2 | 6–30 | Full model | All components unfrozen |
| 3 | 31–50 | Full model (continued) | Early stopping patience = 7 |

- **Loss:** `AsymmetricLoss` (γ⁻=2, γ⁺=1) to handle class imbalance
- **Optimiser:** AdamW with per-component learning rates
- **Scheduler:** CosineAnnealingLR
- **Sampler:** `WeightedRandomSampler` for per-class balance
- **Mixed precision:** PyTorch AMP (`GradScaler`)
- **Checkpoint:** `kerbeus_v5_best.pt`

### Stage B — Curriculum Training for Reliability head (Phase 4)

The reliability head is introduced. Training data is served through `PerturbedCurriculumDataset`, which draws samples from three distributions:

| Perturbation Type | Image | Tabular |
|-------------------|-------|---------|
| **Clean** | — | — |
| **ID** | Gaussian blur σ ∈ [0.05, 0.20], Gaussian noise, random occlusion | Random feature masking 10–40% |
| **OOD** | Strong blur σ=7, noise σ=0.40, occlusion 30% | Full mask (100%), feature shuffle |

- **Reliability loss** guides the gate head (`λ_rel = 0.30`)
- A **gradient balancer** (`FRAG_ALPHA=0.3`) prevents image-gradient dominance
- **Checkpoint:** `kerbeus_best.pt`

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 299 × 299 |
| Batch size | 8 |
| LR (image backbone) | 1e-4 |
| LR (tabular) | 1e-3 |
| LR (fusion) | 3e-4 |
| LR (attention) | 5e-5 |
| Weight decay | 1e-4 |
| FT-Transformer embedding dim | 16 |
| FT-Transformer hidden dim | 128 |
| FT-Transformer heads / layers | 4 / 3 |
| Cross-attention D_MODEL | 256 |
| Cross-attention heads | 8 |
| CLIP embedding dim | 512 |
| CLIP loss weight (λ) | 0.05 |
| Seed | 7 |

---

## Ablation Study

Kerbeus is designed with multiple mechanisms to prevent modality collapse. Component-wise ablation (removing each mechanism individually) shows:

| Model Variant | Macro-F1 |
|---------------|----------|
| Kerbeus (Full Model) | **0.7311** |
| No Cross-Attention | 0.6827 |
| No CLIP Alignment | 0.6586 |
| CLIP Only | 0.5041 |
| No Modality Dropout | 0.6471 |
| No Fragility Sampling | 0.6353 |
| No Gradient Balancing | 0.6797 |

**Finding:** Removing individual components consistently degrades performance, highlighting that each mechanism contributes meaningfully. The **most critical** is the cross-attention fusion combined with gradient balancing.

---

## Project Structure

```
kerbeus.ipynb
│
├── Cell 1   — Imports & Setup
├── Cell 2   — Global Config (CFG, CURR, DEMO)
├── Cell 3   — Data Pipeline
│              TabularPreprocessor · prepare_data · get_transforms
│              Derm7ptDataset · make_class_sampler
├── Cell 4   — Loss & Metrics
│              AsymmetricLoss · compute_metrics · print_metrics
├── Cell 5   — Image Backbone
│              InceptionBase · DiagnosisMultimodalNet
├── Cell 6   — Shared Encoders
│              FTTransformerEncoder · CrossModalAttentionFusion
│              TripleCLIPHead · ReliabilityHead
├── Cell 7   — Ablation Model Variants
│              BaselineFusionModel · ConcatMLPFusion
│              CrossAttnOnlyFusion · CLIPFusion
├── Cell 8   — Kerbeus Full Model
├── Cell 9   — Curriculum Dataset
│              PerturbedCurriculumDataset · AugmentedTestDataset
├── Cell 10  — Training Utilities
│              evaluate · compute_logit_contributions
│              print_modality_balance · evaluate_with_ablation
├── Cell 11  — Baseline Training Loop
├── Cell 12  — Kerbeus Phases 1–3 Training Loop
├── Cell 13  — Kerbeus Phase 4 (Curriculum) Training Loop
├── Cell 14  — Reporting
│              print_ablation_report · print_final_report
├── Cell 15  — Per-Sample Reliability Gate Demo
└── Cell 16  — main() — full pipeline orchestration
```

---

## Requirements

```
torch >= 2.0
torchvision
numpy
pandas
Pillow
scikit-learn
tqdm
```

> Designed to run on **Kaggle** with GPU (tested on T4). `CFG.DEVICE` auto-detects CUDA.

---

## Usage

### Full Pipeline

Run all cells sequentially, then execute `main()` in Cell 16:

```python
main()
```

`main()` runs the following steps in order:

1. **Baseline** — trains and evaluates `BaselineFusionModel`
2. **Kerbeus** — Stage A (3-phase clean) → Stage B (Phase 4 curriculum)
3. **Inference Ablations** — Image Only / Tabular Only on the trained Kerbeus checkpoint
4. **Final Report** — performance comparison, gradient dominance, logit contribution, robustness delta
5. **Demo** — per-sample reliability gate response to progressive perturbation

### Reliability Gate Demo Only

```python
per_sample_reliability_demo()
```

Requires `kerbeus_best.pt` to exist. Iterates over test samples (one per class), applies increasing image blur and tabular masking, and prints how `gate_w_img` / `gate_w_tab` respond.

---

## Configuration

All global settings live in three dataclasses:

- **`CFG`** — paths, model dimensions, training schedule, learning rates, checkpoint names
- **`CURR`** — Phase 4 / curriculum-specific settings (perturbation ranges, loss weights)
- **`DEMO`** — demo sampling parameters (samples per class, blur sigmas, mask fractions)

To adapt to a local environment, update `CFG.BASE`, `CFG.IMG_DIR`, and `CFG.CKPT_DIR`.

---

## Outputs & Checkpoints

During training, the following checkpoints are saved:

- **`kerbeus_v5_best.pt`** — best model from Baseline++ (enhanced tabular encoder)
- **`kerbeus_best.pt`** — final Kerbeus model after all 4 phases

Test-phase predictions and metrics are logged and displayed in the final report.

---

## Evaluation & Reporting

The model is evaluated using:

- **Accuracy, Precision, Recall** — per-class and macro-averaged
- **Macro-F1 & Weighted-F1** — overall fairness metrics
- **Modality contribution %** — image vs. tabular logit contributions
- **Gradient analysis** — gradient norm ratios to diagnose collapse
- **Robustness under perturbation** — in-distribution (ID) and out-of-distribution (OOD) degradation
- **Reliability gate behavior** — how weights shift under modal degradation

See **`report.pdf`** for full tables, figures, and detailed analysis.

---

## References

This work addresses multimodal collapse in medical imaging by building on:
- Cross-modal attention fusion
- Contrastive representation alignment (CLIP-style)
- Curriculum learning for robustness
- Gradient-aware training

For more details, see the course report: **`report.pdf`**