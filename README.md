# Multimodal Biometric Authentication

> **Biometric Authentication and Correlation Analysis Based on CNN–CBAM Fusion Network for Multimodal Recognition**

A deep learning–powered biometric authentication system that fuses **iris** and **palm** images for secure, multi-factor identity verification. Built with a custom CNN–CBAM (Convolutional Block Attention Module) fusion architecture and served via an interactive Streamlit web application.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Admin Panel](#admin-panel)
- [Configuration](#configuration)
- [Model Training](#model-training)

---

## Overview

This system authenticates users by jointly analyzing two biometric modalities — **iris** and **palm** images — reducing the false acceptance rate compared to single-modality systems. The core pipeline:

1. Extracts deep features from both images using a shared MobileNetV2 backbone.
2. Applies an Inception-Lite block for multi-scale feature capture.
3. Refines features with a CBAM attention module (channel + spatial attention).
4. Fuses both modality embeddings and compares against a registered user database using cosine similarity.

---

## Architecture

```
Input (Iris + Palm)
       │
       ▼
 MobileNetV2 Backbone  ← shared feature extractor
       │
       ▼
 InceptionLite Block   ← multi-scale convolutions (1×1, 3×3, 5×5, pool)
       │
       ▼
 CBAM Attention        ← channel attention + spatial attention
       │
       ▼
 Global Average Pool
       │
       ▼
 Fusion FC Layer       ← concatenate iris + palm → 256-d embedding
       │
       ▼
 L2 Normalization
       │
       ▼
 Cosine Similarity ──► Match / No Match
```

### Key Components

| Component | Description |
|---|---|
| `MobileNetV2` | Lightweight CNN backbone for feature extraction |
| `InceptionLite` | Multi-scale feature fusion (1×1, 3×3, 5×5, pooling branches) |
| `CBAM` | Channel + Spatial attention for focused feature refinement |
| `Fusion FC` | Linear layer that maps 512-d concatenated features to 256-d embedding |
| `Cosine Similarity` | Distance metric for identity matching at inference time |

---

## Project Structure

```
multimodal_biometric_authentication/
├── app.py                      # Streamlit web application
├── model.ipynb                 # Model training notebook
├── final_biometric_model.pth   # Pretrained model weights
├── database.pkl                # Registered user embeddings
└── README.md
```

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- Streamlit
- scikit-learn
- Pillow
- NumPy

---

## Installation

```bash
# Clone the repository
git clone https://github.com/rakesh-suru/multimodal_biometric_authentication.git
cd multimodal_biometric_authentication

# Install dependencies
pip install torch torchvision streamlit scikit-learn pillow numpy

# Run the application
streamlit run app.py
```

---

## Usage

### User Registration

1. Navigate to **Register** in the sidebar.
2. Enter a unique username.
3. Upload **3–10 iris images** and **3–10 palm images**.
4. Click **Register** — the system averages embeddings across all uploaded images for a robust profile.

> Images must be valid image files (JPEG, PNG, etc.) under **5 MB** each.

### User Login

1. Navigate to **Login** in the sidebar.
2. Upload one **iris image** and one **palm image**.
3. Click **Login** — the system computes your embedding and matches it against the database.
4. A cosine similarity score above the configured threshold grants access.

---

## How It Works

### Registration
- Multiple iris and palm image pairs are passed through the model.
- Embeddings are averaged and L2-normalized to form a stable user profile.
- A duplicate check is performed before saving — if the new embedding is too similar to an existing user (above threshold), registration is rejected.

### Authentication
- A single iris + palm pair is embedded at inference time.
- Cosine similarity is computed against all registered embeddings.
- The highest scoring match is selected; if it exceeds the threshold, the user is authenticated.

### Similarity Threshold
- Default threshold: **0.95**
- Adjustable via the Admin Panel (persisted to `threshold.pkl`).

---

## Admin Panel

Access via **Admin** in the sidebar. Default credentials:

| Field | Value |
|---|---|
| Username | `admin` |
| Password | `1234` |

Admin capabilities:
- View all registered users
- Delete individual users from the database
- Adjust the cosine similarity threshold via a slider (0.0 – 1.0)

---

## Configuration

Key settings in `app.py`:

```python
MODEL_PATH      = "final_biometric_model.pth"   # Model weights
DB_PATH         = "database.pkl"                 # User database
THRESHOLD_PATH  = "threshold.pkl"                # Persisted threshold
ADMIN_USERNAME  = "admin"
ADMIN_PASSWORD  = "1234"
MAX_FILE_SIZE_MB = 5                             # Max upload size per image
MIN_IMAGES      = 3                              # Min images per modality at registration
MAX_IMAGES      = 10                             # Max images per modality at registration
```

---

## Model Training

The full training pipeline is provided in `model.ipynb`, covering:

- Dataset loading and preprocessing
- MobileNetV2 + InceptionLite + CBAM model definition
- Contrastive / metric learning training loop
- Threshold calibration
- Model export to `final_biometric_model.pth`

Open the notebook in Jupyter or Google Colab to retrain or fine-tune the model on your own biometric dataset.
