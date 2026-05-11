import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pickle
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms

# =====================================================
# SETTINGS
# =====================================================

MODEL_PATH = "final_biometric_model.pth"
DB_PATH = "database.pkl"
THRESHOLD_PATH = "threshold.pkl"

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "1234"

device = torch.device("cpu")

# 🔥 ADDED
MAX_FILE_SIZE_MB = 5
MIN_IMAGES = 3
MAX_IMAGES = 10

# =====================================================
# SESSION STATE
# =====================================================

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if "admin_message" not in st.session_state:
    st.session_state.admin_message = None

# =====================================================
# LOAD THRESHOLD
# =====================================================

if os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH, "rb") as f:
        THRESHOLD = pickle.load(f)
else:
    THRESHOLD = 0.95

# =====================================================
# MODEL ARCHITECTURE (UNCHANGED)
# =====================================================

backbone = models.mobilenet_v2(weights=None)
feature_extractor = backbone.features

class InceptionLite(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch3 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, 64, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch3(x),
            self.branch5(x),
            self.branch_pool(x)
        ], dim=1)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        maxv = self.fc(self.max_pool(x))
        return self.sigmoid(avg + maxv)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, maxv], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# =====================================================
# MODEL BLOCKS
# =====================================================

inception_block = InceptionLite(1280)
cbam = CBAM(256)
gap = nn.AdaptiveAvgPool2d((1,1))
fusion_fc = nn.Linear(512, 256)

# =====================================================
# LOAD WEIGHTS
# =====================================================

checkpoint = torch.load(MODEL_PATH, map_location=device)

feature_extractor.load_state_dict(checkpoint["mobilenet"])
inception_block.load_state_dict(checkpoint["inception"])
cbam.load_state_dict(checkpoint["cbam"])
fusion_fc.load_state_dict(checkpoint["fusion_fc"])

feature_extractor.eval()
inception_block.eval()
cbam.eval()
fusion_fc.eval()

# =====================================================
# TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# =====================================================
# LOAD DATABASE
# =====================================================

if os.path.exists(DB_PATH):
    with open(DB_PATH,"rb") as f:
        database = pickle.load(f)
else:
    database = {}

# =====================================================
# 🔥 ADDED HELPERS
# =====================================================

def is_valid_image(file):
    try:
        Image.open(file)
        return True
    except:
        return False

def is_valid_size(file):
    return file.size / (1024*1024) <= MAX_FILE_SIZE_MB

# =====================================================
# EMBEDDING FUNCTION (UNCHANGED)
# =====================================================

def get_embedding(iris_img, palm_img):
    iris = transform(iris_img).unsqueeze(0)
    palm = transform(palm_img).unsqueeze(0)

    with torch.no_grad():
        iris_feat = feature_extractor(iris)
        palm_feat = feature_extractor(palm)

        iris = cbam(inception_block(iris_feat))
        palm = cbam(inception_block(palm_feat))

        iris = gap(iris).view(1,-1)
        palm = gap(palm).view(1,-1)

        fused = fusion_fc(torch.cat([iris, palm], dim=1))
        emb = torch.nn.functional.normalize(fused)

    return emb.cpu().numpy()

# =====================================================
# UI
# =====================================================

st.title("Multimodal Biometric Authentication")
menu = st.sidebar.selectbox("Menu", ["Register", "Login", "Admin"])

# =====================================================
# REGISTER
# =====================================================

if menu == "Register":

    st.header("User Registration")

    name = st.text_input("Enter Name")
    iris_files = st.file_uploader("Upload Iris Images", accept_multiple_files=True)
    palm_files = st.file_uploader("Upload Palm Images", accept_multiple_files=True)

    if st.button("Register"):

        # 🔥 Missing input
        if not name or not iris_files or not palm_files:
            st.error("Please upload both images")
            st.stop()

        # 🔥 Improved Image Count Validation

        iris_count = len(iris_files)
        palm_count = len(palm_files)

        error_msg = ""

        # Iris validation
        if iris_count < MIN_IMAGES:
            error_msg += f"Iris: Uploaded {iris_count}, add {MIN_IMAGES - iris_count} more.\n"
        elif iris_count > MAX_IMAGES:
            error_msg += f"Iris: Uploaded {iris_count}, remove {iris_count - MAX_IMAGES} images.\n"

        # Palm validation
        if palm_count < MIN_IMAGES:
            error_msg += f"Palm: Uploaded {palm_count}, add {MIN_IMAGES - palm_count} more.\n"
        elif palm_count > MAX_IMAGES:
            error_msg += f"Palm: Uploaded {palm_count}, remove {palm_count - MAX_IMAGES} images.\n"

        # If any error exists → show and stop
        if error_msg:
            st.error(error_msg)
            st.stop()

        embeddings = []

        for i, p in zip(iris_files, palm_files):

            # 🔥 File type check
            if not is_valid_image(i) or not is_valid_image(p):
                st.error("Invalid file format")
                st.stop()

            # 🔥 File size check
            if not is_valid_size(i) or not is_valid_size(p):
                st.error("File too large")
                st.stop()

            emb = get_embedding(
                Image.open(i).convert("RGB"),
                Image.open(p).convert("RGB")
            )
            embeddings.append(emb)

        emb = np.mean(np.vstack(embeddings), axis=0)
        emb = emb / np.linalg.norm(emb)

        # 🔥 DUPLICATE CHECK
        for user, db_emb in database.items():
            score = cosine_similarity(emb.reshape(1,-1), db_emb.reshape(1,-1))[0][0]
            if score > THRESHOLD:
                st.error("User already registered")
                st.stop()

        database[name] = emb

        with open(DB_PATH, "wb") as f:
            pickle.dump(database, f)

        st.success("User Registered")

# =====================================================
# LOGIN
# =====================================================

elif menu == "Login":

    st.header("User Login")

    iris_file = st.file_uploader("Upload Iris Image")
    palm_file = st.file_uploader("Upload Palm Image")

    if st.button("Login"):

        # 🔥 Missing input
        if iris_file is None or palm_file is None:
            st.error("Please upload both images")
            st.stop()

        emb = get_embedding(
            Image.open(iris_file).convert("RGB"),
            Image.open(palm_file).convert("RGB")
        )

        best_score = 0
        best_user = None

        for user, db_emb in database.items():
            db_emb = db_emb / np.linalg.norm(db_emb)

            score = cosine_similarity(
                emb.reshape(1,-1),
                db_emb.reshape(1,-1)
            )[0][0]

            if score > best_score:
                best_score = score
                best_user = user

        st.write("Score:", round(float(best_score),3))

        if best_score > THRESHOLD:
            st.success(f"Welcome {best_user}")
        else:
            st.error("Access Denied")

# =====================================================
# ADMIN PANEL (UNCHANGED)
# =====================================================

elif menu == "Admin":

    st.header("Admin Panel")

    if not st.session_state.admin_logged_in:

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")

    else:

        st.success("Admin Logged In")

        if st.session_state.admin_message:
            st.success(st.session_state.admin_message)
            st.session_state.admin_message = None

        if st.button("Logout"):
            st.session_state.admin_logged_in = False
            st.rerun()

        st.subheader("Registered Users")
        for user in database.keys():
            st.write("•", user)

        st.subheader("Update Threshold")

        new_threshold = st.slider("Threshold", 0.0, 1.0, float(THRESHOLD))

        if st.button("Update Threshold"):
            old_threshold = THRESHOLD

            with open(THRESHOLD_PATH, "wb") as f:
                pickle.dump(new_threshold, f)

            st.session_state.admin_message = f"Threshold updated from {round(old_threshold,3)} → {round(new_threshold,3)}"
            st.rerun()

        if len(database) > 0:
            st.subheader("Delete User")

            user_del = st.selectbox("Select User", list(database.keys()))

            if st.button("Delete User"):
                deleted_user = user_del

                del database[user_del]

                with open(DB_PATH, "wb") as f:
                    pickle.dump(database, f)

                st.session_state.admin_message = f"User '{deleted_user}' deleted successfully"
                st.rerun()