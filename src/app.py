"""
Kerbeus — Multimodal Skin Lesion Classifier
Streamlit application for the Derm7pt multimodal model.

Author  : Expert Python / Streamlit / PyTorch Developer
Dataset : Derm7pt  |  Classes: BCC · MEL · MISC · NEV · SK
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import os
import io
import glob
import pickle
import textwrap
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Third-party
# ─────────────────────────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import inception_v3, Inception_V3_Weights
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Capture exactly why GradCAM is failing to import
# ─────────────────────────────────────────────────────────────────────────────
try:
    import cv2
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRADCAM = True
    GRADCAM_ERR = ""
except Exception as e:
    HAS_GRADCAM = False
    GRADCAM_ERR = f"{type(e).__name__}: {str(e)}"

# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE CONFIG  (must be the very first Streamlit call) ──────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kerbeus",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# (Must be defined before the classes to avoid ReferenceErrors)
# ─────────────────────────────────────────────────────────────────────────────
DERM_DIR       = Path("./test_images/derm")
CLINIC_DIR     = Path("./test_images/clinical")
PREPROCESSOR   = Path("./tabular_preprocessor.pkl")
MODEL_FULL_PATH= Path("./model_full.pt")     
MODEL_OOD_PATH = Path("./model.pt")          
META_PATH      = Path("./meta.csv")          
TEST_IDX_PATH  = Path("./test_indexes.csv")  

CLASSES        = ["BCC", "MEL", "MISC", "NEV", "SK"]
IMG_SIZE       = 299
IMG_MEAN       = [0.485, 0.456, 0.406]
IMG_STD        = [0.229, 0.224, 0.225]

CAT_COLS = [
    "vascular_structures", "blue_whitish_veil", "pigment_network",
    "management", "streaks", "dots_and_globules", "elevation",
    "regression_structures", "pigmentation", "level_of_diagnostic_difficulty",
    "location",
]
NUM_COLS = ["seven_point_score"]

SHORT_COLS = {
    "vascular_structures": "vascular", "blue_whitish_veil": "blue veil",
    "pigment_network": "pigment", "management": "management",
    "streaks": "streaks", "dots_and_globules": "dots",
    "elevation": "elevation", "regression_structures": "regression",
    "pigmentation": "pigmentation", "level_of_diagnostic_difficulty": "difficulty",
    "location": "location", "seven_point_score": "score"
}

CLASS_COLORS = {
    "BCC":  "#00e5ff", "MEL":  "#ff4081", "NEV":  "#69ff47",
    "SK":   "#ffd740", "MISC": "#b388ff",
}

class CFG:
    R_DIM        = 512
    EMB_DIM      = 16
    FT_HIDDEN    = 128
    FT_HEADS     = 4
    FT_LAYERS    = 3
    FT_DROPOUT   = 0.15
    D_MODEL      = 256
    ATTN_HEADS   = 8
    ATTN_DROPOUT = 0.10
    CLIP_DIM     = 512
    DROP_PROB    = 0.15
    NUM_COLS     = NUM_COLS
    CAT_COLS     = CAT_COLS

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CLASSES (Moved to top to prevent unpickling / instantiation AttributeErrors)
# ─────────────────────────────────────────────────────────────────────────────
class TabularPreprocessor:
    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler        = StandardScaler()
        self.fill_values   : dict[str, object]       = {}
        self.cat_dims      : list[int]               = []
        self.tab_dim       : int                     = 0

    def transform(self, df: pd.DataFrame):
        cat_parts, num_parts = [], []
        for col in CFG.CAT_COLS:
            series = df[col].astype(str).fillna(str(self.fill_values[col]))
            le     = self.label_encoders[col]
            codes  = series.map(lambda x, le=le: le.transform([x])[0] if x in le.classes_ else 0).values.astype(np.int64)
            cat_parts.append(codes.reshape(-1, 1))
        for col in CFG.NUM_COLS:
            vals = df[col].fillna(self.fill_values[col]).values.reshape(-1, 1)
            num_parts.append(vals)
        cat_arr = np.hstack(cat_parts).astype(np.int64)
        num_arr = self.scaler.transform(np.hstack(num_parts).astype(np.float32))
        return cat_arr, num_arr

class InceptionBase(nn.Module):
    def __init__(self):
        super().__init__()
        base = inception_v3(weights=Inception_V3_Weights.DEFAULT); base.aux_logits = False
        self.features = nn.Sequential(
            base.Conv2d_1a_3x3, base.Conv2d_2a_3x3, base.Conv2d_2b_3x3, nn.MaxPool2d(3, stride=2),
            base.Conv2d_3b_1x1, base.Conv2d_4a_3x3, nn.MaxPool2d(3, stride=2),
            base.Mixed_5b, base.Mixed_5c, base.Mixed_5d, base.Mixed_6a, base.Mixed_6b,
            base.Mixed_6c, base.Mixed_6d, base.Mixed_6e, base.Mixed_7a, base.Mixed_7b, base.Mixed_7c,
        )
    def forward(self, x): return self.features(x)

class DiagnosisMultimodalNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        f_dim = 2048
        self.backbone_d, self.backbone_c = InceptionBase(), InceptionBase()
        self.L_d_conv, self.L_c_conv = nn.Conv2d(f_dim, num_classes, 1), nn.Conv2d(f_dim, num_classes, 1)
        self.bn_d, self.bn_c = nn.BatchNorm1d(f_dim), nn.BatchNorm1d(f_dim)
        self.combine_visual = nn.Sequential(nn.Linear(f_dim * 2, CFG.R_DIM), nn.ReLU(), nn.BatchNorm1d(CFG.R_DIM))
        self.L_dc_linear = nn.Linear(CFG.R_DIM, num_classes)
    def forward(self, x_d, x_c):
        feat_d, feat_c = self.backbone_d(x_d), self.backbone_c(x_c)
        out_d = F.adaptive_avg_pool2d(self.L_d_conv(feat_d), (1, 1)).view(x_d.size(0), -1)
        out_c = F.adaptive_avg_pool2d(self.L_c_conv(feat_c), (1, 1)).view(x_c.size(0), -1)
        gap_d = F.adaptive_avg_pool2d(feat_d, (1, 1)).view(x_d.size(0), -1)
        gap_c = F.adaptive_avg_pool2d(feat_c, (1, 1)).view(x_c.size(0), -1)
        img_feat = self.combine_visual(torch.cat([self.bn_d(gap_d), self.bn_c(gap_c)], 1))
        return out_d, out_c, self.L_dc_linear(img_feat), gap_d, gap_c, img_feat

class FTTransformerEncoder(nn.Module):
    def __init__(self, cat_dims: list[int], num_classes: int):
        super().__init__()
        self.cat_embeds = nn.ModuleList([nn.Embedding(dim + 1, CFG.EMB_DIM) for dim in cat_dims])
        self.num_projectors = nn.ModuleList([nn.Linear(1, CFG.EMB_DIM) for _ in range(len(CFG.NUM_COLS))])
        self.token_proj = nn.Linear(CFG.EMB_DIM, CFG.FT_HIDDEN)
        self.pos_bias = nn.Parameter(torch.zeros(1, len(cat_dims) + len(CFG.NUM_COLS), CFG.FT_HIDDEN))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CFG.FT_HIDDEN,
            nhead=CFG.FT_HEADS,
            dim_feedforward=CFG.FT_HIDDEN * 4,
            dropout=CFG.FT_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, CFG.FT_LAYERS)
        self.norm, self.classifier = nn.LayerNorm(CFG.FT_HIDDEN), nn.Sequential(nn.Linear(CFG.FT_HIDDEN, CFG.FT_HIDDEN // 2), nn.GELU(), nn.Dropout(CFG.FT_DROPOUT), nn.Linear(CFG.FT_HIDDEN // 2, num_classes))
        self.out_dim = CFG.FT_HIDDEN
    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor):
        tokens = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.cat_embeds)] + [proj(x_num[:, i:i+1]) for i, proj in enumerate(self.num_projectors)]
        x = self.token_proj(torch.stack(tokens, 1)) + self.pos_bias
        tab_feat = self.norm(self.transformer(x).mean(1))
        return self.classifier(tab_feat), tab_feat

class CrossModalAttentionFusion(nn.Module):
    def __init__(self, img_dim, tab_dim, num_classes):
        super().__init__()
        self.img_proj, self.tab_proj = nn.Sequential(nn.Linear(img_dim, CFG.D_MODEL), nn.LayerNorm(CFG.D_MODEL)), nn.Sequential(nn.Linear(tab_dim, CFG.D_MODEL), nn.LayerNorm(CFG.D_MODEL))
        self.attn = nn.MultiheadAttention(CFG.D_MODEL, CFG.ATTN_HEADS, dropout=CFG.ATTN_DROPOUT, batch_first=True)
        self.norm, self.dropout = nn.LayerNorm(CFG.D_MODEL), nn.Dropout(CFG.ATTN_DROPOUT)
        self.classifier = nn.Sequential(nn.Linear(CFG.D_MODEL * 2, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, num_classes))
        self.out_dim = CFG.D_MODEL * 2
    def forward(self, img_feat, tab_feat):
        B = img_feat.size(0)
        tokens = torch.cat([self.img_proj(img_feat).unsqueeze(1), self.tab_proj(tab_feat).unsqueeze(1)], 1)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        fused_feat = self.norm(tokens + self.dropout(attn_out)).view(B, -1)
        return self.classifier(fused_feat), fused_feat

class TripleCLIPHead(nn.Module):
    def __init__(self, img_dim, tab_dim, fused_dim):
        super().__init__()
        def _make(in_dim): return nn.Sequential(nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, CFG.CLIP_DIM))
        self.proj_img, self.proj_tab, self.proj_fused = _make(img_dim), _make(tab_dim), _make(fused_dim)
    def forward(self, img, tab, fused):
        z_i, z_t, z_f = F.normalize(self.proj_img(img), dim=-1), F.normalize(self.proj_tab(tab), dim=-1), F.normalize(self.proj_fused(fused), dim=-1)
        return (3.0 - (z_i*z_t).sum(1).mean() - (z_i*z_f).sum(1).mean() - (z_t*z_f).sum(1).mean()) / 3.0

class ReliabilityHead(nn.Module):
    def __init__(self, img_dim: int, tab_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim + tab_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 3),
        )
    def forward(self, img_feat, tab_feat):
        x = torch.cat([img_feat, tab_feat], dim=1)
        logits = self.net(x)
        p = F.softmax(logits, dim=1)
        w_img = (p[:, 0] + p[:, 2]).unsqueeze(1)
        w_tab = (p[:, 0] + p[:, 1]).unsqueeze(1)
        return logits, w_img, w_tab

class ReliabilityGatedModelV5(nn.Module):
    def __init__(self, cat_dims, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.net = DiagnosisMultimodalNet(num_classes)
        self.ft_encoder = FTTransformerEncoder(cat_dims, num_classes)
        tab_dim = self.ft_encoder.out_dim
        self.fusion = CrossModalAttentionFusion(img_dim=CFG.R_DIM, tab_dim=tab_dim, num_classes=num_classes)
        self.clip_head = TripleCLIPHead(img_dim=CFG.R_DIM, tab_dim=tab_dim, fused_dim=self.fusion.out_dim)
        self.reliability = ReliabilityHead(img_dim=CFG.R_DIM, tab_dim=tab_dim)

    def forward(self, derm, clinic, tab_cat, tab_num):
        out_d, out_c, out_dc_raw, _, _, img_feat = self.net(derm, clinic)
        out_tab_raw, tab_feat = self.ft_encoder(tab_cat, tab_num)
        pert_logits, w_img, w_tab = self.reliability(img_feat, tab_feat)
        img_feat_gated = img_feat * w_img
        tab_feat_gated = tab_feat * w_tab
        out_dc = out_dc_raw.clone()
        out_tab = out_tab_raw.clone()
        final_logits, fused_feat = self.fusion(img_feat_gated, tab_feat_gated)
        clip_loss = self.clip_head(img_feat, tab_feat, fused_feat)
        extras = dict(w_img=w_img, w_tab=w_tab, pert_logits=pert_logits)
        return out_d, out_c, out_dc, out_tab, final_logits, clip_loss, extras

class RepairedFusionModelV5(nn.Module):
    def __init__(self, cat_dims, num_classes):
        super().__init__()
        self.net, self.ft_encoder = DiagnosisMultimodalNet(num_classes), FTTransformerEncoder(cat_dims, num_classes)
        self.fusion = CrossModalAttentionFusion(CFG.R_DIM, self.ft_encoder.out_dim, num_classes)
        self.clip_head = TripleCLIPHead(CFG.R_DIM, self.ft_encoder.out_dim, self.fusion.out_dim)
    def forward(self, derm, clinic, tab_cat, tab_num):
        out_d, out_c, out_dc, _, _, img_f = self.net(derm, clinic)
        out_tab, tab_f = self.ft_encoder(tab_cat, tab_num)
        final, _ = self.fusion(img_f, tab_f)
        return out_d, out_c, out_dc, out_tab, final, None

class DermWrapper(nn.Module):
    def __init__(self, model, clinic_t, tab_cat_t, tab_num_t): super().__init__(); self.model, self.clinic_t, self.tab_cat_t, self.tab_num_t = model, clinic_t, tab_cat_t, tab_num_t
    def forward(self, derm_t):
        out = self.model(derm_t, self.clinic_t, self.tab_cat_t, self.tab_num_t)
        return out[4] if len(out) >= 5 else out[-1]

class ClinicWrapper(nn.Module):
    def __init__(self, model, derm_t, tab_cat_t, tab_num_t): super().__init__(); self.model, self.derm_t, self.tab_cat_t, self.tab_num_t = model, derm_t, tab_cat_t, tab_num_t
    def forward(self, clinic_t):
        out = self.model(self.derm_t, clinic_t, self.tab_cat_t, self.tab_num_t)
        return out[4] if len(out) >= 5 else out[-1]

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — futuristic dark theme with scan-line texture
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Google fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg-void:      #040810;
    --bg-panel:     #070d1a;
    --bg-card:      #0c1526;
    --border-glow:  #00e5ff33;
    --border-hard:  #00e5ff66;
    --accent-cyan:  #00e5ff;
    --accent-pink:  #ff4081;
    --accent-green: #69ff47;
    --text-primary: #e0f7ff;
    --text-muted:   #4a7a8a;
    --text-dim:     #1e3a4a;
    --font-display: 'Orbitron', monospace;
    --font-mono:    'Share Tech Mono', monospace;
    --font-body:    'Rajdhani', sans-serif;
}

/* ── App shell ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: var(--bg-void) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

/* scan-line overlay */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 9999;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,229,255,0.015) 2px,
        rgba(0,229,255,0.015) 4px
    );
}

/* ── Sidebar — no scroll ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060f1e 0%, #040810 100%) !important;
    border-right: 1px solid var(--border-glow) !important;
}
[data-testid="stSidebar"] > div:first-child {
    overflow: hidden !important;
    overflow-y: hidden !important;
}
/* Scoped: targeting specific elements to prevent icon font breakage */
[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] label { 
    font-family: var(--font-body) !important; 
}

/* ── Main content padding ── */
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1400px; }

/* ── Typography overrides ── */
h1, h2, h3 {
    font-family: var(--font-display) !important;
    color: var(--accent-cyan) !important;
    letter-spacing: 0.12em;
    text-shadow: 0 0 18px rgba(0,229,255,0.5);
}
label, .stMarkdown p, .stText { font-family: var(--font-body) !important; font-size: 1rem; }

/* ── Inputs & selects ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stMultiselect > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-hard) !important;
    border-radius: 4px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.875rem !important;
}

/* ── RUN button — true center via wrapper class ── */
.run-btn-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2rem 0;
    width: 100%;
}
.run-btn-wrapper [data-testid="stButton"] button,
div[data-testid="stButton"] > button {
    min-width: 320px !important;
    border-color: var(--accent-pink) !important;
    color: var(--accent-pink) !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.25em !important;
    padding: 0.9rem 2rem !important;
    background: rgba(255,64,129,0.07) !important;
}
div[data-testid="stButton"] {
    margin-top: 2.5rem !important;  /* ← add this line, tune the value */
}
div[data-testid="stButton"] > button:hover {
    background: rgba(255,64,129,0.18) !important;
    box-shadow: 0 0 28px rgba(255,64,129,0.55) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 6px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"]  { font-family: var(--font-mono) !important; color: var(--text-muted) !important; }
[data-testid="stMetricValue"]  { font-family: var(--font-display) !important; color: var(--accent-cyan) !important; font-size: 1.6rem !important; }

/* ── Utility: glow panel ── */
.glow-panel {
    background: var(--bg-card);
    border: 1px solid var(--border-glow);
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: inset 0 0 30px rgba(0,229,255,0.03),
                0 4px 24px rgba(0,0,0,0.6);
}
.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    color: #00e5ff99;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.prob-bar-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: #a0c4cc;
}
.result-winner {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem;
    font-weight: 900;
    color: #00e5ff;
    text-shadow: 0 0 30px rgba(0,229,255,0.7), 0 0 60px rgba(0,229,255,0.3);
    letter-spacing: 0.18em;
    text-align: center;
    padding: 0.5rem 0;
}
.conf-badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1rem;
    background: rgba(0,229,255,0.1);
    border: 1px solid rgba(0,229,255,0.4);
    border-radius: 20px;
    padding: 0.2rem 1rem;
    color: #00e5ff;
    margin-top: 0.3rem;
}

/* Custom header sizes */
.ablation-header {
    font-family: 'Orbitron', sans-serif;
    color: white;
    font-size: 2.2rem;
    font-weight: 600;
    text-align: left;
}

/* Auto-run indicator badge */
.autorun-badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    background: rgba(105,255,71,0.1);
    border: 1px solid rgba(105,255,71,0.4);
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    color: #69ff47;
    letter-spacing: 0.12em;
    vertical-align: middle;
    margin-left: 0.5rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT / OPTION MAPS for categorical inputs
# ─────────────────────────────────────────────────────────────────────────────
CAT_OPTIONS: dict[str, list] = {
    "vascular_structures":            ["absent", "arborizing", "comma", "dotted",
                                        "glomerular", "hairpin", "irregular",
                                        "within_regression", "wreath"],
    "blue_whitish_veil":              ["absent", "present"],
    "pigment_network":                ["absent", "atypical", "typical"],
    "management":                     ["clinic_followup", "excision", "no_action"],
    "streaks":                        ["absent", "irregular", "regular"],
    "dots_and_globules":              ["absent", "irregular", "regular"],
    "elevation":                      ["elevated", "flat", "mixed"],
    "regression_structures":          ["absent", "blue_areas", "combinations",
                                        "scarlike_depigmentation", "white_areas"],
    "pigmentation":                   ["absent", "diffuse_irregular", "diffuse_regular",
                                        "localised_irregular", "localised_regular"],
    "level_of_diagnostic_difficulty": ["low", "medium", "high"],
    "location":                       ["head_neck", "lower_extremity", "oral_genital",
                                        "palms_soles", "trunk", "upper_extremity"],
}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT — must happen before any widget reads
# ─────────────────────────────────────────────────────────────────────────────
_ss_defaults = {
    "has_run":         False,   
    "_prev_blur":      0.0,
    "_prev_mask":      0,
    "_prev_gradcam":   False,
    "_prev_modality":  "Full Modalities",
}
for _k, _v in _ss_defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — file & data discovery
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

@st.cache_data(show_spinner=False)
def load_meta(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data(show_spinner=False)
def load_test_indexes(path: Path) -> Optional[list]:
    if path.exists():
        return pd.read_csv(path).values.flatten().tolist()
    return None

def get_image_path(base_dir: Path, relative_path: str) -> Optional[Path]:
    if pd.isna(relative_path): return None
    exact_path = base_dir / relative_path
    if exact_path.exists(): return exact_path
    flattened_path = base_dir / Path(relative_path).name
    if flattened_path.exists(): return flattened_path
    return exact_path

def get_cat_index(col_name: str, row_val) -> int:
    if pd.isna(row_val): return 0
    val = str(row_val).strip()
    opts = CAT_OPTIONS.get(col_name, [])
    if val in opts: return opts.index(val)
    val_lower = val.lower().replace(" ", "_")
    for i, opt in enumerate(opts):
        if opt.lower() == val_lower: return i
    return 0

def get_cat_mode_idx(preprocessor, col_name: str) -> int:
    if preprocessor and col_name in preprocessor.fill_values and col_name in preprocessor.label_encoders:
        mode_val = str(preprocessor.fill_values[col_name])
        le = preprocessor.label_encoders[col_name]
        if mode_val in le.classes_:
            return int(le.transform([mode_val])[0])
    return 0

def merge_diagnosis_label(raw_diag: str) -> str:
    if pd.isna(raw_diag): return "MISC"
    d = raw_diag.lower().strip()

    melanoma = ["melanoma", "melanoma (0.76 to 1.5 mm)", "melanoma (in situ)",
                "melanoma (less than 0.76 mm)", "melanoma (more than 1.5 mm)",
                "melanoma metastasis"]
    nevus = ["clark nevus", "combined nevus", "congenital nevus", "dermal nevus",
             "recurrent nevus", "reed or spitz nevus", "blue nevus"]
    misc = ["dermatofibroma", "lentigo", "melanosis", "miscellaneous", "vascular lesion"]

    if any(m in d for m in melanoma): return "MEL"
    if any(n in d for n in nevus): return "NEV"
    if any(m in d for m in misc): return "MISC"
    if "basal cell" in d: return "BCC"
    if "seborrheic" in d: return "SK"
    return "MISC"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — perturbation preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def apply_gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0: return x
    k = max(3, int(2 * round(sigma) + 1))
    sigma = max(sigma, 0.1)
    coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g[None, None, :, None] * g[None, None, None, :]).to(x.device)
    c = x.size(1)
    w = kernel.expand(c, 1, k, k)
    return F.conv2d(x, w, padding=k // 2, groups=c)

def tensor_to_image(tensor):
    t = tensor.squeeze(0).clone()
    for i in range(3):
        t[i].mul_(IMG_STD[i]).add_(IMG_MEAN[i])
    t.clamp_(0, 1)
    return transforms.ToPILImage()(t)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — image preprocessing
# ─────────────────────────────────────────────────────────────────────────────
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])

def load_and_transform(image_path: Path) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    tensor = IMG_TRANSFORM(img)
    return tensor.unsqueeze(0)

def zero_image_tensor() -> torch.Tensor:
    return torch.zeros((1, 3, IMG_SIZE, IMG_SIZE))

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — tabular loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_preprocessor(path: Path):
    if not path.exists(): return None
    return joblib.load(path)

def build_tab_tensors(cat_values: dict, num_values: dict, preprocessor) -> tuple[torch.Tensor, torch.Tensor]:
    row = {**cat_values, **num_values}
    cat_arr, num_arr = preprocessor.transform(pd.DataFrame([row]))
    return torch.tensor(cat_arr, dtype=torch.long), torch.tensor(num_arr, dtype=torch.float32)

def zero_tab_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros((1, len(CFG.CAT_COLS)), dtype=torch.long), torch.zeros((1, len(CFG.NUM_COLS)), dtype=torch.float32)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — loading & inference & interpretability
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    if not path.exists(): return None
    try:
        sd = torch.load(str(path), map_location="cpu")
        if "model_state_dict" in sd: sd = sd["model_state_dict"]
        elif "state_dict" in sd: sd = sd["state_dict"]
        cat_dims, i = [], 0
        while f"ft_encoder.cat_embeds.{i}.weight" in sd:
            cat_dims.append(sd[f"ft_encoder.cat_embeds.{i}.weight"].shape[0] - 1); i += 1
        has_reliability = any("reliability" in k for k in sd.keys())
        if has_reliability:
            model = ReliabilityGatedModelV5(cat_dims, len(CLASSES))
        else:
            model = RepairedFusionModelV5(cat_dims, len(CLASSES))
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model
    except Exception as exc:
        st.error(f"Model load failed from {path.name}: {exc}")
        return None

def generate_gradcam(model, derm_t, clinic_t, tab_cat_t, tab_num_t, target_idx, modality="derm"):
    if not HAS_GRADCAM: return None
    model.eval()
    if modality == "derm": wrapper, layer, tensor = DermWrapper(model, clinic_t, tab_cat_t, tab_num_t), [model.net.backbone_d.features[-1]], derm_t
    else: wrapper, layer, tensor = ClinicWrapper(model, derm_t, tab_cat_t, tab_num_t), [model.net.backbone_c.features[-1]], clinic_t
    cam = GradCAM(model=wrapper, target_layers=layer)
    return cam(input_tensor=tensor, targets=[ClassifierOutputTarget(target_idx)])[0, :]

@torch.no_grad()
def get_tabular_attention(model, derm_t, clinic_t, tab_cat_t, tab_num_t):
    ft = model.ft_encoder
    tokens = [emb(tab_cat_t[:, i].long()) for i, emb in enumerate(ft.cat_embeds)] + \
             [proj(tab_num_t[:, i:i+1]) for i, proj in enumerate(ft.num_projectors)]
    x = ft.token_proj(torch.stack(tokens, 1)) + ft.pos_bias
    attn_weights = None
    for i, layer in enumerate(ft.transformer.layers):
        if i == len(ft.transformer.layers) - 1:
            x_norm = layer.norm1(x)
            _, attn_weights = layer.self_attn(x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=True)
            break
        else:
            x = layer(x)
    if attn_weights is not None:
        aw = attn_weights.squeeze(0)
        token_importance = aw.sum(dim=0).cpu().numpy()
        return token_importance
    return None

@torch.no_grad()
def run_inference(model, derm_t, clinic_t, tab_cat_t, tab_num_t) -> dict:
    outputs = model(derm_t, clinic_t, tab_cat_t, tab_num_t)
    if len(outputs) == 7:
        out_d, out_c, out_dc, out_tab, final_logits, clip_loss, extras = outputs
        w_img = extras["w_img"].squeeze().item()
        w_tab = extras["w_tab"].squeeze().item()
    elif len(outputs) == 6:
        out_d, out_c, out_dc, out_tab, final_logits, clip_loss = outputs
        w_img, w_tab = 1.0, 1.0
    else:
        final_logits = outputs[4]
        w_img, w_tab = 1.0, 1.0
    probs    = F.softmax(final_logits, dim=-1).squeeze(0).numpy()
    pred_idx = int(np.argmax(probs))
    return {
        "predicted":  CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs":      {cls: float(probs[i]) for i, cls in enumerate(CLASSES)},
        "w_img":      w_img,
        "w_tab":      w_tab,
    }

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h1 style='font-size:1.1rem; margin-bottom:0'>🔬 Kerbeus</h1>"
        "<p style='font-size:0.65rem; color:#4a7a8a; letter-spacing:0.2em; margin-top:0.2rem'>MULTIMODAL LESION CLASSIFIER</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Modality ──────────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>⚙ MODALITY CONFIGURATION</div>", unsafe_allow_html=True)
    modality_mode = st.radio(
        "Active modalities",
        options=["Full Modalities", "Exclude Tabular Data", "Exclude Images"],
        index=0,
        key="modality_mode",
    )
    use_images   = modality_mode != "Exclude Images"
    use_tabular  = modality_mode != "Exclude Tabular Data"
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Interpretability ──────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>👁️ INTERPRETABILITY</div>", unsafe_allow_html=True)
    show_gradcam = st.checkbox(
        "Generate Feature Explanations",
        value=False,
        disabled=not (use_images or use_tabular),
        key="show_gradcam",
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── OOD Controls ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>🛠️ ABLATION / OOD TESTING</div>", unsafe_allow_html=True)
    blur_sigma   = st.slider("Image Blur (σ)", 0.0, 5.0, 0.0, step=0.1, key="blur_sigma")
    tab_mask_pct = st.slider("Tabular Masking (%)", 0, 100, 0, step=10, key="tab_mask_pct")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='section-label'>RUNTIME</div>"
        f"<p style='font-family:monospace; font-size:0.78rem; color:#00e5ff88'>"
        f"▶ DEVICE: {'CUDA' if torch.cuda.is_available() else 'CPU'}</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-RUN DETECTION
# ─────────────────────────────────────────────────────────────────────────────
_param_changed = (
    st.session_state["_prev_blur"]     != blur_sigma     or
    st.session_state["_prev_mask"]     != tab_mask_pct   or
    st.session_state["_prev_gradcam"]  != show_gradcam   or
    st.session_state["_prev_modality"] != modality_mode
)

st.session_state["_prev_blur"]     = blur_sigma
st.session_state["_prev_mask"]     = tab_mask_pct
st.session_state["_prev_gradcam"]  = show_gradcam
st.session_state["_prev_modality"] = modality_mode

# ─────────────────────────────────────────────────────────────────────────────
# HEADER & DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center; padding:1.5rem 0 0.5rem'>"
    "<h1 style='font-size:2.4rem; margin:0; letter-spacing:0.25em'>Kerbeus</h1>"
    "<p style='font-family:monospace; font-size:0.8rem; color:#4a7a8a; letter-spacing:0.3em; margin-top:0.3rem'>"
    "MULTIMODAL SKIN LESION CLASSIFICATION SYSTEM</p></div>",
    unsafe_allow_html=True,
)

with st.spinner("Loading artifacts …"):
    preprocessor = load_preprocessor(PREPROCESSOR)
    model_full   = load_model(MODEL_FULL_PATH)
    model_ood    = load_model(MODEL_OOD_PATH)
    meta_df, test_indexes = load_meta(META_PATH), load_test_indexes(TEST_IDX_PATH)

selected_row = None
if meta_df is not None:
    st.markdown("<div class='section-label'>00 · SELECT PATIENT CASE (meta.csv)</div>", unsafe_allow_html=True)
    display_df  = meta_df.iloc[test_indexes].reset_index(drop=True) if test_indexes is not None else meta_df.copy()
    case_labels = display_df.apply(lambda r: f"Case {r.get('case_num','?')} — {r.get('diagnosis','Unknown')}", axis=1).tolist()
    selected_idx = st.selectbox("Select a unified patient case", options=range(len(display_df)), format_func=lambda i: case_labels[i])
    selected_row = display_df.iloc[selected_idx]
    st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
col_derm, col_clinic, col_tab = st.columns([1, 1, 1.4], gap="large")

with col_derm:
    st.markdown("<div class='section-label'>01 · DERMOSCOPY IMAGE</div>", unsafe_allow_html=True)
    selected_derm = get_image_path(DERM_DIR, selected_row["derm"]) if selected_row is not None else None
    if use_images and selected_derm and selected_derm.exists():
        st.image(Image.open(selected_derm).convert("RGB"), width="stretch")

with col_clinic:
    st.markdown("<div class='section-label'>02 · CLINICAL IMAGE</div>", unsafe_allow_html=True)
    selected_clinic = get_image_path(CLINIC_DIR, selected_row["clinic"]) if selected_row is not None else None
    if use_images and selected_clinic and selected_clinic.exists():
        st.image(Image.open(selected_clinic).convert("RGB"), width="stretch")

with col_tab:
    st.markdown("<div class='section-label'>03 · TABULAR FEATURES</div>", unsafe_allow_html=True)
    cat_inputs, num_inputs = {}, {}
    if use_tabular:
        if selected_row is not None:
            score         = selected_row.get("seven_point_score")
            default_score = float(preprocessor.fill_values.get("seven_point_score", 0.0)) if preprocessor else 0.0
            num_inputs["seven_point_score"] = float(score) if pd.notnull(score) else default_score
            for col in CFG.CAT_COLS:
                val = selected_row.get(col)
                if pd.isna(val) and preprocessor:
                    val = preprocessor.fill_values.get(col, CAT_OPTIONS[col][0])
                cat_inputs[col] = str(val)
        else:
            num_inputs["seven_point_score"] = float(preprocessor.fill_values.get("seven_point_score", 0.0)) if preprocessor else 0.0
            for col in CFG.CAT_COLS:
                val = preprocessor.fill_values.get(col) if preprocessor else CAT_OPTIONS[col][0]
                cat_inputs[col] = str(val)

        html = "<div style='display:grid; grid-template-columns: 1fr 1fr; gap:0.5rem;'>"
        for col in CFG.CAT_COLS + CFG.NUM_COLS:
            val  = cat_inputs[col] if col in CFG.CAT_COLS else num_inputs[col]
            name = SHORT_COLS.get(col, col)
            html += (
                f"<div class='glow-panel' style='padding:0.5rem; margin-bottom:0;'>"
                f"<p style='font-size:0.6rem; color:var(--text-muted); margin:0; text-transform:uppercase;'>{name}</p>"
                f"<p style='font-family:var(--font-mono); font-size:0.85rem; color:var(--accent-cyan); margin:0;'>{val}</p>"
                f"</div>"
            )
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RUN BUTTON 
# ─────────────────────────────────────────────────────────────────────────────
_left, _mid, _right = st.columns([2, 1, 2])
with _mid:
    run_clicked = st.button("⚡  RUN PREDICTION", key="run_btn", width="stretch")

should_run = run_clicked or (st.session_state["has_run"] and _param_changed)

if run_clicked:
    st.session_state["has_run"] = True

is_ood       = blur_sigma > 0 or tab_mask_pct > 0
active_model = model_ood if is_ood else model_full

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if should_run:
    if active_model is None:
        st.error(
            f"The required model file "
            f"({'model.pt' if is_ood else 'model_full.pt'}) "
            f"was not found. Please verify the checkpoints are in the root directory."
        )
    else:
        model_label = "Reliability-Gated OOD Model" if is_ood else "Standard Full Model"
        with st.spinner(f"Processing with {model_label}…"):
            if use_images:
                derm_t   = load_and_transform(selected_derm)   if (selected_derm   and selected_derm.exists())   else zero_image_tensor()
                clinic_t = load_and_transform(selected_clinic) if (selected_clinic and selected_clinic.exists()) else zero_image_tensor()
            else:
                derm_t = clinic_t = zero_image_tensor()

            tab_cat_t, tab_num_t = (
                build_tab_tensors(cat_inputs, num_inputs, preprocessor)
                if use_tabular else zero_tab_tensors()
            )

            derm_p   = apply_gaussian_blur(derm_t,   blur_sigma) if use_images else derm_t
            clinic_p = apply_gaussian_blur(clinic_t, blur_sigma) if use_images else clinic_t

            tab_cat_p = tab_cat_t.clone()
            tab_num_p = tab_num_t.clone()
            masked_cols: list[str] = []

            if use_tabular and tab_mask_pct > 0:
                frac          = tab_mask_pct / 100.0
                n_cat, n_num  = len(CFG.CAT_COLS), len(CFG.NUM_COLS)
                n_mask_cat    = int(round(frac * n_cat))
                n_mask_num    = int(round(frac * n_num))
                if n_mask_cat > 0:
                    for i in np.random.choice(n_cat, n_mask_cat, replace=False):
                        col_name = CFG.CAT_COLS[i]
                        masked_cols.append(col_name)
                        tab_cat_p[0, i] = get_cat_mode_idx(preprocessor, col_name)
                if n_mask_num > 0:
                    for i in np.random.choice(n_num, n_mask_num, replace=False):
                        masked_cols.append(CFG.NUM_COLS[i])
                        tab_num_p[0, i] = 0.0

            results           = run_inference(active_model, derm_p, clinic_p, tab_cat_p, tab_num_p)
            pred, conf, probs = results["predicted"], results["confidence"], results["probs"]
            w_img, w_tab      = results["w_img"], results["w_tab"]

        st.markdown("<hr style='margin-top:3rem; margin-bottom:2rem;'>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION A — STANDARD RESULTS (always shown)
        # ══════════════════════════════════════════════════════════════════════
        color = CLASS_COLORS.get(pred, "#00e5ff")
        st.markdown(
            f"<div class='glow-panel' style='text-align:center; border-color:{color}55; box-shadow:0 0 40px {color}22'>"
            f"<p style='font-family:monospace; font-size:0.7rem; color:#4a7a8a; letter-spacing:0.3em; margin:0'>PREDICTED CLASS</p>"
            f"<div class='result-winner' style='color:{color}; text-shadow:0 0 30px {color}99'>{pred}</div>"
            f"<span class='conf-badge' style='border-color:{color}66; color:{color}'>CONFIDENCE &nbsp; {conf*100:.2f} %</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if selected_row is not None:
            actual  = merge_diagnosis_label(selected_row["diagnosis"])
            match   = "CORRECT" if actual == pred else "MISMATCH"
            m_color = "#69ff47"  if actual == pred else "#ff4081"
            st.markdown(
                f"<div class='glow-panel' style='border-color:{m_color}44; padding:1rem; text-align:center'>"
                f"<p style='font-family:monospace; font-size:0.6rem; color:#4a7a8a; letter-spacing:0.2em; margin:0'>VALIDATION VS GROUND TRUTH</p>"
                f"<p style='font-family:monospace; font-size:0.9rem; color:{m_color}; margin:0.3rem 0'>"
                f"<b>{match}</b> — ACTUAL: {actual} ({selected_row['diagnosis']})</p></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='section-label'>PROBABILITY DISTRIBUTION</div>", unsafe_allow_html=True)
        prob_rows_html = ""
        for cls, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            b_c = CLASS_COLORS.get(cls, "#00e5ff"); b_w = max(p * 100, 0.5)
            prob_rows_html += (
                f"<div style='display:flex; align-items:center; margin-bottom:0.5rem; gap:0.5rem'>"
                f"<span style='font-family:monospace; font-size:0.7rem; width:3rem; color:{b_c}'>{cls}</span>"
                f"<div style='flex:1; background:#0c1526; height:18px; border-radius:2px; overflow:hidden'>"
                f"<div style='width:{b_w:.1f}%; height:100%; background:linear-gradient(90deg, {b_c}cc, {b_c}33); box-shadow:0 0 5px {b_c}44'></div></div>"
                f"<span style='font-family:monospace; font-size:0.7rem; width:4rem; text-align:right; color:{b_c}'>{p*100:.2f}%</span>"
                f"</div>"
            )
        st.markdown(prob_rows_html, unsafe_allow_html=True)

        if show_gradcam:
            st.markdown("<div class='section-label'>👁️ MODEL ATTENTION & FEATURE IMPORTANCE</div>", unsafe_allow_html=True)

            if use_images and use_tabular:
                gc1, gc2, gc3 = st.columns([1, 1, 1.2])
            elif use_images:
                gc1, gc2 = st.columns(2); gc3 = None
            elif use_tabular:
                gc3 = st.columns(1)[0]; gc1 = gc2 = None
            else:
                gc1 = gc2 = gc3 = None

            if use_images and active_model is not None:
                p_idx = CLASSES.index(pred)
                
                # Check for GradCAM availability and show explicit warnings if missing
                if not HAS_GRADCAM:
                    if gc1: gc1.error(f"GradCAM Import Failed: {GRADCAM_ERR}")
                    if gc2: gc2.info("Ensure 'grad-cam' and 'opencv-python-headless' are correctly installed in your deployment environment.")
                else:
                    if selected_derm and gc1:
                        cam_d = generate_gradcam(active_model, derm_p, clinic_p, tab_cat_p, tab_num_p, p_idx, "derm")
                        if cam_d is not None:
                            gc1.image(
                                show_cam_on_image(
                                    np.float32(Image.open(selected_derm).convert("RGB").resize((IMG_SIZE, IMG_SIZE))) / 255.0,
                                    cam_d, use_rgb=True,
                                ),
                                caption="Dermoscopy Focus", width="stretch",
                            )
                    if selected_clinic and gc2:
                        cam_c = generate_gradcam(active_model, derm_p, clinic_p, tab_cat_p, tab_num_p, p_idx, "clinic")
                        if cam_c is not None:
                            gc2.image(
                                show_cam_on_image(
                                    np.float32(Image.open(selected_clinic).convert("RGB").resize((IMG_SIZE, IMG_SIZE))) / 255.0,
                                    cam_c, use_rgb=True,
                                ),
                                caption="Clinical Focus", width="stretch",
                            )

            if use_tabular and gc3 and active_model is not None:
                tab_attn = get_tabular_attention(active_model, derm_p, clinic_p, tab_cat_p, tab_num_p)
                if tab_attn is not None:
                    features      = CFG.CAT_COLS + CFG.NUM_COLS
                    short_features = [SHORT_COLS[f] for f in features]
                    tab_attn      = tab_attn / (tab_attn.max() + 1e-8)
                    sorted_idx    = np.argsort(tab_attn)[::-1]
                    attn_html     = (
                        "<div class='glow-panel' style='height:100%; padding:1rem;'>"
                        "<p style='text-align:center; font-family:var(--font-mono); color:var(--text-primary); margin-bottom:15px;'>"
                        "Tabular Feature Importance (FT-Transformer)</p>"
                    )
                    for i in sorted_idx:
                        feat = short_features[i]; val = tab_attn[i]
                        attn_html += (
                            f"<div style='display:flex; align-items:center; margin-bottom:8px;'>"
                            f"<span style='width:35%; font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted); text-align:right; padding-right:10px;'>{feat}</span>"
                            f"<div style='flex:1; background:var(--bg-void); height:14px; border-radius:2px; overflow:hidden; border:1px solid #1e3a4a;'>"
                            f"<div style='width:{val*100}%; height:100%; background:linear-gradient(90deg,#00e5ff,#69ff47); box-shadow:0 0 5px #00e5ff88;'></div>"
                            f"</div>"
                            f"<span style='width:15%; font-family:var(--font-mono); font-size:0.75rem; color:var(--accent-cyan); text-align:right;'>{val:.2f}</span>"
                            f"</div>"
                        )
                    attn_html += "</div>"
                    gc3.markdown(attn_html, unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION B — ABLATION STUDY (shown below standard results when OOD active)
        # ══════════════════════════════════════════════════════════════════════
        if is_ood:
            st.markdown("<hr style='margin-top:2.5rem; margin-bottom:2rem;'>", unsafe_allow_html=True)
            st.markdown("<div class='ablation-header'>Ablation Study Results</div>", unsafe_allow_html=True)

            subtitle_parts = []
            if blur_sigma   > 0:   subtitle_parts.append(f"Blur σ = {blur_sigma}")
            if tab_mask_pct > 0:   subtitle_parts.append(f"Tabular Masking = {tab_mask_pct}%")
            st.markdown(
                f"<p style='text-align:center; color:#e0f7ff; margin-top:15px;'>{' | '.join(subtitle_parts)}</p>"
                f"<p style='text-align:center; color:#00e5ff; font-family:\"Share Tech Mono\",monospace; font-size:1.1rem; margin-bottom:30px;'>"
                f"Prediction: {pred} | Confidence: {conf:.3f}</p>",
                unsafe_allow_html=True,
            )

            table_html = (
                "<table style='width:100%; border-collapse:collapse; text-align:center; "
                "font-family:\"Share Tech Mono\",monospace; font-size:0.8rem; "
                "background:#0c1526; border:1px solid #333; margin-bottom:40px;'>"
                "<tr style='border-bottom:1px solid #333; background:#111827; color:#a0c4cc;'>"
            )
            for col in CFG.CAT_COLS + CFG.NUM_COLS:
                table_html += f"<th style='padding:8px; border-right:1px solid #333;'>{SHORT_COLS[col]}</th>"
            table_html += "</tr><tr>"
            for col in CFG.CAT_COLS:
                is_masked  = col in masked_cols
                val        = "" if is_masked else cat_inputs[col]
                bg, fg     = ("#4a0000", "#4a0000") if is_masked else ("transparent", "#00e5ff")
                table_html += f"<td style='padding:8px; border-right:1px solid #333; background:{bg}; color:{fg};'>{val}</td>"
            for col in CFG.NUM_COLS:
                is_masked  = col in masked_cols
                val        = "" if is_masked else f"{num_inputs[col]:.4f}"
                bg, fg     = ("#4a0000", "#4a0000") if is_masked else ("transparent", "#00e5ff")
                table_html += f"<td style='padding:8px; border-right:1px solid #333; background:{bg}; color:{fg};'>{val}</td>"
            table_html += "</tr></table>"
            st.markdown(table_html, unsafe_allow_html=True)

            r1, r2, r3 = st.columns([1, 1, 1.2])
            img_tag = "(Blurred)" if blur_sigma > 0 else "(Clean)"

            r1.markdown(
                f"<p style='text-align:center; color:#e0f7ff; font-family:\"Share Tech Mono\",monospace;'>Derm Image {img_tag}</p>",
                unsafe_allow_html=True,
            )
            if use_images:
                r1.image(tensor_to_image(derm_p), width="stretch")

            r2.markdown(
                f"<p style='text-align:center; color:#e0f7ff; font-family:\"Share Tech Mono\",monospace;'>Clinic Image {img_tag}</p>",
                unsafe_allow_html=True,
            )
            if use_images:
                r2.image(tensor_to_image(clinic_p), width="stretch")

            chart_html = textwrap.dedent(f"""
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; width:100%; height:100%;">
                <div style="color:#e0f7ff; font-family:'Share Tech Mono',monospace; font-size:1rem; margin-bottom:10px;">Model Behavior</div>
                <div style="position:relative; width:100%; max-width:320px; height:250px; padding-left:30px;">
                    <div style="position:absolute; left:0; top:0; height:100%; display:flex; flex-direction:column; justify-content:space-between; color:#a0c4cc; font-family:'Share Tech Mono',monospace; font-size:0.7rem; padding-bottom:20px;">
                        <span>1.0</span><span>0.8</span><span>0.6</span><span>0.4</span><span>0.2</span><span>0.0</span>
                    </div>
                    <div style="height:100%; border-left:1px solid #a0c4cc; border-bottom:1px solid #a0c4cc; display:flex; align-items:flex-end; justify-content:space-around; padding:0 10px;">
                        <div style="width:20%; height:{w_img*100}%; background:#ff4081; box-shadow:0 0 10px rgba(255,64,129,0.5);"></div>
                        <div style="width:20%; height:{w_tab*100}%; background:#00e5ff; box-shadow:0 0 10px rgba(0,229,255,0.5);"></div>
                        <div style="width:20%; height:{conf*100}%; background:#ffd740; box-shadow:0 0 10px rgba(255,215,64,0.5);"></div>
                    </div>
                    <div style="display:flex; justify-content:space-around; color:#e0f7ff; font-family:'Share Tech Mono',monospace; font-size:0.8rem; margin-top:10px; margin-left:25px;">
                        <span style="width:33%; text-align:center;">Image</span>
                        <span style="width:33%; text-align:center;">Tabular</span>
                        <span style="width:33%; text-align:center;">Confidence</span>
                    </div>
                </div>
            </div>
            """)
            r3.markdown(chart_html, unsafe_allow_html=True)

            st.markdown(
                "<h2 style='text-align:center; color:white; font-family:\"Orbitron\",sans-serif; "
                "font-size:1.8rem; padding:2rem 0;'>"
                "Evaluating Model Stability in Lesion Detection Using Image Blur and Feature Masking Techniques</h2>",
                unsafe_allow_html=True,
            )

st.markdown(
    "<hr style='margin-top:3rem'>"
    "<p style='font-size:0.6rem; color:#1e3a4a; text-align:center; letter-spacing:0.25em'>"
    "Kerbeus &nbsp;·&nbsp; DERM7PT DATASET &nbsp;·&nbsp; FOR RESEARCH USE ONLY</p>",
    unsafe_allow_html=True,
)
