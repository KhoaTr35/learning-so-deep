"""
Streamlit demo for the 20 Newsgroups text classifiers.

Supports three model checkpoints:
  - transformer_gpu_safe.pt   (Transformer encoder)
  - rnn_gpu_safe.pt           (Bidirectional GRU + attention)
  - ensemble.pt               (Learned meta-ensemble of both)

Prerequisites (run the notebook cells first):
  - model_meta.json
  - transformer_gpu_safe.pt
  - rnn_gpu_safe.pt
  - ensemble.pt

Launch:
  streamlit run streamt_lit.py
"""

import json
import math
import re

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="20 Newsgroups Classifier · Model Comparison",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — clean, modern look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── global ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── header banner ── */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem 1.8rem;
        margin-bottom: 1.8rem;
        color: white;
    }
    .hero h1 { font-size: 2.2rem; font-weight: 700; margin: 0 0 0.3rem; }
    .hero p  { font-size: 1rem; opacity: 0.8; margin: 0; }

    /* ── prediction card ── */
    .pred-card {
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1rem;
        color: white;
        font-size: 1.15rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    .pred-card .conf {
        margin-left: auto;
        font-size: 1.05rem;
        opacity: 0.9;
        background: rgba(255,255,255,0.18);
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
    }

    /* ── info badge ── */
    .badge {
        display: inline-block;
        background: #e8f4fd;
        color: #1a6ea8;
        border-radius: 20px;
        padding: 0.2rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0.15rem;
    }

    /* ── section title ── */
    .section-title {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #666;
        margin: 1.4rem 0 0.6rem;
    }

    /* ── sidebar ── */
    section[data-testid="stSidebar"] { background: #f8f9fb; }

    /* hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Category colour palette  (maps newsgroup prefix → hex colour)
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "alt"   : "#e74c3c",
    "comp"  : "#3498db",
    "misc"  : "#95a5a6",
    "rec"   : "#2ecc71",
    "sci"   : "#9b59b6",
    "soc"   : "#e67e22",
    "talk"  : "#1abc9c",
}
DEFAULT_COLOR = "#34495e"


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert a 6-digit hex color string to an rgba() string Plotly accepts."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def label_color(label_name: str) -> str:
    prefix = label_name.split(".")[0].lower()
    return CATEGORY_COLORS.get(prefix, DEFAULT_COLOR)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer architecture  (matches transformer_gpu_safe.pt from BTL_FINAL.ipynb)
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    """Improved Transformer — PyTorch built-in encoder, CLS + mean-pool classifier."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        max_len: int = 192,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding     = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder   = SinusoidalPositionalEncoding(d_model, max_len)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        # Input = [CLS repr ; mean-pool of content] → 2 × d_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pad_mask = (input_ids == self.pad_idx)          # (B, L) True = PAD
        x = self.embed_dropout(self.pos_encoder(self.embedding(input_ids)))
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, L, D)

        cls_repr    = x[:, 0, :]                        # (B, D)
        content_x   = x[:, 1:, :]
        content_pad = pad_mask[:, 1:]
        valid_mask  = (~content_pad).unsqueeze(-1).float()
        mean_repr   = (content_x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)

        return self.classifier(torch.cat([cls_repr, mean_repr], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# RNN architecture  (mirrors the notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 192,
        hidden_size: int = 192,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx   = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout   = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pad_mask = (input_ids != self.pad_idx)
        lengths  = pad_mask.sum(dim=1).cpu()
        x = self.dropout(self.embedding(input_ids))
        packed     = nn.utils.rnn.pack_padded_sequence(
            x, lengths=lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=input_ids.size(1)
        )
        attn_scores  = self.attn(out).squeeze(-1)
        attn_scores  = attn_scores.masked_fill(~pad_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        pooled = (out * attn_weights).sum(dim=1)
        return self.classifier(pooled)


# ─────────────────────────────────────────────────────────────────────────────
# Learned Ensemble architecture  (mirrors the notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
class LearnedEnsemble(nn.Module):
    """Meta-MLP that combines logits from frozen Transformer + RNN."""

    def __init__(self, model_a: nn.Module, model_b: nn.Module, num_classes: int):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
        for p in self.model_a.parameters():
            p.requires_grad = False
        for p in self.model_b.parameters():
            p.requires_grad = False
        self.meta = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes * 4),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(num_classes * 4, num_classes * 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(num_classes * 2, num_classes),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            la = self.model_a(input_ids)
            lb = self.model_b(input_ids)
        return self.meta(torch.cat([la, lb], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities  (identical to the notebook)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(
        r"^(from|subject|organization|lines|nntp-posting-host|reply-to|x-newsreader)[^\n]*\n",
        "", text, flags=re.MULTILINE,
    )
    text = re.sub(r"^>.*\n?", "",          text, flags=re.MULTILINE)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+",          " ", text)
    text = re.sub(r"[^a-z\s]",         " ", text)
    return re.sub(r"\s+", " ", text).strip()


def encode(text: str, word2idx: dict, max_len: int) -> list:
    PAD_IDX = word2idx["<PAD>"]
    CLS_IDX = word2idx["<CLS>"]
    UNK_IDX = word2idx["<UNK>"]
    tokens  = text.split()[: max_len - 1]
    ids     = [CLS_IDX] + [word2idx.get(t, UNK_IDX) for t in tokens]
    ids     = ids[:max_len]
    ids    += [PAD_IDX] * (max_len - len(ids))
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# Model specs — one entry per checkpoint
# ─────────────────────────────────────────────────────────────────────────────
# max_len must match what each model was trained with (BTL_FINAL: MAX_SEQ_LEN=192)
_TF_MAX_LEN  = 192
_RNN_MAX_LEN = 192

MODEL_SPECS = {
    "🤖 Transformer": {
        "file"       : "models/transformer_gpu_safe.pt",
        "description": "Improved Transformer encoder (PyTorch built-in, pre-norm, CLS+mean-pool)",
        "color"      : "#3498db",
        "max_len"    : _TF_MAX_LEN,
    },
    "🔁 RNN (Bi-GRU)": {
        "file"       : "models/rnn_gpu_safe.pt",
        "description": "Bidirectional GRU with attention pooling",
        "color"      : "#2ecc71",
        "max_len"    : _RNN_MAX_LEN,
    },
    "⚡ Ensemble": {
        "file"       : "models/ensemble.pt",
        "description": "Learned meta-MLP over Transformer + RNN logits (both backbones frozen)",
        "color"      : "#e67e22",
        "max_len"    : _TF_MAX_LEN,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Cached resource loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading shared metadata …")
def load_meta():
    with open("artifacts/model_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    word2idx    = meta["word2idx"]
    label_names = meta["label_names"]
    pad_token   = meta.get("pad_token", "<PAD>")
    derived_cfg = {
        "vocab_size":  len(word2idx),
        "num_classes": len(label_names),
        "pad_idx":     word2idx.get(pad_token, 0),
    }
    return word2idx, label_names, derived_cfg


# Hardcoded config that matches the transformer_gpu_safe.pt checkpoint
_TRANSFORMER_CFG = dict(
    vocab_size=45115, num_classes=20,
    d_model=256, num_heads=8, num_layers=6, d_ff=1024,
    max_len=192, dropout=0.3, pad_idx=0,
)


@st.cache_resource(show_spinner="Loading Transformer …")
def load_transformer():
    model = TransformerClassifier(**_TRANSFORMER_CFG)
    state = torch.load("models/transformer_gpu_safe.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading RNN …")
def load_rnn(vocab_size, num_classes, pad_idx):
    model = RNNClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embed_dim=192,
        hidden_size=192,
        num_layers=2,
        dropout=0.3,
        pad_idx=pad_idx,
    )
    state = torch.load("models/rnn_gpu_safe.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading Ensemble …")
def load_ensemble(vocab_size, num_classes, pad_idx):
    transformer_model = load_transformer()
    rnn_model         = load_rnn(vocab_size, num_classes, pad_idx)
    model = LearnedEnsemble(transformer_model, rnn_model, num_classes)
    meta_state = torch.load("models/ensemble.pt", map_location="cpu")
    model.meta.load_state_dict(meta_state)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(text: str, model, word2idx: dict, max_len: int, label_names: list):
    clean   = preprocess_text(text)
    ids     = encode(clean, word2idx, max_len)
    x       = torch.tensor([ids], dtype=torch.long)
    logits  = model(x)
    probs   = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    top_idx = probs.index(max(probs))
    return label_names[top_idx], probs[top_idx], probs, clean


# ─────────────────────────────────────────────────────────────────────────────
# Load shared metadata (always needed)
# ─────────────────────────────────────────────────────────────────────────────
try:
    word2idx, label_names, _meta_cfg = load_meta()
    # These values come from model_meta.json (saved by btl-v6-gpu-rewrite);
    # the transformer has its own hardcoded config in _TRANSFORMER_CFG above.
    VOCAB_SIZE  = _meta_cfg["vocab_size"]   # used for RNN loader
    NUM_CLASSES = _meta_cfg["num_classes"]
    PAD_IDX     = _meta_cfg["pad_idx"]
    meta_loaded = True
except FileNotFoundError as e:
    meta_loaded = False
    missing_file = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# ── Header ────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>📰 20 Newsgroups Classifier</h1>
        <p>Compare Transformer · Bi-GRU RNN · Learned Ensemble — all trained from scratch in PyTorch</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not meta_loaded:
    st.error(
        f"**Could not load `model_meta.json`.**\n\n"
        f"`{missing_file}`\n\n"
        "Run the notebook cells first, then relaunch this app."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# ── Sidebar ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Select Model")
    selected_model_name = st.radio(
        "Model",
        list(MODEL_SPECS.keys()),
        label_visibility="collapsed",
    )
    spec = MODEL_SPECS[selected_model_name]

    st.markdown(
        f'<div style="background:{spec["color"]}22;border-left:4px solid {spec["color"]};'
        f'padding:0.6rem 0.9rem;border-radius:8px;font-size:0.85rem;margin-bottom:1rem">'
        f'{spec["description"]}</div>',
        unsafe_allow_html=True,
    )

    # ── Model card ────────────────────────────────────────────────────────────
    st.markdown("## 🔧 Model Card")

    if selected_model_name == "🤖 Transformer":
        cfg = _TRANSFORMER_CFG
        st.markdown(
            f"""
            | Parameter | Value |
            |-----------|-------|
            | d_model   | {cfg['d_model']} |
            | Heads     | {cfg['num_heads']} |
            | Layers    | {cfg['num_layers']} |
            | d_ff      | {cfg['d_ff']} |
            | Vocab     | {cfg['vocab_size']:,} |
            | Max seq   | {cfg['max_len']} |
            | Classes   | {cfg['num_classes']} |
            """
        )
    elif selected_model_name == "🔁 RNN (Bi-GRU)":
        st.markdown(
            f"""
            | Parameter   | Value |
            |-------------|-------|
            | embed_dim   | 192 |
            | hidden_size | 192 (bi → 384) |
            | Layers      | 2 |
            | Pooling     | Attention |
            | Vocab       | {VOCAB_SIZE:,} |
            | Max seq     | {spec["max_len"]} |
            | Classes     | {NUM_CLASSES} |
            """
        )
    else:  # Ensemble
        st.markdown(
            f"""
            | Parameter    | Value |
            |--------------|-------|
            | Backbone A   | Transformer |
            | Backbone B   | Bi-GRU RNN |
            | Meta input   | {NUM_CLASSES * 2} logits |
            | Meta hidden  | {NUM_CLASSES * 4} / {NUM_CLASSES * 2} |
            | Meta output  | {NUM_CLASSES} classes |
            | Vocab        | {VOCAB_SIZE:,} |
            """
        )

    st.markdown("---")
    st.markdown("### 🏷️ Classes")
    for name in sorted(label_names):
        color = label_color(name)
        st.markdown(
            f'<span class="badge" style="background:{color}22;color:{color}">'
            f"{name}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<small>Built with PyTorch + Streamlit<br>"
        "Transformer: Vaswani et al., 2017<br>"
        "RNN: Bidirectional GRU + attention pooling</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# ── Load the selected model ───────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
active_model = None
load_error   = None

try:
    if selected_model_name == "🤖 Transformer":
        active_model = load_transformer()
    elif selected_model_name == "🔁 RNN (Bi-GRU)":
        active_model = load_rnn(VOCAB_SIZE, NUM_CLASSES, PAD_IDX)
    else:
        active_model = load_ensemble(VOCAB_SIZE, NUM_CLASSES, PAD_IDX)
except FileNotFoundError as e:
    load_error = str(e)

if load_error:
    st.error(
        f"**Could not load checkpoint.**\n\n"
        f"`{load_error}`\n\n"
        f"Make sure `{spec['file']}` exists in the working directory."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# ── Example selector ──────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLES = {
    "🚀 Space (sci.space)":
        "NASA has announced the launch of a new deep-space telescope designed to "
        "detect signs of life on distant exoplanets and study the formation of galaxies.",
    "💻 Hardware (comp.sys)":
        "I just upgraded my PC with the latest graphics card. The benchmark scores in "
        "3D rendering and gaming look incredibly promising compared to the old setup.",
    "⚾ Baseball (rec.sport.baseball)":
        "The team had an amazing season this year. Their pitcher dominated every game "
        "and the batting lineup was consistent throughout the playoffs.",
    "🏥 Medicine (sci.med)":
        "Recent clinical trials have shown promising results for the new vaccine against "
        "the virus. Side effects were minimal and efficacy was above 90%.",
    "✝️ Religion (soc.religion.christian)":
        "The sermon today focused on the power of forgiveness and the teachings of "
        "Christ regarding mercy, compassion, and love for one another.",
    "🔐 Cryptography (sci.crypt)":
        "The RSA algorithm relies on the computational difficulty of factoring large "
        "prime numbers. Modern quantum computers may eventually threaten its security.",
    "🏍️ Motorcycles (rec.motorcycles)":
        "Just got back from a 500-mile road trip on my bike. The engine ran perfectly "
        "and the suspension handled mountain curves without any issues.",
    "🗳️ Politics (talk.politics)":
        "The new legislation on gun control was hotly debated in Congress. Supporters "
        "argue it will reduce violence while opponents cite constitutional rights.",
    "✏️ Custom — type your own": "",
}

st.markdown('<p class="section-title">Try an example or write your own</p>', unsafe_allow_html=True)
selected = st.selectbox("Pick a sample text:", list(EXAMPLES.keys()), label_visibility="collapsed")

input_text = st.text_area(
    "Text to classify",
    value=EXAMPLES[selected],
    height=160,
    placeholder="Paste or type a newsgroup-style post here …",
    label_visibility="collapsed",
)

col_btn, col_spacer = st.columns([1, 5])
with col_btn:
    classify_clicked = st.button("🔍  Classify", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ── Results ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if classify_clicked:
    if not input_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner(f"Running inference with {selected_model_name} …"):
            pred_label, confidence, all_probs, cleaned = predict(
                input_text, active_model, word2idx, spec["max_len"], label_names
            )

        color = label_color(pred_label)
        model_color = spec["color"]

        # ── Model indicator ───────────────────────────────────────────────────
        st.markdown(
            f'<div style="display:inline-block;background:{model_color}22;color:{model_color};'
            f'border:1px solid {model_color}55;border-radius:20px;padding:0.25rem 0.9rem;'
            f'font-size:0.85rem;font-weight:700;margin-bottom:0.8rem">'
            f'Model: {selected_model_name}</div>',
            unsafe_allow_html=True,
        )

        # ── Top prediction card ───────────────────────────────────────────────
        st.markdown('<p class="section-title">Prediction</p>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="pred-card" style="background: linear-gradient(135deg, {color}cc, {color});">
                <span>📂 {pred_label}</span>
                <span class="conf">{confidence * 100:.1f}% confidence</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Two columns: bar chart + stats ────────────────────────────────────
        col_chart, col_stats = st.columns([3, 1])

        with col_chart:
            st.markdown('<p class="section-title">All class probabilities</p>', unsafe_allow_html=True)

            sorted_pairs  = sorted(zip(label_names, all_probs), key=lambda x: x[1], reverse=True)
            names_sorted  = [p[0] for p in sorted_pairs]
            probs_sorted  = [p[1] * 100 for p in sorted_pairs]
            colors_sorted = [
                hex_to_rgba(label_color(n), 1.0) if n == pred_label
                else hex_to_rgba(label_color(n), 0.35)
                for n in names_sorted
            ]

            fig = go.Figure(
                go.Bar(
                    x=probs_sorted,
                    y=names_sorted,
                    orientation="h",
                    marker_color=colors_sorted,
                    text=[f"{p:.1f}%" for p in probs_sorted],
                    textposition="outside",
                    hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
                )
            )
            fig.update_layout(
                xaxis=dict(title="Probability (%)", range=[0, max(probs_sorted) * 1.18]),
                yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                margin=dict(l=10, r=60, t=10, b=30),
                height=520,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
            st.plotly_chart(fig, use_container_width=True)

        with col_stats:
            st.markdown('<p class="section-title">Top 5</p>', unsafe_allow_html=True)
            for rank, (name, prob) in enumerate(sorted_pairs[:5], start=1):
                c = label_color(name)
                st.markdown(
                    f'<div style="margin-bottom:0.55rem">'
                    f'<span style="color:{c};font-weight:700">#{rank}</span> '
                    f'<span style="font-size:0.85rem">{name}</span><br>'
                    f'<span style="font-size:0.92rem;font-weight:600">{prob*100:.2f}%</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            word_count  = len(input_text.split())
            clean_count = len(cleaned.split())
            st.markdown(
                f'<p class="section-title">Input stats</p>'
                f'<span style="font-size:0.85rem">Raw words: <b>{word_count}</b><br>'
                f'After cleaning: <b>{clean_count}</b><br>'
                f'Tokens fed: <b>{min(clean_count + 1, spec["max_len"])}</b></span>',
                unsafe_allow_html=True,
            )

        # ── Cleaned input preview (collapsed) ─────────────────────────────────
        with st.expander("🔎 View cleaned input text"):
            st.code(cleaned, language=None)
