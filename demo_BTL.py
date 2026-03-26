"""
Streamlit demo for the 20 Newsgroups Transformer text classifier.

Prerequisites (run the notebook cells in order first):
  - model_meta.json   (saved by Cell 7b)
  - best_transformer.pt  (saved by Cell 8 after training)

Launch:
  streamlit run demo_BTL.py
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
    page_title="20 Newsgroups Classifier",
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
# Full Transformer architecture  (mirrors the notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k    = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-1e9"))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        Q = self._split_heads(self.W_Q(Q))
        K = self._split_heads(self.W_K(K))
        V = self._split_heads(self.W_V(V))
        context, _ = scaled_dot_product_attention(Q, K, V, mask)
        context = context.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.dropout(self.W_O(context))


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff,    d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attn(normed, normed, normed, mask))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=128, num_heads=4,
                 num_layers=3, d_ff=512, max_len=128, dropout=0.0, pad_idx=0):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc    = PositionalEncoding(d_model, max_len, dropout)
        self.encoder    = TransformerEncoder(d_model, num_heads, num_layers, d_ff, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def make_padding_mask(self, input_ids, pad_idx=0):
        return (input_ids != pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(self, input_ids):
        mask     = self.make_padding_mask(input_ids).to(input_ids.device)
        x        = self.pos_enc(self.embedding(input_ids))
        x        = self.encoder(x, mask)
        cls_repr = x[:, 0, :]
        return self.classifier(cls_repr)


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
# Cached resources  (loaded once, reused across every user interaction)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model_and_meta():
    with open("model_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    cfg   = meta["model_config"]
    model = TransformerClassifier(**cfg)
    state = torch.load("best_transformer.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, meta["word2idx"], meta["label_names"], cfg


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(text: str, model, word2idx: dict, max_len: int, label_names: list):
    clean  = preprocess_text(text)
    ids    = encode(clean, word2idx, max_len)
    x      = torch.tensor([ids], dtype=torch.long)
    logits = model(x)
    probs  = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    top_idx = probs.index(max(probs))
    return label_names[top_idx], probs[top_idx], probs, clean


# ─────────────────────────────────────────────────────────────────────────────
# Load model (done once)
# ─────────────────────────────────────────────────────────────────────────────
try:
    model, word2idx, label_names, model_cfg = load_model_and_meta()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    missing_file = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# ── Header ────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>📰 20 Newsgroups Classifier</h1>
        <p>Transformer encoder trained from scratch · PyTorch · 20-class text classification</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# ── Sidebar — model card ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Model Card")

    if model_loaded:
        cfg = model_cfg
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

        total_params = sum(p.numel() for p in model.parameters())
        st.metric("Trainable parameters", f"{total_params:,}")

        st.markdown("---")
        st.markdown("### 🏷️ Classes")
        for name in sorted(label_names):
            color = label_color(name)
            st.markdown(
                f'<span class="badge" style="background:{color}22;color:{color}">'
                f"{name}</span>",
                unsafe_allow_html=True,
            )
    else:
        st.error(
            "**Files missing.**\n\n"
            "Please run the notebook cells in order (through Cell 8) first to generate:\n"
            "- `model_meta.json`\n- `best_transformer.pt`"
        )

    st.markdown("---")
    st.markdown(
        "<small>Built with PyTorch + Streamlit<br>"
        "Architecture: Attention Is All You Need (Vaswani et al., 2017)</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# ── Main content ──────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if not model_loaded:
    st.error(
        f"**Could not load model files.**\n\n"
        f"`{missing_file}`\n\n"
        "Run the notebook cells first, then relaunch this app."
    )
    st.stop()

# ── Example selector ─────────────────────────────────────────────────────────
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

default_text = EXAMPLES[selected]

input_text = st.text_area(
    "Text to classify",
    value=default_text,
    height=160,
    placeholder="Paste or type a newsgroup-style post here …",
    label_visibility="collapsed",
)

col_btn, col_spacer = st.columns([1, 5])
with col_btn:
    classify_clicked = st.button("🔍  Classify", type="primary", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if classify_clicked:
    if not input_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Running inference …"):
            pred_label, confidence, all_probs, cleaned = predict(
                input_text, model, word2idx, model_cfg["max_len"], label_names
            )

        color = label_color(pred_label)

        # ── Top prediction card ────────────────────────────────────────────────
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

            sorted_pairs = sorted(
                zip(label_names, all_probs), key=lambda x: x[1], reverse=True
            )
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
            word_count = len(input_text.split())
            clean_count = len(cleaned.split())
            st.markdown(
                f'<p class="section-title">Input stats</p>'
                f'<span style="font-size:0.85rem">Raw words: <b>{word_count}</b><br>'
                f'After cleaning: <b>{clean_count}</b><br>'
                f'Tokens fed: <b>{min(clean_count + 1, model_cfg["max_len"])}</b></span>',
                unsafe_allow_html=True,
            )

        # ── Cleaned input preview (collapsed) ─────────────────────────────────
        with st.expander("🔎 View cleaned input text"):
            st.code(cleaned, language=None)
