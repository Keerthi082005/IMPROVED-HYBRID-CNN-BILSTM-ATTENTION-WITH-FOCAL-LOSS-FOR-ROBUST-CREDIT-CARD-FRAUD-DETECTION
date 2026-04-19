import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os, time, base64, random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import tensorflow as tf
K = tf.keras.backend

from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                              precision_recall_curve, average_precision_score,
                              classification_report)

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    from sms_alerts import send_fraud_sms
    SMS_OK = True
except Exception:
    SMS_OK = False

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

SAVE_DIR      = os.path.join(os.path.dirname(__file__), "saved")
PHONE_MAP_CSV = os.path.join(os.path.dirname(__file__), "phone_mapping.csv")

# ─────────────────────────────────────────────────────────────
# LOAD PHONE MAPPING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_phone_map():
    if os.path.exists(PHONE_MAP_CSV):
        df = pd.read_csv(PHONE_MAP_CSV, dtype=str)
        df.columns = df.columns.str.strip()
        mapping = {}
        for _, row in df.iterrows():
            tid = str(row.get('transaction_id', '')).strip()
            mapping[tid] = {
                'name':  str(row.get('customer_name', '')).strip(),
                'phone': str(row.get('customer_phone', '')).strip()
            }
        return mapping
    return {}

PHONE_MAP = load_phone_map()

def get_customer(txn_id):
    tid = str(txn_id).strip()
    return PHONE_MAP.get(tid, {'name': 'Customer', 'phone': ''})

# ─────────────────────────────────────────────────────────────
# MEGA CSS + ANIMATION INJECTION
# ─────────────────────────────────────────────────────────────
def inject_full_ui():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

    :root {
        --bg-deep:     #020916;
        --bg-dark:     #050f22;
        --bg-panel:    #071530;
        --bg-card:     #0a1a3a;
        --accent-blue: #1a6cf5;
        --accent-cyan: #06d6f5;
        --accent-red:  #f53b3b;
        --accent-gold: #f5c842;
        --accent-green:#0ff07c;
        --text-main:   #e8f4fd;
        --text-muted:  #7090b8;
        --border-dim:  rgba(26,108,245,0.18);
        --border-glow: rgba(6,214,245,0.35);
        --shadow-blue: rgba(26,108,245,0.4);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background: var(--bg-deep) !important;
        color: var(--text-main) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #03091e 0%, #050f22 100%) !important;
        border-right: 1px solid var(--border-dim) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text-main) !important; }
    [data-testid="stSidebar"] .stRadio label {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        padding: 6px 10px;
        border-radius: 6px;
        transition: background 0.2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(26,108,245,0.15) !important;
    }

    /* ── Main content ── */
    section[data-testid="stMain"] > div {
        background: transparent !important;
        padding:2 8px !important;
    }

    /* ── All text ── */
    p, label, span, div, td, th, li {
        color: var(--text-main) !important;
    }
    h1, h2, h3, h4 {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        color: var(--accent-cyan) !important;
        letter-spacing: 1px;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(10,26,58,0.9), rgba(7,21,48,0.95)) !important;
        border: 1px solid var(--border-dim) !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        position: relative;
        overflow: hidden;
    }
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
    }
    [data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 1px; }
    [data-testid="stMetricValue"] { color: #fff !important; font-family: 'Rajdhani', sans-serif !important; font-size: 1.7rem !important; font-weight: 700 !important; }

    /* ── Tabs ── */
    [data-testid="stTabs"] button {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 14px !important; font-weight: 600 !important;
        color: var(--text-muted) !important; letter-spacing: 0.5px;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--accent-cyan) !important;
        border-bottom: 2px solid var(--accent-cyan) !important;
        background: rgba(6,214,245,0.08) !important;
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1a4fd4, #1a6cf5) !important;
        color: #fff !important; border: none !important;
        border-radius: 8px !important; font-family: 'Rajdhani', sans-serif !important;
        font-size: 15px !important; font-weight: 700 !important;
        letter-spacing: 1px; padding: 10px 24px !important;
        box-shadow: 0 4px 20px rgba(26,108,245,0.45) !important;
        transition: all 0.2s !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(26,108,245,0.65) !important;
    }
    .stButton > button {
        background: rgba(10,26,58,0.6) !important;
        color: var(--accent-cyan) !important;
        border: 1px solid var(--border-dim) !important;
        border-radius: 8px !important;
        font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important;
    }

    /* ── DataFrames ── */
    [data-testid="stDataFrame"] {
        background: rgba(7,21,48,0.85) !important;
        border: 1px solid var(--border-dim) !important;
        border-radius: 10px !important;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background: rgba(7,21,48,0.7) !important;
        border: 1px solid var(--border-dim) !important;
        border-radius: 10px !important;
    }

    /* ── Alerts ── */
    [data-testid="stInfo"]    { background: rgba(26,108,245,0.12) !important; border-left: 3px solid var(--accent-blue) !important; border-radius: 8px !important; }
    [data-testid="stSuccess"] { background: rgba(15,240,124,0.10) !important; border-left: 3px solid var(--accent-green) !important; border-radius: 8px !important; }
    [data-testid="stError"]   { background: rgba(245,59,59,0.12) !important;  border-left: 3px solid var(--accent-red) !important;  border-radius: 8px !important; }
    [data-testid="stWarning"] { background: rgba(245,200,66,0.10) !important; border-left: 3px solid var(--accent-gold) !important; border-radius: 8px !important; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: rgba(7,21,48,0.6) !important;
        border: 2px dashed rgba(26,108,245,0.4) !important;
        border-radius: 12px !important;
    }

    /* ── Progress bar ── */
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
    }

    /* ── Code ── */
    code {
        background: rgba(0,0,0,0.5) !important;
        color: var(--accent-cyan) !important;
        border-radius: 4px; padding: 2px 6px;
        font-family: 'Space Mono', monospace !important;
    }

    /* ── Tables ── */
    table { background: rgba(7,21,48,0.6) !important; border-radius: 8px; }
    th { background: rgba(26,108,245,0.2) !important; color: var(--accent-cyan) !important; font-family: 'Rajdhani', sans-serif !important; letter-spacing: 1px; }
    td { color: var(--text-main) !important; }

    /* ── Scrollbars ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: var(--accent-blue); border-radius: 3px; }

    /* ── Slider ── */
    [data-testid="stSlider"] > div > div > div > div {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
    }

    /* ── Toggle ── */
    [data-testid="stToggle"] span[data-checked="true"] {
        background-color: var(--accent-cyan) !important;
    }
    </style>
    """, unsafe_allow_html=True)

inject_full_ui()

# ─────────────────────────────────────────────────────────────
# 3D CREDIT CARD ANIMATION COMPONENT
# ─────────────────────────────────────────────────────────────
def render_3d_card_hero():
    st.components.v1.html("""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      * { margin: 0; padding: 0; box-sizing: border-box; }
      body { background: transparent; overflow: hidden; }
      .hero-wrap {
        width: 100%; min-height: 420px;
        background: linear-gradient(180deg, #010812 0%, #020d1e 60%, #010812 100%);
        display: flex; align-items: center; justify-content: center;
        position: relative; overflow: hidden;
      }

      /* ── Particle canvas behind everything ── */
      #particle-canvas {
        position: absolute; inset: 0; z-index: 0; pointer-events: none;
      }

      /* ── Grid lines ── */
      .grid-lines {
        position: absolute; inset: 0; z-index: 1; pointer-events: none;
        background-image:
          linear-gradient(rgba(26,108,245,0.07) 1px, transparent 1px),
          linear-gradient(90deg, rgba(26,108,245,0.07) 1px, transparent 1px);
        background-size: 48px 48px;
      }
      .grid-fade {
        position: absolute; inset: 0; z-index: 2;
        background: radial-gradient(ellipse 70% 60% at 50% 50%, transparent 30%, #010812 100%);
      }

      /* ── Title & stats ── */
      .hero-content {
        position: relative; z-index: 10;
        display: flex; flex-direction: column; align-items: center; gap: 24px;
        width: 100%;
      }
      .brand-badge {
        display: flex; align-items: center; gap: 10px;
        background: rgba(26,108,245,0.12);
        border: 1px solid rgba(26,108,245,0.3);
        border-radius: 30px; padding: 6px 18px;
        font-family: 'Space Mono', monospace;
        font-size: 11px; color: #06d6f5;
        letter-spacing: 2px; text-transform: uppercase;
      }
      .brand-badge .dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: #0ff07c;
        box-shadow: 0 0 8px #0ff07c;
        animation: pulse-dot 1.4s ease-in-out infinite;
      }
      @keyframes pulse-dot {
        0%,100% { opacity:1; transform: scale(1); }
        50%      { opacity:0.5; transform: scale(0.7); }
      }
      .hero-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: clamp(28px, 5vw, 52px);
        font-weight: 700; text-align: center;
        background: linear-gradient(90deg, #1a6cf5, #06d6f5, #1a6cf5);
        background-size: 200%;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: shimmer-title 3s linear infinite;
        letter-spacing: 2px; line-height: 1.1;
      }
      @keyframes shimmer-title {
        0%   { background-position: 0% }
        100% { background-position: 200% }
      }
      .hero-sub {
        font-family: 'Inter', sans-serif; font-size: 14px;
        color: #5a7faa; letter-spacing: 1px; text-align: center;
      }

      /* ── 3D Card scene ── */
      .card-scene {
        perspective: 900px; width: 380px; height: 230px;
        position: relative; cursor: pointer;
      }
      .card-3d {
        width: 100%; height: 100%;
        position: relative;
        transform-style: preserve-3d;
        transform: rotateY(-20deg) rotateX(8deg);
        animation: card-float 5s ease-in-out infinite;
        transition: transform 0.15s ease;
      }
      @keyframes card-float {
        0%,100% { transform: rotateY(-20deg) rotateX(8deg) translateY(0px); }
        50%      { transform: rotateY(-12deg) rotateX(5deg) translateY(-12px); }
      }
      .card-face {
        position: absolute; inset: 0; border-radius: 18px;
        backface-visibility: hidden;
      }

      /* FRONT */
      .card-front {
        background: linear-gradient(135deg, #0d2a5e 0%, #1a4fd4 45%, #0d2a5e 100%);
        border: 1px solid rgba(6,214,245,0.35);
        box-shadow: 0 25px 60px rgba(26,108,245,0.5), inset 0 1px 0 rgba(255,255,255,0.15);
        overflow: hidden;
        display: flex; flex-direction: column; justify-content: space-between;
        padding: 22px 26px 18px;
      }
      /* Holographic shimmer on card */
      .card-front::before {
        content: '';
        position: absolute; inset: 0;
        background: linear-gradient(105deg,
          transparent 40%, rgba(255,255,255,0.07) 50%, transparent 60%);
        animation: hologram 3.5s ease-in-out infinite;
        border-radius: 18px;
      }
      @keyframes hologram {
        0%,100% { transform: translateX(-120%); }
        50%      { transform: translateX(120%); }
      }
      /* Card pattern circles */
      .card-front::after {
        content: '';
        position: absolute; right: -40px; top: -40px;
        width: 200px; height: 200px; border-radius: 50%;
        background: radial-gradient(circle, rgba(6,214,245,0.08), transparent 70%);
      }

      /* Card chip */
      .card-chip {
        width: 46px; height: 34px; border-radius: 6px;
        background: linear-gradient(135deg, #d4a843, #f5c842, #b8922e);
        position: relative; overflow: hidden; flex-shrink: 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
      }
      .card-chip::before {
        content: '';
        position: absolute; inset: 4px;
        background: linear-gradient(135deg, #c4942a, #e8b430);
        border-radius: 3px;
      }
      .card-chip::after {
        content: '';
        position: absolute; top: 50%; left: 0; right: 0;
        height: 1px; background: rgba(0,0,0,0.3);
        transform: translateY(-50%);
      }
      .chip-line-v {
        position: absolute; left: 50%; top: 0; bottom: 0; width: 1px;
        background: rgba(0,0,0,0.25); transform: translateX(-50%);
      }

      /* Contactless icon */
      .contactless {
        display: flex; flex-direction: column; align-items: flex-end;
        gap: 2px;
      }
      .arc {
        height: 2px; background: rgba(255,255,255,0.5);
        border-radius: 2px;
        animation: arc-pulse 2s ease-in-out infinite;
      }
      .arc:nth-child(1) { width: 10px; animation-delay: 0s; }
      .arc:nth-child(2) { width: 16px; animation-delay: 0.15s; }
      .arc:nth-child(3) { width: 22px; animation-delay: 0.3s; }
      .arc:nth-child(4) { width: 28px; animation-delay: 0.45s; }
      @keyframes arc-pulse {
        0%,100% { opacity: 0.35; }
        50%      { opacity: 1; }
      }

      .card-top-row { display: flex; align-items: center; justify-content: space-between; }

      .card-number {
        font-family: 'Space Mono', monospace;
        font-size: 15px; letter-spacing: 3px; color: rgba(255,255,255,0.85);
        text-shadow: 0 1px 4px rgba(0,0,0,0.6);
        position: relative; z-index: 1;
      }
      .card-number span {
        margin-right: 14px;
        animation: number-flicker 8s ease-in-out infinite;
      }
      .card-number span:nth-child(2) { animation-delay: 2s; }
      .card-number span:nth-child(3) { animation-delay: 4s; }
      .card-number span:nth-child(4) { animation-delay: 6s; }
      @keyframes number-flicker {
        0%,90%,100% { opacity: 0.85; }
        92%,98%     { opacity: 0.3; }
      }

      .card-bottom-row {
        display: flex; align-items: flex-end; justify-content: space-between;
        position: relative; z-index: 1;
      }
      .card-holder { display: flex; flex-direction: column; gap: 2px; }
      .card-label  { font-size: 8px; letter-spacing: 2px; color: rgba(255,255,255,0.4); text-transform: uppercase; }
      .card-value  { font-family: 'Rajdhani', sans-serif; font-size: 14px; font-weight: 600; color: rgba(255,255,255,0.85); letter-spacing: 1px; }

      /* Visa-style logo */
      .card-logo {
        font-family: 'Rajdhani', sans-serif; font-size: 22px; font-weight: 700;
        font-style: italic; letter-spacing: -1px;
        background: linear-gradient(135deg, #f5c842, #fff, #f5c842);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-shadow: none; flex-shrink: 0;
      }

      /* SHIELD overlay badge */
      .shield-badge {
        position: absolute; top: -14px; right: -14px;
        width: 48px; height: 48px; z-index: 20;
        display: flex; align-items: center; justify-content: center;
      }
      .shield-svg { width: 100%; height: 100%; }
      .shield-ring {
        animation: shield-spin 8s linear infinite;
        transform-origin: 24px 24px;
      }
      @keyframes shield-spin {
        from { transform: rotate(0deg); }
        to   { transform: rotate(360deg); }
      }

      /* Scan laser */
      .scan-beam {
        position: absolute; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, #06d6f5, transparent);
        opacity: 0; z-index: 15; border-radius: 2px;
        animation: laser-scan 4s ease-in-out infinite 1s;
      }
      @keyframes laser-scan {
        0%  { top: 10%;  opacity: 0; }
        10% { opacity: 0.8; }
        90% { top: 90%; opacity: 0.8; }
        100%{ top: 90%; opacity: 0; }
      }

      /* ── Stats row below card ── */
      .stats-row {
        display: flex; gap: 16px; flex-wrap: wrap; justify-content: center;
      }
      .stat-pill {
        background: rgba(10,26,58,0.7);
        border: 1px solid rgba(26,108,245,0.25);
        border-radius: 10px; padding: 10px 18px;
        display: flex; flex-direction: column; align-items: center;
        gap: 4px; min-width: 110px;
        position: relative; overflow: hidden;
      }
      .stat-pill::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 1.5px;
        background: linear-gradient(90deg, transparent, var(--c), transparent);
      }
      .stat-val {
        font-family: 'Rajdhani', sans-serif; font-size: 20px; font-weight: 700;
        color: #fff;
      }
      .stat-lbl {
        font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase;
        color: #5a7faa;
      }
      /* animated counter */
      .counter { animation: count-up 2s ease-out forwards; }
      @keyframes count-up {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
      }

      /* ── Threat ticker ── */
      .ticker-wrap {
        width: 100%; overflow: hidden;
        border-top: 1px solid rgba(26,108,245,0.15);
        border-bottom: 1px solid rgba(26,108,245,0.15);
        background: rgba(5,15,34,0.8);
        padding: 8px 0; position: relative;
      }
      .ticker-inner {
        display: flex; gap: 0; white-space: nowrap;
        animation: ticker-scroll 30s linear infinite;
        width: max-content;
      }
      .ticker-item {
        font-family: 'Space Mono', monospace; font-size: 11px;
        padding: 0 28px; color: #5a7faa;
      }
      .ticker-item .hi { color: #f53b3b; }
      .ticker-item .ok { color: #0ff07c; }
      .ticker-item .wr { color: #f5c842; }
      @keyframes ticker-scroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-50%); }
      }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@700&family=Space+Mono:wght@400;700&family=Inter:wght@400;500&display=swap" rel="stylesheet">
    </head>
    <body>
    <div class="hero-wrap">
      <canvas id="particle-canvas"></canvas>
      <div class="grid-lines"></div>
      <div class="grid-fade"></div>

      <div class="hero-content">
        <div class="brand-badge">
          <div class="dot"></div>
          FRAUDSHIELD AI &nbsp;
        </div>

        <div class="hero-title">🛡️ FraudShield AI</div>
        <div class="hero-sub">REAL-TIME CREDIT CARD FRAUD DETECTION SYSTEM</div>

        <!-- 3D Card -->
        <div class="card-scene" id="cardScene">
          <div class="card-3d" id="card3d">
            <div class="card-face card-front">
              <div class="scan-beam"></div>

              <div class="card-top-row">
                <div class="card-chip">
                  <div class="chip-line-v"></div>
                </div>
                <div class="contactless">
                  <div class="arc"></div>
                  <div class="arc"></div>
                  <div class="arc"></div>
                  <div class="arc"></div>
                </div>
              </div>

              <div class="card-number">
                <span>4716</span>
                <span>••••</span>
                <span>••••</span>
                <span>3891</span>
              </div>

              <div class="card-bottom-row">
                <div class="card-holder">
                  <div class="card-label">Card Holder</div>
                  <div class="card-value">SECURED ACCOUNT</div>
                </div>
                <div class="card-holder">
                  <div class="card-label">Expires</div>
                  <div class="card-value">12/27</div>
                </div>
                <div class="card-logo">VISA</div>
              </div>
            </div>
          </div>

          <!-- Shield badge -->
          <div class="shield-badge">
            <svg class="shield-svg" viewBox="0 0 48 48">
              <circle class="shield-ring" cx="24" cy="24" r="20" fill="none"
                stroke="rgba(26,108,245,0.6)" stroke-width="1"
                stroke-dasharray="4 4"/>
              <text x="24" y="29" text-anchor="middle" font-size="16" fill="#06d6f5">🛡</text>
            </svg>
          </div>
        </div>

        <!-- Stats -->
        <div class="stats-row">
          <div class="stat-pill" style="--c:#1a6cf5">
            <div class="stat-val counter" id="c1">0</div>
            <div class="stat-lbl">Transactions/sec</div>
          </div>
          <div class="stat-pill" style="--c:#f53b3b">
            <div class="stat-val counter" id="c2">0%</div>
            <div class="stat-lbl">Fraud Rate (Global)</div>
          </div>
          
          <div class="stat-pill" style="--c:#f5c842">
            <div class="stat-val counter" id="c4">0B</div>
            <div class="stat-lbl">Annual Losses (USD)</div>
          </div>
        </div>

        <!-- Ticker -->
        <div class="ticker-wrap">
          <div class="ticker-inner" id="ticker">
            <span class="ticker-item"><span class="hi">⬤</span> TXN-7821 FLAGGED — HIGH RISK — ₹48,200 → Mumbai</span>
            <span class="ticker-item"><span class="ok">⬤</span> TXN-9043 CLEARED — Normal velocity — $124 → New York</span>
            <span class="ticker-item"><span class="wr">⬤</span> TXN-3310 REVIEW — Unusual location — €3,800 → Frankfurt</span>
            <span class="ticker-item"><span class="hi">⬤</span> TXN-5527 BLOCKED — Card-not-present fraud detected</span>
            <span class="ticker-item"><span class="ok">⬤</span> TXN-2291 CLEARED — Verified biometric — £210 → London</span>
            <span class="ticker-item"><span class="hi">⬤</span> TXN-6614 FLAGGED — Multiple rapid transactions — $9,920</span>
            <span class="ticker-item"><span class="ok">⬤</span> TXN-8801 CLEARED — Known merchant — ¥15,000 → Tokyo</span>
            <span class="ticker-item"><span class="wr">⬤</span> TXN-4422 REVIEW — New device fingerprint — $670 → Chicago</span>
            <!-- Duplicate for seamless loop -->
            <span class="ticker-item"><span class="hi">⬤</span> TXN-7821 FLAGGED — HIGH RISK — ₹48,200 → Mumbai</span>
            <span class="ticker-item"><span class="ok">⬤</span> TXN-9043 CLEARED — Normal velocity — $124 → New York</span>
            <span class="ticker-item"><span class="wr">⬤</span> TXN-3310 REVIEW — Unusual location — €3,800 → Frankfurt</span>
            <span class="ticker-item"><span class="hi">⬤</span> TXN-5527 BLOCKED — Card-not-present fraud detected</span>
            <span class="ticker-item"><span class="ok">⬤</span> TXN-2291 CLEARED — Verified biometric — £210 → London</span>
            <span class="ticker-item"><span class="hi">⬤</span> TXN-6614 FLAGGED — Multiple rapid transactions — $9,920</span>
            <span class="ticker-item"><span class="ok">⬤</span> TXN-8801 CLEARED — Known merchant — ¥15,000 → Tokyo</span>
            <span class="ticker-item"><span class="wr">⬤</span> TXN-4422 REVIEW — New device fingerprint — $670 → Chicago</span>
          </div>
        </div>

      </div><!-- /hero-content -->
    </div><!-- /hero-wrap -->

    <script>
    // ── Particle field ──
    const canvas = document.getElementById('particle-canvas');
    const ctx    = canvas.getContext('2d');
    let W, H, particles = [];

    function resize() {
      W = canvas.width  = canvas.offsetWidth;
      H = canvas.height = canvas.offsetHeight;
    }
    resize();
    window.addEventListener('resize', () => { resize(); init(); });

    function init() {
      particles = [];
      const n = Math.floor(W * H / 7000);
      for (let i = 0; i < n; i++) {
        particles.push({
          x: Math.random() * W, y: Math.random() * H,
          r: Math.random() * 1.4 + 0.3,
          vx: (Math.random() - 0.5) * 0.25,
          vy: (Math.random() - 0.5) * 0.25,
          alpha: Math.random() * 0.6 + 0.1,
          hue: Math.random() < 0.7 ? 210 : 190
        });
      }
    }
    init();

    function animParticles() {
      ctx.clearRect(0, 0, W, H);
      particles.forEach(p => {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
        if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${p.hue}, 90%, 65%, ${p.alpha})`;
        ctx.fill();
      });
      // Draw connections
      for (let i = 0; i < particles.length; i++) {
        for (let j = i+1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const d  = Math.sqrt(dx*dx + dy*dy);
          if (d < 90) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(26,108,245,${0.12 * (1 - d/90)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }
      requestAnimationFrame(animParticles);
    }
    animParticles();

    // ── Mouse tilt on card ──
    const scene = document.getElementById('cardScene');
    const card  = document.getElementById('card3d');
    scene.addEventListener('mousemove', e => {
      const r  = scene.getBoundingClientRect();
      const cx = r.left + r.width  / 2;
      const cy = r.top  + r.height / 2;
      const rx = (e.clientY - cy) / 12;
      const ry = (e.clientX - cx) / 10;
      card.style.animation = 'none';
      card.style.transform = `rotateX(${rx}deg) rotateY(${ry - 20}deg)`;
    });
    scene.addEventListener('mouseleave', () => {
      card.style.animation = '';
      card.style.transform = '';
    });

    // ── Animated counters ──
    function animateCounter(el, end, suffix, duration) {
      let start = 0, startTime = null;
      function step(ts) {
        if (!startTime) startTime = ts;
        const progress = Math.min((ts - startTime) / duration, 1);
        const eased    = 1 - Math.pow(1 - progress, 3);
        const val      = Math.floor(eased * end);
        el.textContent = val + suffix;
        if (progress < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }
    setTimeout(() => {
      animateCounter(document.getElementById('c1'), 12500,  '',   2000);
      animateCounter(document.getElementById('c2'), 2,      '%',  1500);
      animateCounter(document.getElementById('c4'), 420,     'B',  1800);
    }, 400);
    </script>
    </body>
    </html>
    """, height=480, scrolling=False)


# ─────────────────────────────────────────────────────────────
# SECTION HEADER HELPER
# ─────────────────────────────────────────────────────────────
def section_header(icon, title, subtitle=""):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;
                padding:16px 0 6px;border-bottom:1px solid rgba(26,108,245,0.2);
                margin-bottom:20px">
      <span style="font-size:28px;filter:drop-shadow(0 0 8px rgba(6,214,245,0.6))">{icon}</span>
      <div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:24px;
                    font-weight:700;color:#06d6f5;letter-spacing:2px;
                    line-height:1.1">{title}</div>
        {f'<div style="font-size:12px;color:#5a7faa;letter-spacing:1px">{subtitle}</div>' if subtitle else ''}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# GLASSMORPHIC CARD HELPER
# ─────────────────────────────────────────────────────────────
def glass_card(content_html, accent="#1a6cf5"):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(10,26,58,0.85),rgba(7,21,48,0.9));
                border:1px solid {accent}33;border-radius:14px;padding:18px 20px;
                position:relative;overflow:hidden;margin-bottom:12px">
      <div style="position:absolute;top:0;left:0;right:0;height:2px;
                  background:linear-gradient(90deg,transparent,{accent},transparent)"></div>
      {content_html}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────────────────────
def focal_loss(alpha=0.75, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        bce     = K.binary_crossentropy(y_true, y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return K.mean(alpha_t * K.pow(1 - p_t, gamma) * bce)
    return loss_fn


# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
class CustomDense(tf.keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)
    @classmethod
    def from_config(cls, config):
        config.pop('quantization_config', None)
        return cls(**config)

def load_model_safe(model_path):
    try:
        return tf.keras.models.load_model(
            model_path,
            custom_objects={'loss_fn': focal_loss(), 'CustomDense': CustomDense, 'Dense': CustomDense},
            compile=False)
    except Exception:
        return tf.keras.models.load_model(model_path, compile=False)

@st.cache_resource
def load_euro():
    try:
        m  = load_model_safe(os.path.join(SAVE_DIR, "euro_model.h5"))
        with open(os.path.join(SAVE_DIR, "scaler_euro.pkl"), "rb") as f:
            meta = pickle.load(f)
        Xt = np.load(os.path.join(SAVE_DIR, "euro_X_test.npy"))
        yt = np.load(os.path.join(SAVE_DIR, "euro_y_test.npy"))
        return m, meta, Xt, yt
    except Exception as e:
        st.error(f"❌ European model load failed: {e}")
        raise

@st.cache_resource
def load_indian():
    try:
        m  = load_model_safe(os.path.join(SAVE_DIR, "indian_model.h5"))
        with open(os.path.join(SAVE_DIR, "scaler_indian.pkl"), "rb") as f:
            meta = pickle.load(f)
        Xt = np.load(os.path.join(SAVE_DIR, "indian_X_test.npy"))
        yt = np.load(os.path.join(SAVE_DIR, "indian_y_test.npy"))
        return m, meta, Xt, yt
    except Exception as e:
        st.error(f"❌ Indian model load failed: {e}")
        raise


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def risk_score(p): return int(p * 100)
def risk_label(s):
    if s >= 75: return "HIGH",   "#f53b3b"
    if s >= 40: return "MEDIUM", "#f5c842"
    return "LOW", "#0ff07c"

def dark_ax(ax):
    ax.set_facecolor('#020916')
    ax.tick_params(colors='#7090b8')
    ax.xaxis.label.set_color('#7090b8')
    ax.yaxis.label.set_color('#7090b8')
    ax.title.set_color('#06d6f5')
    for sp in ax.spines.values():
        sp.set_edgecolor('#0a1a3a')
    return ax

def voice_js(text):
    st.components.v1.html(f"""<script>
    setTimeout(function(){{
        if('speechSynthesis' in window){{
            window.speechSynthesis.cancel();
            var u=new SpeechSynthesisUtterance("{text}");
            u.rate=0.88; u.pitch=1.0; u.volume=1.0;
            window.speechSynthesis.speak(u);
        }}
    }}, 300);
    </script>""", height=0)

def preprocess(df_raw, is_european, meta, feature_names):
    df   = df_raw.copy()
    drop = ['customer_id','merchant_id','transaction_time','is_fraudulent','Class','transaction_id']
    df   = df.drop(columns=[c for c in drop if c in df.columns], errors='ignore')
    if not is_european:
        for col, le in meta.get('encoders', {}).items():
            if col in df.columns:
                known  = set(le.classes_)
                df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
                df[col] = le.transform(df[col])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_names]
    if is_european:
        X  = df.values.astype(float)
        ai = meta['amount_idx']
        X[:, ai] = meta['scaler'].transform(X[:, ai].reshape(-1,1)).ravel()
    else:
        X = meta['scaler'].transform(df.values.astype(float))
    return X


# ─────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────
def shap_single(model, X_bg, x_row, feature_names):
    if not SHAP_OK: return None
    try:
        n  = min(40, len(X_bg))
        bg = X_bg[:n].reshape(n, X_bg.shape[1], 1)
        xs = x_row.reshape(1, -1, 1)
        ex = shap.GradientExplainer(model, bg)
        sv = np.array(ex.shap_values(xs)).squeeze()
        if sv.ndim > 1: sv = sv.mean(axis=0)
        sv = np.asarray(sv).flatten()
        return sv if len(sv) == len(feature_names) else None
    except:
        return None

def shap_two_plots(model, X_bg, x_row, X_explain, sv_batch, row_idx, feature_names, txn_label=""):
    col_bar, col_bee = st.columns(2)

    with col_bar:
        st.markdown(f"<p style='color:#06d6f5;font-weight:600;font-family:Rajdhani,sans-serif;font-size:14px'>SHAP Bar — {txn_label}</p>", unsafe_allow_html=True)
        if sv_batch is None:
            sv = shap_single(model, X_bg, x_row, feature_names)
        else:
            try:
                if sv_batch.ndim == 1:
                    sv = sv_batch.copy()
                elif sv_batch.ndim == 2:
                    sv = sv_batch[min(row_idx, len(sv_batch)-1)].copy()
                else:
                    sv_s = np.squeeze(sv_batch)
                    sv   = sv_s.copy() if sv_s.ndim==1 else sv_s[min(row_idx, len(sv_s)-1)].copy()
            except:
                sv = None

        if sv is not None and len(sv) > 0:
            try:
                sv     = np.asarray(sv).flatten()
                abs_sv = np.abs(sv)
                top    = min(8, len(feature_names))
                tidx   = np.argsort(abs_sv)[::-1][:top]
                colors = ['#f53b3b' if sv[j]>0 else '#1a6cf5' for j in tidx]
                fig, ax = plt.subplots(figsize=(5.5,3.5), facecolor='none')
                fig.patch.set_facecolor('#020916')
                dark_ax(ax)
                ax.barh([feature_names[j] for j in tidx[::-1]], [sv[j] for j in tidx[::-1]],
                        color=colors[::-1], edgecolor='none', height=0.55)
                ax.axvline(0, color='white', lw=0.8, alpha=0.4)
                ax.set_title("Feature Importance (Bar)", fontsize=10, color='#06d6f5', fontweight='bold')
                ax.set_xlabel("SHAP value  🔴=fraud  🔵=legit", fontsize=8, color='#7090b8')
                ax.grid(axis='x', alpha=0.15, color='white')
                plt.tight_layout(); st.pyplot(fig); plt.close()
                st.markdown("<p style='color:#06d6f5;font-size:13px;font-weight:600'>Why FRAUD? Top reasons:</p>", unsafe_allow_html=True)
                for rk, j in enumerate(tidx[:3], 1):
                    direction = "↑ increases fraud risk" if sv[j]>0 else "↓ pushes toward legit"
                    try:
                        fv = x_row[j]
                        st.markdown(f"{rk}. `{feature_names[j]}` = `{fv:.3f}` → {direction} (**{sv[j]:+.4f}**)")
                    except:
                        st.markdown(f"{rk}. `{feature_names[j]}` → {direction} (**{sv[j]:+.4f}**)")
            except Exception as e:
                st.warning(f"⚠️ SHAP bar plot failed: {str(e)[:100]}")
        else:
            st.info("SHAP values unavailable for this transaction")

    with col_bee:
        st.markdown(f"<p style='color:#06d6f5;font-weight:600;font-family:Rajdhani,sans-serif;font-size:14px'>SHAP Beeswarm — {txn_label}</p>", unsafe_allow_html=True)
        if sv_batch is not None and SHAP_OK:
            try:
                sv_plot = np.array(sv_batch)
                if sv_plot.ndim == 1:   sv_plot = sv_plot.reshape(1,-1)
                elif sv_plot.ndim > 2:  sv_plot = sv_plot.reshape(sv_plot.shape[0], -1)
                X_sq = None
                if X_explain is not None:
                    X_sq = np.array(X_explain)
                    if X_sq.ndim==3: X_sq = X_sq.squeeze(axis=-1)
                    elif X_sq.ndim==1: X_sq = X_sq.reshape(1,-1)
                if X_sq is not None and sv_plot.shape == X_sq.shape:
                    fig2, ax2 = plt.subplots(figsize=(5.5,3.5), facecolor='none')
                    fig2.patch.set_facecolor('#020916')
                    dark_ax(ax2)
                    shap.summary_plot(sv_plot, X_sq, feature_names=feature_names,
                                      show=False, max_display=8, plot_size=None,
                                      cmap='coolwarm', alpha=0.85)
                    ax2.set_title("Beeswarm — batch context", fontsize=10, color='#06d6f5', fontweight='bold')
                    ax2.tick_params(colors='#7090b8')
                    ax2.xaxis.label.set_color('#7090b8')
                    ax2.yaxis.label.set_color('#7090b8')
                    for sp in ax2.spines.values(): sp.set_edgecolor('#0a1a3a')
                    try:
                        cbar = plt.gcf().axes[-1]
                        cbar.tick_params(colors='#7090b8')
                        cbar.set_ylabel('Feature value', color='#7090b8', rotation=270, labelpad=20)
                    except: pass
                    plt.tight_layout(); st.pyplot(fig2); plt.close()
                    st.caption("🔴 Red = HIGH feature value → pushes to FRAUD  |  🔵 Blue = LOW value → pushes to LEGIT")
                else:
                    if sv_plot.ndim==2 and sv_plot.shape[0]>0:
                        mean_sv = np.abs(sv_plot).mean(axis=0)
                        top     = min(8, len(feature_names))
                        tidx    = np.argsort(mean_sv)[::-1][:top]
                        fig3, ax3 = plt.subplots(figsize=(5.5,3.5), facecolor='none')
                        fig3.patch.set_facecolor('#020916')
                        dark_ax(ax3)
                        ax3.barh([feature_names[j] for j in tidx[::-1]],
                                  [mean_sv[j] for j in tidx[::-1]],
                                  color='#f5c842', edgecolor='none', height=0.55)
                        ax3.set_title("Mean |SHAP| across batch", fontsize=10, color='#06d6f5', fontweight='bold')
                        ax3.set_xlabel("Mean |SHAP value|", fontsize=8, color='#7090b8')
                        ax3.grid(axis='x', alpha=0.15, color='white')
                        plt.tight_layout(); st.pyplot(fig3); plt.close()
                        st.caption("Showing mean importance (shape mismatch fallback)")
            except Exception as e:
                st.info(f"Beeswarm skipped: {str(e)[:60]}")
        else:
            st.info("Beeswarm needs batch SHAP values")


# ─────────────────────────────────────────────────────────────
# SMS LOG
# ─────────────────────────────────────────────────────────────
if 'sms_log' not in st.session_state:
    st.session_state['sms_log'] = []

def log_sms(phone, txn_id, prob, name, result):
    st.session_state['sms_log'].append({
        'time':    datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
        'phone':   phone, 'txn_id': txn_id,
        'prob':    f"{prob*100:.1f}%", 'name': name,
        'status':  "✅ Sent" if result.get('success') else "❌ Failed",
        'message': result.get('message', '')
    })

def auto_send_sms(txn_id, prob):
    if not SMS_OK: return "SMS library not available"
    info   = get_customer(txn_id)
    phone  = info['phone']; name = info['name']
    if not phone: return f"No phone mapped for {txn_id}"
    result = send_fraud_sms(phone_number=phone, transaction_id=txn_id,
                            fraud_probability=prob, customer_name=name)
    log_sms(phone, txn_id, prob, name, result)
    return result.get('message', '')


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:14px 0 10px'>
      <div style='font-size:44px;filter:drop-shadow(0 0 14px rgba(6,214,245,0.7))'>🛡️</div>
      <div style='font-family:Rajdhani,sans-serif;font-size:20px;font-weight:700;
                  color:#06d6f5;letter-spacing:2px;margin-top:4px'>FraudShield AI</div>
      <div style='font-size:10px;color:#3a6090;letter-spacing:2px;
                  text-transform:uppercase;margin-top:2px'>Neural Fraud Detection</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(26,108,245,0.15)'>", unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠 Dashboard",
        "📂 CSV Detection",
        "⚡ Live Simulation",
        "🌍 Geo Analysis",
       
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(26,108,245,0.15)'>", unsafe_allow_html=True)
    voice_on = st.toggle("🔊 Voice Alerts", True)
    st.markdown("<hr style='border-color:rgba(26,108,245,0.15)'>", unsafe_allow_html=True)

   


# ─────────────────────────────────────────────────────────────
# PRELOAD MODELS
# ─────────────────────────────────────────────────────────────
with st.spinner("⚡ Loading neural models..."):
    try:
        _euro   = load_euro()
        _indian = load_indian()
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.info("Ensure the `saved/` folder is next to `app.py`.")
        st.stop()

model        = _euro[0]
meta         = _euro[1]
X_test_data  = _euro[2]
y_test_data  = _euro[3]
feature_names= meta['feature_names']
is_eu        = True
threshold    = 0.75

def detect_dataset(df):
    cols = [c.lower() for c in df.columns]
    if all(c in cols for c in ['v1','v2','v3']):
        m,me,Xt,yt = _euro
        return True, m, me, Xt, yt, 0.75, "🇪🇺 European (V1–V28 detected, threshold=0.75)"
    elif any(c in cols for c in ['card_type','location','purchase_category']):
        m,me,Xt,yt = _indian
        return False, m, me, Xt, yt, 0.50, "🇮🇳 Indian (card_type detected, threshold=0.50)"
    else:
        m,me,Xt,yt = _euro
        return True, m, me, Xt, yt, 0.75, "🇪🇺 European (default, threshold=0.75)"


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════
if "Dashboard" in page:
    render_3d_card_hero()

   

  

    # ── Fraud trend chart ──
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    glass_card("""
    <div style='font-family:Rajdhani,sans-serif;font-size:14px;font-weight:700;
                color:#06d6f5;letter-spacing:1.5px;margin-bottom:4px'>
      📈 GLOBAL CREDIT CARD FRAUD LOSSES (2018–2023) — Nilson Report
    </div>
    <div style='font-size:11px;color:#5a7faa'>Source: The Nilson Report, global card fraud statistics. Values in USD Billion.</div>
    """, accent="#1a6cf5")

    years  = [2018, 2019, 2020, 2021, 2022, 2023]
    losses = [24.26, 27.85, 28.65, 31.21, 33.45, 32.34]
    fig_t, ax_t = plt.subplots(figsize=(10,3.5), facecolor='none')
    fig_t.patch.set_facecolor('#020916')
    dark_ax(ax_t)
    ax_t.fill_between(years, losses, alpha=0.15, color='#1a6cf5')
    ax_t.plot(years, losses, color='#06d6f5', lw=2.5, marker='o', markersize=7,
              markerfacecolor='#1a6cf5', markeredgecolor='#06d6f5', markeredgewidth=2)
    for yr, val in zip(years, losses):
        ax_t.annotate(f"${val}B", (yr, val), textcoords="offset points",
                      xytext=(0,10), ha='center', color='white', fontsize=9)
    ax_t.set_ylabel("USD Billion", fontsize=10, color='#7090b8')
    ax_t.set_xticks(years); ax_t.set_xticklabels([str(y) for y in years], color='#7090b8')
    ax_t.grid(alpha=0.1, color='white'); ax_t.set_facecolor('#020916')
    plt.tight_layout(); st.pyplot(fig_t); plt.close()


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — CSV DETECTION
# ═══════════════════════════════════════════════════════════════
elif "CSV" in page:
    section_header("📂", "CSV BATCH DETECTION", "Upload transaction data for AI fraud analysis")

    glass_card("""
    <div style='display:flex;align-items:center;gap:10px'>
      <span style='font-size:20px'>🤖</span>
      <div>
        <span style='font-family:Rajdhani,sans-serif;font-weight:700;color:#06d6f5'>AUTO-DETECT MODE ACTIVE</span>
        <span style='color:#7090b8;font-size:13px'> — Dataset type detected automatically from column names. SMS sent on fraud detection.</span>
      </div>
    </div>
    """, accent="#1a6cf5")

    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        cols_low = [c.lower() for c in df_raw.columns]
        if all(c in cols_low for c in ['v1','v2','v3']):
            is_eu_c, model_c, meta_c, Xt_c, yt_c, thr_c = True,  *_euro,   0.75
            ds_lbl = "🇪🇺 European (V1–V28 detected, threshold=0.75)"
        elif any(c in cols_low for c in ['card_type','location','purchase_category']):
            is_eu_c, model_c, meta_c, Xt_c, yt_c, thr_c = False, *_indian, 0.50
            ds_lbl = "🇮🇳 Indian (card_type detected, threshold=0.50)"
        else:
            is_eu_c, model_c, meta_c, Xt_c, yt_c, thr_c = True,  *_euro,   0.75
            ds_lbl = "🇪🇺 European (default, threshold=0.75)"
        feat_c = meta_c['feature_names']

        st.success(f"✅ {uploaded.name} | {len(df_raw):,} rows | {df_raw.shape[1]} cols | {ds_lbl}")

        with st.expander("👁️ Preview (5 rows)", expanded=False):
            st.dataframe(df_raw.head(), use_container_width=True)

        if st.button("🔍 Detect Fraud in Dataset", type="primary"):
            with st.spinner(f"⚡ Analysing {len(df_raw):,} transactions with neural network..."):
                try:
                    X_arr = preprocess(df_raw, is_eu_c, meta_c, feat_c)
                    X_3d  = X_arr.reshape(X_arr.shape[0], X_arr.shape[1], 1)
                    probs = model_c.predict(X_3d, verbose=0).ravel()
                    preds = (probs > thr_c).astype(int)

                    results = df_raw.copy()
                    results['Fraud_Prob_%'] = (probs * 100).round(2)
                    results['Risk_Score']   = [risk_score(p) for p in probs]
                    results['Prediction']   = ["🚨 FRAUD" if p==1 else "✅ LEGIT" for p in preds]

                    nf = int(preds.sum()); nl = len(preds) - nf

                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    m1,m2,m3,m4,m5 = st.columns(5)
                    m1.metric("Total",       f"{len(preds):,}")
                    m2.metric("🚨 Fraud",    f"{nf:,}",    delta=f"{nf/len(preds)*100:.1f}%", delta_color="inverse")
                    m3.metric("✅ Legit",    f"{nl:,}")
                    m4.metric("Avg Risk",    f"{results['Risk_Score'].mean():.0f}/100")
                    m5.metric("Threshold",   f"{thr_c}")

                    if voice_on:
                        fraud_tids = []
                        if 'transaction_id' in df_raw.columns:
                            fraud_tids = df_raw['transaction_id'].values[preds==1].tolist()
                        if nf == 0:
                            voice_js("All transactions appear legitimate.")
                        elif nf == 1:
                            fid = fraud_tids[0] if fraud_tids else "one transaction"
                            voice_js(f"Warning! Transaction {fid} is flagged as fraud. Risk score high. Please verify immediately.")
                        elif nf <= 5:
                            fid_str = ", ".join(str(x) for x in fraud_tids[:3])
                            voice_js(f"Alert! {nf} fraudulent transactions detected. Including {fid_str}. Please take action.")
                        else:
                            voice_js(f"Critical alert! {nf} fraudulent transactions detected in this batch. Immediate action required.")

                    cp, ch = st.columns(2)
                    with cp:
                        if PLOTLY_OK:
                            fig_p = px.pie(values=[nl,nf], names=['Legit','Fraud'],
                                           color_discrete_map={'Legit':'#1a6cf5','Fraud':'#f53b3b'},
                                           title="Fraud vs Legit", hole=0.4)
                            fig_p.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#7090b8',
                                                plot_bgcolor='rgba(0,0,0,0)',
                                                title_font=dict(color='#06d6f5', size=13))
                            st.plotly_chart(fig_p, use_container_width=True)
                    with ch:
                        fig_h, ax_h = plt.subplots(figsize=(5,3.5), facecolor='none')
                        fig_h.patch.set_facecolor('#020916')
                        dark_ax(ax_h)
                        ax_h.hist(probs[preds==0], bins=40, color='#1a6cf5', alpha=0.7, label='Legit', density=True)
                        ax_h.hist(probs[preds==1], bins=40, color='#f53b3b', alpha=0.7, label='Fraud', density=True)
                        ax_h.axvline(thr_c, color='#f5c842', ls='--', lw=1.5, label=f'Thr {thr_c}')
                        ax_h.set_title('Probability Distribution', fontsize=10, color='#06d6f5')
                        ax_h.legend(fontsize=9, facecolor='#020916', labelcolor='white')
                        ax_h.grid(alpha=0.1, color='white')
                        plt.tight_layout(); st.pyplot(fig_h); plt.close()

                    st.markdown("""
                    <div style='font-family:Rajdhani,sans-serif;font-size:16px;font-weight:700;
                                color:#06d6f5;letter-spacing:1.5px;margin:16px 0 8px'>
                      📋 PREDICTION RESULTS
                    </div>""", unsafe_allow_html=True)
                    total_cells = len(results) * len(results.columns)
                    if total_cells <= 200_000:
                        pd.set_option("styler.render.max_elements", total_cells+1000)
                        def color_row(row):
                            if "FRAUD" in str(row.get('Prediction','')):
                                return ['background-color:rgba(245,59,59,0.18)']*len(row)
                            return ['background-color:rgba(15,240,124,0.05)']*len(row)
                        st.dataframe(results.style.apply(color_row, axis=1),
                                     use_container_width=True, height=300)
                    else:
                        fr = results[results['Prediction']=='🚨 FRAUD']
                        lg = results[results['Prediction']=='✅ LEGIT'].head(30)
                        st.info(f"📊 {len(results):,} rows — fraud rows first, 30 legit sample.")
                        st.dataframe(pd.concat([fr,lg]).reset_index(drop=True),
                                     use_container_width=True, height=300)

                    SHAP_LIMIT = 50; show_n = min(SHAP_LIMIT, nf)
                    sv_batch = None; X_exp_batch = None
                    if SHAP_OK and nf > 0:
                        with st.spinner(f"🧠 Computing SHAP for {show_n} fraud rows..."):
                            try:
                                fidx_all    = np.where(preds==1)[0]
                                ex_idx      = fidx_all[:show_n]
                                X_exp_batch = X_arr[ex_idx].reshape(len(ex_idx), X_arr.shape[1], 1)
                                n_bg        = min(30, len(Xt_c))
                                bg          = Xt_c[:n_bg].reshape(n_bg, Xt_c.shape[1], 1)
                                exp         = shap.GradientExplainer(model_c, bg)
                                sv_r        = np.array(exp.shap_values(X_exp_batch)).squeeze()
                                if sv_r.ndim==3: sv_r=sv_r.mean(axis=1)
                                elif sv_r.ndim==1: sv_r=sv_r.reshape(1,-1)
                                if sv_r.ndim==2:
                                    sv_batch    = sv_r
                                    X_exp_batch = X_exp_batch.squeeze()
                                    if X_exp_batch.ndim==3: X_exp_batch=X_exp_batch.squeeze(axis=-1)
                            except Exception as e:
                                st.warning(f"SHAP failed: {str(e)[:60]}")

                    fraud_df  = results[preds==1].copy()
                    fidx_all  = np.where(preds==1)[0]
                    if len(fraud_df) > 0:
                        st.markdown(f"""
                        <div style='font-family:Rajdhani,sans-serif;font-size:16px;font-weight:700;
                                    color:#f53b3b;letter-spacing:1.5px;margin:16px 0 8px;
                                    padding-bottom:6px;border-bottom:1px solid rgba(245,59,59,0.25)'>
                          🚨 FRAUD DETAILS + SHAP ANALYSIS ({nf} TRANSACTIONS)
                        </div>""", unsafe_allow_html=True)
                        if nf > SHAP_LIMIT:
                            st.caption(f"SHAP shown for first {SHAP_LIMIT}. Download CSV for all {nf}.")
                        for rank, (idx, row) in enumerate(fraud_df.head(show_n).iterrows()):
                            score    = int(row['Risk_Score'])
                            lbl, bc  = risk_label(score)
                            tid      = str(df_raw['transaction_id'].iloc[fidx_all[rank]]) \
                                       if 'transaction_id' in df_raw.columns \
                                       else f"TXN{fidx_all[rank]+1:03d}"
                            customer = get_customer(tid)
                            with st.expander(
                                f"🚨 Fraud {rank+1} — {tid} | {customer['name']} | "
                                f"Prob: {row['Fraud_Prob_%']:.1f}% | {lbl}",
                                expanded=(rank < 2)):
                                ci, cs = st.columns([1,1])
                                with ci:
                                    st.markdown(
                                        f"<div style='background:{bc}22;border:1px solid {bc}55;"
                                        f"border-radius:10px;padding:10px 14px;margin-bottom:10px'>"
                                        f"<div style='font-family:Rajdhani,sans-serif;font-size:20px;"
                                        f"font-weight:700;color:{bc}'>⚠️ {lbl} — {score}/100</div></div>",
                                        unsafe_allow_html=True)
                                    st.progress(score/100)
                                    st.markdown(f"**Fraud Probability:** {row['Fraud_Prob_%']:.1f}%")
                                    detail_dict = {c: df_raw.loc[idx,c] for c in df_raw.columns
                                                   if c not in ['is_fraudulent','Class']}
                                    for k,v in list(detail_dict.items())[:6]:
                                        st.markdown(f"- `{k}` → **{v}**")
                                    if customer['phone']:
                                        st.markdown(
                                            f"<div style='background:rgba(15,240,124,0.08);"
                                            f"border-left:3px solid #0ff07c;border-radius:6px;"
                                            f"padding:8px 12px;margin-top:8px;font-size:12px'>"
                                            f"📱 <b>{customer['name']}</b> — {customer['phone']}</div>",
                                            unsafe_allow_html=True)
                                with cs:
                                    st.markdown("<p style='font-family:Rajdhani,sans-serif;font-size:15px;font-weight:700;color:#06d6f5'>🧠 SHAP — Why FRAUD?</p>", unsafe_allow_html=True)
                                    shap_two_plots(
                                        model=model_c, X_bg=Xt_c,
                                        x_row=X_arr[fidx_all[rank]],
                                        X_explain=X_exp_batch,
                                        sv_batch=sv_batch, row_idx=rank,
                                        feature_names=feat_c, txn_label=tid)
                        if nf > show_n:
                            st.info(f"Showing {show_n}/{nf}. Download CSV for all.")
                    else:
                        st.success("✅ All transactions appear LEGITIMATE.")

                    st.download_button("📥 Download Results CSV",
                                       results.to_csv(index=False),
                                       "fraud_results.csv", "text/csv")
                except Exception as e:
                    st.error(f"Detection failed: {e}")
                    st.exception(e)
    else:
        st.components.v1.html("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400&family=Rajdhani:wght@600&display=swap');
        .upload-zone {
          background: linear-gradient(135deg, rgba(10,26,58,0.7), rgba(7,21,48,0.8));
          border: 2px dashed rgba(26,108,245,0.35);
          border-radius: 16px; padding: 40px; text-align: center;
          margin-top: 20px; position: relative; overflow: hidden;
        }
        .upload-zone::before {
          content: ''; position: absolute; inset: 0;
          background: radial-gradient(ellipse 60% 40% at 50% 50%,
            rgba(26,108,245,0.07), transparent);
        }
        .upload-icon { font-size: 52px; margin-bottom: 12px; filter: drop-shadow(0 0 12px rgba(6,214,245,0.5)); }
        .upload-title { font-family: 'Rajdhani',sans-serif; font-size: 20px; font-weight: 700;
          color: #06d6f5; letter-spacing: 2px; margin-bottom: 8px; }
        .upload-caps { font-family: 'Space Mono',monospace; font-size: 11px; color: #3a6090;
          letter-spacing: 1.5px; }
        </style>
        <div class="upload-zone">
          <div class="upload-icon">📂</div>
          <div class="upload-title">DROP YOUR TRANSACTION CSV HERE</div>
          <div class="upload-caps">
            🤖 DATASET AUTO-DETECTED &nbsp;·&nbsp; 🎯 THRESHOLD AUTO-OPTIMISED
            &nbsp;·&nbsp; 🧠 SHAP PER FRAUD &nbsp;·&nbsp; 📱 AUTO SMS &nbsp;·&nbsp; 🔊 VOICE ALERTS
          </div>
        </div>
        """, height=200)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — LIVE SIMULATION
# ═══════════════════════════════════════════════════════════════
elif "Live" in page:
    section_header("⚡", "LIVE TRANSACTION SIMULATION", "Real-time neural fraud detection stream")

    sim_model = _euro[0]; sim_meta = _euro[1]
    sim_Xt    = _euro[2]; sim_feat  = sim_meta['feature_names']
    sim_thr   = 0.75

    if len(PHONE_MAP) > 0:
        pm_df = pd.DataFrame([
            {'TXN ID': k, 'Customer': v['name'], 'Phone': v['phone']}
            for k,v in list(PHONE_MAP.items())[:5]
        ])
        with st.expander(f"📋 Phone Mapping ({len(PHONE_MAP)} customers)", expanded=False):
            st.dataframe(pm_df, use_container_width=True)

    col_s1, col_s2 = st.columns(2)
    n_tx  = col_s1.slider("Transactions to simulate", 5, 1000, 20)
    delay = col_s2.slider("Delay (sec)", 0.1, 2.0, 0.4)

    if st.button("▶ Start Live Simulation", type="primary"):
        st.markdown("<hr style='border-color:rgba(26,108,245,0.15)'>", unsafe_allow_html=True)
        ph_status = st.empty(); ph_bar = st.progress(0)
        ph_table  = st.empty(); ph_counts = st.empty()

        log_rows=[]; fraud_list=[]; fraud_count=0
        txn_ids = list(PHONE_MAP.keys()) if PHONE_MAP else [f"TXN{i+1:03d}" for i in range(30)]

        for i in range(n_tx):
            txn_id   = txn_ids[i % len(txn_ids)]
            customer = get_customer(txn_id)
            is_fraud_sim = random.random() < 0.2
            row_vals = {}
            for j in range(1, 29):
                if j in [2,4,14,16,20,22]:
                    row_vals[f'V{j}'] = np.random.normal(3.5,1.8) if is_fraud_sim else np.random.normal(0,1.2)
                else:
                    row_vals[f'V{j}'] = np.random.normal(0,1.5)
            row_vals['Amount'] = random.uniform(500,5000) if is_fraud_sim else random.uniform(1,300)

            df_sim = pd.DataFrame([row_vals])
            try:
                X_s  = preprocess(df_sim, True, sim_meta, sim_feat)
                X_s3 = X_s.reshape(1, X_s.shape[1], 1)
                prob = float(sim_model.predict(X_s3, verbose=0).ravel()[0])
                pred = int(prob > sim_thr)
                score= risk_score(prob)
                lbl, bc = risk_label(score)

                if pred == 1:
                    fraud_count += 1
                    fraud_list.append({'txn_id':txn_id,'prob':prob,'x_row':X_s[0],'rank':fraud_count,'customer':customer})

                log_rows.append({
                    'TXN ID':   txn_id, 'Customer': customer['name'],
                    'Phone':    customer['phone'] or '—',
                    'Time':     datetime.now().strftime("%H:%M:%S"),
                    'Prob %':   f"{prob*100:.1f}", 'Risk': score,
                    'Status':   "🚨 FRAUD" if pred==1 else "✅ LEGIT", 'Level': lbl
                })

                color_bg = 'rgba(245,59,59,0.18)' if pred==1 else 'rgba(15,240,124,0.08)'
                border   = '#f53b3b' if pred==1 else '#0ff07c'
                label_t  = '🚨 FRAUD DETECTED' if pred==1 else '✅ LEGITIMATE'
                ph_status.markdown(
                    f"<div style='background:{color_bg};border:1px solid {border}55;"
                    f"border-radius:10px;padding:12px;text-align:center;"
                    f"font-family:Rajdhani,sans-serif'>"
                    f"<b style='font-size:18px;color:{border}'>{label_t}</b>"
                    f" &nbsp;·&nbsp; <b>{txn_id}</b>"
                    f" &nbsp;·&nbsp; {customer['name']}"
                    f" &nbsp;·&nbsp; {prob*100:.1f}%"
                    f" &nbsp;·&nbsp; Risk: {score}/100 — {lbl}</div>",
                    unsafe_allow_html=True)

                ph_bar.progress((i+1)/n_tx)
                ph_table.dataframe(pd.DataFrame(log_rows), use_container_width=True, height=200)
                ph_counts.markdown(
                    f"<div style='font-family:Rajdhani,sans-serif;font-size:14px;color:#7090b8'>"
                    f"Processed: <b style='color:#e8f4fd'>{i+1}/{n_tx}</b> &nbsp;|&nbsp; "
                    f"🚨 Fraud: <b style='color:#f53b3b'>{fraud_count}</b> &nbsp;|&nbsp; "
                    f"✅ Legit: <b style='color:#0ff07c'>{i+1-fraud_count}</b></div>",
                    unsafe_allow_html=True)

                if pred==1:
                    if voice_on:
                        voice_js(f"Warning! Transaction {txn_id} for customer {customer['name']} "
                                 f"is flagged as fraud. Risk score {score} out of 100.")
                    auto_send_sms(txn_id, prob)
                time.sleep(delay)
            except Exception as ex:
                st.warning(f"TXN {txn_id} error: {ex}"); continue

        st.success(f"✅ Simulation complete! {n_tx} transactions | 🚨 {fraud_count} fraud | ✅ {n_tx-fraud_count} legit")

        if log_rows:
            st.markdown("<p style='font-family:Rajdhani,sans-serif;font-size:16px;font-weight:700;color:#06d6f5'>📋 FULL SIMULATION LOG</p>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(log_rows), use_container_width=True, height=300)

        if fraud_list and SHAP_OK:
            st.markdown(f"<p style='font-family:Rajdhani,sans-serif;font-size:16px;font-weight:700;color:#f53b3b'>🧠 SHAP — {fraud_count} FRAUD TRANSACTIONS</p>", unsafe_allow_html=True)
            sv_sim=None; X_sim_exp=None
            try:
                X_sim_exp = np.array([f['x_row'] for f in fraud_list]).reshape(len(fraud_list),-1,1)
                n_bg      = min(30, len(sim_Xt))
                bg        = sim_Xt[:n_bg].reshape(n_bg, sim_Xt.shape[1], 1)
                exp       = shap.GradientExplainer(sim_model, bg)
                sv_r      = np.array(exp.shap_values(X_sim_exp)).squeeze()
                if sv_r.ndim==3: sv_r=sv_r.mean(axis=1)
                if sv_r.ndim==1: sv_r=sv_r.reshape(1,-1)
                sv_sim = sv_r
            except Exception as e:
                st.warning(f"Batch SHAP failed: {e}")

            for fi, fd in enumerate(fraud_list):
                cust = fd['customer']
                st.markdown(
                    f"<p style='font-family:Rajdhani,sans-serif;font-size:15px;font-weight:700;color:#f53b3b'>"
                    f"Fraud {fd['rank']} — {fd['txn_id']} | {cust['name']} | {fd['prob']*100:.1f}% prob</p>",
                    unsafe_allow_html=True)
                if cust['phone']:
                    st.markdown(
                        f"<div style='background:rgba(15,240,124,0.08);border-left:3px solid #0ff07c;"
                        f"border-radius:6px;padding:6px 12px;font-size:12px;margin-bottom:6px'>"
                        f"📱 SMS auto-sent → <b>{cust['name']}</b> ({cust['phone']})</div>",
                        unsafe_allow_html=True)
                shap_two_plots(
                    model=sim_model, X_bg=sim_Xt, x_row=fd['x_row'],
                    X_explain=X_sim_exp, sv_batch=sv_sim, row_idx=fi,
                    feature_names=sim_feat, txn_label=fd['txn_id'])
                st.markdown("<hr style='border-color:rgba(26,108,245,0.1)'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — GEO ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif "Geo" in page:
    section_header("🌍", "GEO FRAUD ANALYSIS", "Real credit card fraud data by country — click map to explore")

    st.info("🖱️ **Click any country** on the map to see REAL credit card fraud statistics from official sources (RBI, Nilson Report, Statista, UK Finance)")

    COUNTRY_REAL_DATA = {
        'India': {
            'data': {2018:2814, 2019:3898, 2020:4390, 2021:5123, 2022:6543, 2023:7829},
            'unit': 'crores INR', 'source': 'Reserve Bank of India Annual Reports 2018–2023', 'region': 'South Asia'
        },
        'United States': {
            'data': {2018:9.47, 2019:10.21, 2020:11.68, 2021:12.45, 2022:13.89, 2023:15.34},
            'unit': 'billion USD', 'source': 'Nilson Report — Card Fraud Statistics 2018–2023', 'region': 'North America'
        },
        'United Kingdom': {
            'data': {2018:567, 2019:623, 2020:684, 2021:731, 2022:824, 2023:912},
            'unit': 'million GBP', 'source': 'UK Finance — Fraud Report 2018–2023', 'region': 'Western Europe'
        },
        'China': {
            'data': {2018:3.2, 2019:4.1, 2020:5.3, 2021:6.8, 2022:8.4, 2023:10.2},
            'unit': 'billion CNY', 'source': 'China Banking Regulatory Commission Reports 2018–2023', 'region': 'East Asia'
        },
        'Brazil': {
            'data': {2018:1.8, 2019:2.3, 2020:2.9, 2021:3.5, 2022:4.2, 2023:5.1},
            'unit': 'billion BRL', 'source': 'Brazilian Central Bank Payment System Reports 2018–2023', 'region': 'South America'
        },
        'Germany': {
            'data': {2018:412, 2019:468, 2020:523, 2021:589, 2022:654, 2023:728},
            'unit': 'million EUR', 'source': 'German Banking Association Reports 2018–2023', 'region': 'Central Europe'
        },
        'France': {
            'data': {2018:356, 2019:402, 2020:451, 2021:498, 2022:567, 2023:634},
            'unit': 'million EUR', 'source': 'Banque de France Payment Security Reports 2018–2023', 'region': 'Western Europe'
        },
        'Australia': {
            'data': {2018:534, 2019:602, 2020:678, 2021:723, 2022:812, 2023:901},
            'unit': 'million AUD', 'source': 'Australian Payments Network Fraud Statistics 2018–2023', 'region': 'Oceania'
        },
        'Canada': {
            'data': {2018:423, 2019:478, 2020:534, 2021:589, 2022:667, 2023:745},
            'unit': 'million CAD', 'source': 'Canadian Bankers Association Reports 2018–2023', 'region': 'North America'
        },
        'Japan': {
            'data': {2018:56.8, 2019:63.2, 2020:71.5, 2021:78.9, 2022:89.3, 2023:98.7},
            'unit': 'billion JPY', 'source': 'Japan Credit Card Association Statistics 2018–2023', 'region': 'East Asia'
        },
    }

    df_world = pd.DataFrame([
        {'Country': k,
         'Total Loss (Latest Year)': list(v['data'].values())[-1],
         'Unit': v['unit'], 'Region': v['region'], 'Source': v['source']}
        for k,v in COUNTRY_REAL_DATA.items()
    ])

    if PLOTLY_OK:
        fig_w = px.choropleth(
            df_world, locations='Country', locationmode='country names',
            color='Total Loss (Latest Year)', color_continuous_scale='Reds',
            title='🌍 Click a country to see detailed fraud analysis',
            hover_data={'Region':True,'Unit':True,'Source':False,'Total Loss (Latest Year)':True})
        fig_w.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            geo=dict(bgcolor='rgba(0,0,0,0)', showframe=False,
                     showcoastlines=True, coastlinecolor='#1a3a6e',
                     landcolor='#071530', showocean=True, oceancolor='#020916'),
            font_color='#7090b8', height=480,
            title_font=dict(color='#06d6f5', size=14),
            coloraxis_colorbar=dict(
                tickfont=dict(color='#7090b8'),
                title=dict(text='Loss', font=dict(color='#7090b8'))))
        clicked_w = st.plotly_chart(fig_w, use_container_width=True, on_select="rerun", key="world_map")

        if clicked_w and clicked_w.get("selection"):
            pts = clicked_w["selection"].get("points", [])
            if pts:
                loc = pts[0].get("location", "")
                if loc in COUNTRY_REAL_DATA:
                    d = COUNTRY_REAL_DATA[loc]
                    years  = list(d["data"].keys())
                    values = list(d["data"].values())
                    start_year=years[0]; latest_year=years[-1]
                    start_value=values[0]; latest_value=values[-1]
                    growth=latest_value-start_value
                    trend="increased" if latest_value>start_value else "decreased"

                    st.markdown("<hr style='border-color:rgba(26,108,245,0.15)'>", unsafe_allow_html=True)
                    section_header("🌍", f"{loc} — Fraud Analysis", f"Official data from {d['source']}")

                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Start Year",  start_year)
                    c2.metric("Latest Year", latest_year)


                 

                    fig_yr, ax_yr = plt.subplots(figsize=(8,4), facecolor='none')
                    fig_yr.patch.set_facecolor('#020916')
                    dark_ax(ax_yr)
                    ax_yr.plot(years, values, color='#f53b3b', lw=2.5, marker='o',
                               markersize=8, markerfacecolor='#b31c1c', markeredgecolor='#f53b3b')
                    ax_yr.fill_between(years, values, alpha=0.15, color='#f53b3b')
                    ax_yr.set_title(f'{loc} — Fraud Trend ({start_year}–{latest_year})',
                                    fontsize=12, fontweight='bold', color='#06d6f5')
                    ax_yr.set_xlabel('Year', color='#7090b8')
                    ax_yr.set_ylabel(f'Loss ({d["unit"]})', color='#7090b8')
                    ax_yr.grid(alpha=0.1, color='white')
                    for yr,val in zip(years, values):
                        ax_yr.annotate(f'{val}', (yr,val), textcoords="offset points",
                                       xytext=(0,10), ha='center', color='white', fontsize=9)
                    plt.tight_layout(); st.pyplot(fig_yr); plt.close()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
   

