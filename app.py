import streamlit as st

# Simple password gate
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.markdown("""
    <div style="max-width:400px;margin:100px auto;text-align:center">
        <div style="font-size:3rem;margin-bottom:12px">🔍</div>
        <h2 style="color:#ffffff;margin-bottom:4px">ShipScan</h2>
        <p style="color:#64748b;margin-bottom:28px">Enter your access code to continue</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        pwd = st.text_input("Access code", type="password", label_visibility="collapsed",
                            placeholder="Enter access code...")
        if st.button("Access ShipScan", use_container_width=True, type="primary"):
            if pwd == st.secrets.get("APP_PASSWORD", "shipscan2024"):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect access code")
    return False

if not check_password():
    st.stop()

import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import time

from data_generator import generate_dataset
from utils import run_feature_pipeline, load_file
from model import run_detection

st.set_page_config(
    page_title="ShipScan — Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE CSS — fixes all reported issues
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    color: #f0f4ff;
}
.stApp { background: #060b18; }

/* ── REMOVE ALL GHOST/TRANSPARENT BOXES ── */
/* This is the core fix for the transparent box issue */
div[data-testid="stVerticalBlock"] > div:has(> div.stRadio) { background: transparent !important; }
div[class*="stRadio"] > div { background: transparent !important; border: none !important; }
[data-testid="stForm"] { background: transparent !important; border: none !important; }
div[data-baseweb="card"] { background: transparent !important; border: none !important; box-shadow: none !important; }
.element-container { background: transparent !important; }
div[data-testid="column"] > div { background: transparent !important; }
section[data-testid="stSidebar"] > div { background: transparent !important; }

/* ── MULTISELECT FIX — show full text, no cutoff ── */
[data-baseweb="tag"] {
    background: #1a3a6b !important;
    color: #60a5fa !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 6px !important;
    max-width: 120px !important;
    overflow: visible !important;
    white-space: nowrap !important;
    padding: 2px 8px !important;
}
[data-baseweb="tag"] span {
    color: #60a5fa !important;
    overflow: visible !important;
    text-overflow: unset !important;
}
[data-baseweb="select"] > div {
    background: #0d1f3c !important;
    border: 1px solid #1e3a6b !important;
    border-radius: 8px !important;
    min-height: 44px !important;
}
[data-baseweb="select"] span { color: #f0f4ff !important; }
[data-baseweb="popover"] { background: #0d1f3c !important; border: 1px solid #1e3a6b !important; border-radius: 8px !important; }
[data-baseweb="menu"] { background: #0d1f3c !important; }
[data-baseweb="menu"] li { background: #0d1f3c !important; color: #f0f4ff !important; padding: 10px 16px !important; }
[data-baseweb="menu"] li:hover { background: #1a3a6b !important; }
[role="option"] { background: #0d1f3c !important; color: #f0f4ff !important; }

/* ── INPUTS ── */
input, textarea {
    background: #0d1f3c !important;
    color: #f0f4ff !important;
    border: 1px solid #1e3a6b !important;
    border-radius: 8px !important;
}
[data-testid="stNumberInput"] > div {
    background: #0d1f3c !important;
    border: 1px solid #1e3a6b !important;
    border-radius: 8px !important;
}
[data-testid="stNumberInput"] input { background: #0d1f3c !important; color: #f0f4ff !important; }
[data-testid="stNumberInput"] button { background: #1a3a6b !important; color: #60a5fa !important; border: none !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] section {
    background: #0d1f3c !important;
    border: 2px dashed #1e3a6b !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] * { color: #94a3b8 !important; }
[data-testid="stFileUploader"] button {
    background: #1a3a6b !important;
    color: #60a5fa !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 8px !important;
}

/* ── SLIDER ── */
[data-testid="stSlider"] { padding: 12px 0 !important; }
[data-testid="stSlider"] [data-baseweb="slider"] { margin-top: 14px !important; }
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
    background: #1e3a6b !important;
    height: 6px !important;
    border-radius: 3px !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #3b82f6 !important;
    border: 3px solid #93c5fd !important;
    width: 20px !important;
    height: 20px !important;
    box-shadow: 0 0 12px rgba(59,130,246,0.6) !important;
}
[data-testid="stSlider"] p {
    color: #60a5fa !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    background: #0d1f3c !important;
    border: 1px solid #1e3a6b !important;
    border-radius: 6px !important;
    padding: 2px 10px !important;
    display: inline-block !important;
}

/* ── LABELS ── */
label, [data-testid="stWidgetLabel"] p { color: #94a3b8 !important; font-size: 0.82rem !important; letter-spacing: 0.03em !important; }
p, li { color: #e2e8f0 !important; }
h1, h2, h3, h4 { color: #ffffff !important; font-weight: 700 !important; }
small, .stCaption { color: #64748b !important; }

/* ── METRICS — uniform size, bright colors, hover ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1f3c 0%, #0f2647 100%) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    border: 1px solid #1e3a6b !important;
    min-height: 110px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    cursor: default !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(59,130,246,0.2) !important;
    border-color: #3b82f6 !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.7rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── ALERTS with hover ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="stAlert"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.4) !important;
}

/* ── CODE BLOCKS — dark ── */
[data-testid="stCode"] {
    background: #0d1f3c !important;
    border: 1px solid #1e3a6b !important;
    border-radius: 8px !important;
}
[data-testid="stCode"] pre { background: #0d1f3c !important; }
[data-testid="stCode"] code { background: transparent !important; color: #60a5fa !important; border: none !important; }
code { background: #0d1f3c !important; color: #60a5fa !important; border: 1px solid #1e3a6b !important; padding: 2px 8px !important; border-radius: 4px !important; }
pre { background: #0d1f3c !important; color: #60a5fa !important; border-radius: 8px !important; }

/* ── BUTTONS ── */
.stDownloadButton button {
    background: linear-gradient(135deg, #1a3a6b, #1e4080) !important;
    color: #60a5fa !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}
.stDownloadButton button:hover {
    background: #2563eb !important;
    color: #ffffff !important;
}
/* Fix ALL buttons — including Contact and Home */
button[kind="secondary"], button[data-testid="baseButton-secondary"] {
    background: #0d1f3c !important;
    color: #60a5fa !important;
    border: 1px solid #1e3a6b !important;
    border-radius: 8px !important;
}
button[kind="secondary"]:hover, button[data-testid="baseButton-secondary"]:hover {
    background: #1a3a6b !important;
    color: #ffffff !important;
    border-color: #3b82f6 !important;
}
button[kind="primary"], button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg,#1d4ed8,#2563eb) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
}

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    background: #0d1f3c !important;
    border: 1px solid #1e3a6b !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
}
.streamlit-expanderContent {
    background: #080f1e !important;
    border: 1px solid #1e3a6b !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { border: 1px solid #1e3a6b !important; border-radius: 10px !important; overflow: hidden !important; }

/* ── CUSTOM COMPONENTS ── */
.section-title {
    font-size: 0.72rem;
    font-weight: 600;
    color: #475569 !important;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid #0d1f3c;
    padding-bottom: 10px;
    margin: 36px 0 20px 0;
}

/* Cards — same height, hover effect */
.stat-card {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 100%);
    border: 1px solid #1e3a6b;
    border-radius: 12px;
    padding: 20px;
    min-height: 100px;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.stat-card:hover {
    border-color: #3b82f6;
    box-shadow: 0 8px 32px rgba(59,130,246,0.15);
    transform: translateY(-2px);
}
.stat-card .card-label {
    font-size: 0.72rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}
.stat-card .card-value {
    font-size: 1.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #ffffff;
    line-height: 1;
}
.stat-card .card-sub {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 6px;
}

/* Insight cards — equal height */
.insight-card {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 100%);
    border: 1px solid #1e3a6b;
    border-radius: 12px;
    padding: 18px;
    transition: all 0.2s ease;
}
.insight-card:hover {
    border-color: #3b82f6;
    box-shadow: 0 0 28px rgba(59,130,246,0.12);
}
.insight-card h4 { color: #60a5fa !important; margin: 0 0 14px 0 !important; font-size: 1rem !important; }

/* IP / user display boxes */
.item-box {
    background: #0a1628;
    border: 1px solid #1e3a6b;
    border-radius: 8px;
    padding: 9px 14px;
    margin: 5px 0;
    font-family: 'JetBrains Mono', monospace;
    color: #60a5fa;
    font-size: 0.85rem;
    cursor: default;
    transition: all 0.15s ease;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.item-box:hover {
    background: #1a3a6b;
    border-color: #3b82f6;
    color: #93c5fd;
    overflow: visible;
    white-space: normal;
    word-break: break-all;
    z-index: 10;
    position: relative;
    box-shadow: 0 4px 20px rgba(59,130,246,0.3);
}

/* Risk badges */
.badge-high   { background: linear-gradient(135deg,#dc2626,#b91c1c); color:white !important; padding:4px 14px; border-radius:20px; font-size:0.78rem; font-weight:700; display:inline-block; letter-spacing:0.05em; }
.badge-medium { background: linear-gradient(135deg,#d97706,#b45309); color:white !important; padding:4px 14px; border-radius:20px; font-size:0.78rem; font-weight:700; display:inline-block; letter-spacing:0.05em; }
.badge-low    { background: linear-gradient(135deg,#059669,#047857); color:white !important; padding:4px 14px; border-radius:20px; font-size:0.78rem; font-weight:700; display:inline-block; letter-spacing:0.05em; }

/* Mapper box */
.mapper-box { background: #0d1f3c; border: 1px solid #1e3a6b; border-radius: 12px; padding: 20px; margin: 16px 0; }

/* Reason flags */
.reason-box {
    background: rgba(220,38,38,0.08);
    border-left: 3px solid #dc2626;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    margin: 6px 0;
    color: #fca5a5 !important;
    font-size: 0.88rem;
    transition: background 0.15s;
}
.reason-box:hover { background: rgba(220,38,38,0.15); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
RISK_COLOURS = {"High":"#ef4444","Medium":"#f59e0b","Low":"#10b981"}
CHART_BG     = "rgba(0,0,0,0)"
CHART_GRID   = "#0d1f3c"
CHART_TEXT   = "#64748b"

COLUMN_ALIASES = {
    "transaction_id": ["transaction_id","txn_id","order_id","id","transaction","order_number","txn","invoice_id","bill_no"],
    "user_id":        ["user_id","customer_id","buyer_id","client_id","user","customer","buyer","member_id","account_id"],
    "amount":         ["amount","order_value","price","total","value","transaction_amount","sale_amount","order_amount","net_amount","gross_amount","payment_amount","amt"],
    "timestamp":      ["timestamp","date","order_date","transaction_date","created_at","datetime","time","purchase_date","txn_date","order_time"],
    "payment_method": ["payment_method","payment_type","payment_mode","method","mode","pay_type","payment","pay_method","payment_option"],
    "device_id":      ["device_id","device","device_name","device_type","platform","source_device"],
    "ip_address":     ["ip_address","ip","ip_addr","user_ip","client_ip","buyer_ip","source_ip"],
    "location":       ["location","city","state","address","region","area","delivery_city","shipping_city","pincode","pin_code","zip"],
}

def auto_map_columns(df_columns):
    df_cols_lower = {c.lower().strip().replace(" ","_"): c for c in df_columns}
    mapping = {}
    for req, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df_cols_lower:
                mapping[req] = df_cols_lower[alias]
                break
    return mapping

def apply_column_mapping(df, mapping):
    return df.rename(columns={v: k for k, v in mapping.items()})

def fill_optional_cols(df):
    """Fill missing optional columns with safe defaults.
    Also auto-generate transaction_id and user_id if not present —
    so datasets like creditcard.csv work without manual mapping."""
    # Auto-generate IDs if missing — no need to ask user
    if "transaction_id" not in df.columns:
        df["transaction_id"] = [f"TXN{str(i).zfill(6)}" for i in range(len(df))]
    if "user_id" not in df.columns:
        df["user_id"] = [f"USR{str(i % 500).zfill(4)}" for i in range(len(df))]
    # Fill other optional columns
    for col, default in {"payment_method":"Unknown","device_id":"Unknown",
                         "ip_address":"0.0.0.0","location":"Unknown"}.items():
        if col not in df.columns:
            df[col] = default
    return df

def show_column_mapper(df):
    st.markdown("""
    <div class="mapper-box">
        <h4 style="color:#60a5fa;margin:0 0 8px 0">Column Mapping</h4>
        <p style="color:#94a3b8;margin:0 0 6px">
        We auto-detected these matches. Only <strong style="color:#e2e8f0">amount</strong>
        and <strong style="color:#e2e8f0">timestamp</strong> are required — everything
        else will be filled automatically if skipped.</p>
    </div>
    """, unsafe_allow_html=True)
    auto_mapping = auto_map_columns(list(df.columns))
    file_columns = ["-- skip --"] + list(df.columns)
    final_mapping = {}
    col1, col2 = st.columns(2)
    # Only show mappable columns — transaction_id and user_id are auto-generated
    mappable = ["amount","timestamp","payment_method","device_id","ip_address","location"]
    # Also allow is_fraud mapping for ML training
    if any("fraud" in c.lower() or "class" in c.lower() or "label" in c.lower()
           for c in df.columns):
        mappable.append("is_fraud")
        COLUMN_ALIASES["is_fraud"] = ["is_fraud","fraud","class","label","fraudulent",
                                      "Class","is_Fraud","isFraud"]
    left_cols  = mappable[:4]
    right_cols = mappable[4:]
    with col1:
        for req in left_cols:
            auto = auto_mapping.get(req,"-- skip --")
            # For is_fraud, also check for "Class" specifically
            if req == "is_fraud" and auto == "-- skip --":
                for c in df.columns:
                    if c.lower() in ["class","is_fraud","fraud","label","isFraud".lower()]:
                        auto = c
                        break
            idx = file_columns.index(auto) if auto in file_columns else 0
            sel = st.selectbox(f"{req}", file_columns, index=idx, key=f"map_{req}")
            if sel != "-- skip --": final_mapping[req] = sel
    with col2:
        for req in right_cols:
            auto = auto_mapping.get(req,"-- skip --")
            idx = file_columns.index(auto) if auto in file_columns else 0
            sel = st.selectbox(f"{req}", file_columns, index=idx, key=f"map_{req}")
            if sel != "-- skip --": final_mapping[req] = sel

    # Only amount and timestamp are truly required now
    missing = [c for c in ["amount","timestamp"] if c not in final_mapping]
    if missing:
        st.warning(f"Please map these columns: {missing}")
        return None

    df_mapped = df.rename(columns={v: k for k, v in final_mapping.items()})
    return fill_optional_cols(df_mapped)

def style_chart(fig, top=60):
    fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=dict(color=CHART_TEXT, family="Space Grotesk"),
        margin=dict(t=top, b=30, l=10, r=10), height=340,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        title_font=dict(color="#e2e8f0", size=14, family="Space Grotesk")
    )
    fig.update_xaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID, color=CHART_TEXT, tickfont=dict(color=CHART_TEXT))
    fig.update_yaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID, color=CHART_TEXT, tickfont=dict(color=CHART_TEXT))
    return fig

def risk_badge(label):
    css = {"High":"badge-high","Medium":"badge-medium","Low":"badge-low"}.get(str(label),"badge-low")
    return f'<span class="{css}">{label}</span>'

@st.cache_data(show_spinner=False)
def get_sample_data():
    return generate_dataset(n=2000)

@st.cache_data(show_spinner=False)
def run_pipeline_cached(df_hash, df):
    features = run_feature_pipeline(df)
    scored, metrics, _ = run_detection(features)
    return scored, metrics

def df_to_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    return out.getvalue()

# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False

# ─────────────────────────────────────────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.show_dashboard:

    # Centered logo + name
    st.markdown("""
    <div style="text-align:center;padding:60px 0 40px">
        <div style="font-size:4rem;margin-bottom:12px">🔍</div>
        <h1 style="font-size:3rem;font-weight:800;margin:0;
        background:linear-gradient(135deg,#60a5fa 0%,#a78bfa 50%,#34d399 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        font-family:'Space Grotesk'">ShipScan</h1>
        <p style="color:#475569;font-size:1.1rem;margin:10px 0 0;letter-spacing:0.05em">
        AI-POWERED FRAUD DETECTION FOR ECOMMERCE SELLERS</p>
    </div>
    """, unsafe_allow_html=True)

    # Three feature pills
    st.markdown("""
    <div style="display:flex;justify-content:center;gap:16px;flex-wrap:wrap;margin-bottom:48px">
        <div style="background:#0d1f3c;border:1px solid #1e3a6b;border-radius:100px;
        padding:10px 22px;font-size:0.88rem;color:#94a3b8">
            📊 Fraud score per order
        </div>
        <div style="background:#0d1f3c;border:1px solid #1e3a6b;border-radius:100px;
        padding:10px 22px;font-size:0.88rem;color:#94a3b8">
            🔍 Plain language explanations
        </div>
        <div style="background:#0d1f3c;border:1px solid #1e3a6b;border-radius:100px;
        padding:10px 22px;font-size:0.88rem;color:#94a3b8">
            💡 Actionable recommendations
        </div>
        <div style="background:#0d1f3c;border:1px solid #1e3a6b;border-radius:100px;
        padding:10px 22px;font-size:0.88rem;color:#94a3b8">
            ⚡ Works with any CSV format
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Two options — SIDE BY SIDE
    st.markdown("""
    <p style="text-align:center;color:#475569;font-size:0.82rem;
    text-transform:uppercase;letter-spacing:0.12em;margin-bottom:16px">
    Choose how to begin</p>
    """, unsafe_allow_html=True)

    opt_a, opt_b = st.columns(2)

    with opt_a:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0d2d4a,#0a2040);
        border:1px solid #1e5a8b;border-radius:14px;padding:24px;
        text-align:center;min-height:160px;display:flex;flex-direction:column;
        justify-content:center">
            <div style="font-size:2rem;margin-bottom:10px">🧪</div>
            <div style="font-weight:700;color:#ffffff;font-size:1.05rem;margin-bottom:6px">
            Try Sample Data</div>
            <div style="color:#64748b;font-size:0.82rem;line-height:1.5">
            2,000 realistic transactions<br>with fraud patterns pre-loaded</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("▶ Launch Sample Analysis", use_container_width=True, type="primary", key="btn_sample"):
            with st.spinner("Generating sample data..."):
                raw = get_sample_data()
            st.session_state.raw_df = raw
            st.session_state.show_dashboard = True
            st.rerun()

    with opt_b:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1a1f3c,#141830);
        border:1px solid #2e3a6b;border-radius:14px;padding:24px;
        text-align:center;min-height:160px;display:flex;flex-direction:column;
        justify-content:center">
            <div style="font-size:2rem;margin-bottom:10px">📁</div>
            <div style="font-weight:700;color:#ffffff;font-size:1.05rem;margin-bottom:6px">
            Upload Your File</div>
            <div style="color:#64748b;font-size:0.82rem;line-height:1.5">
            CSV or Excel — any column names<br>we map them automatically</div>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["csv","xlsx","xls"], label_visibility="collapsed", key="file_upload")
        if uploaded is not None:
            try:
                name = uploaded.name.lower()
                raw_orig = pd.read_csv(uploaded) if name.endswith(".csv") else pd.read_excel(uploaded)
                auto_mapping = auto_map_columns(list(raw_orig.columns))
                essential = ["transaction_id","user_id","amount","timestamp"]
                if all(c in auto_mapping for c in essential):
                    raw = apply_column_mapping(raw_orig, auto_mapping)
                    raw = fill_optional_cols(raw)
                    st.success(f"✅ {len(raw):,} rows loaded")
                    st.session_state.raw_df = raw
                    if st.button("▶ Run Fraud Analysis", use_container_width=True, key="btn_upload"):
                        st.session_state.show_dashboard = True
                        st.rerun()
                else:
                    raw = show_column_mapper(raw_orig)
                    if raw is not None:
                        st.session_state.raw_df = raw
                        if st.button("▶ Run Fraud Analysis", use_container_width=True, key="btn_upload2"):
                            st.session_state.show_dashboard = True
                            st.rerun()
            except Exception as e:
                st.error(f"Error reading file: {e}")

    st.markdown("""
    <div style="text-align:center;margin-top:16px">
        <p style="color:#1e3a6b;font-size:0.75rem;margin:0">
        🔒 Your data is never stored or shared. Deleted after your session.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div style="margin:60px auto;max-width:800px">
        <h3 style="text-align:center;color:#475569;font-size:0.8rem;
        text-transform:uppercase;letter-spacing:0.15em;margin-bottom:28px">
        How It Works</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px">
            <div style="background:#0d1f3c;border:1px solid #1e3a6b;border-radius:12px;
            padding:20px;text-align:center">
                <div style="font-size:1.8rem;margin-bottom:10px">📤</div>
                <div style="font-weight:600;color:#e2e8f0;margin-bottom:6px">Upload</div>
                <div style="color:#475569;font-size:0.82rem">Export your orders as CSV from any platform</div>
            </div>
            <div style="background:#0d1f3c;border:1px solid #1e3a6b;border-radius:12px;
            padding:20px;text-align:center">
                <div style="font-size:1.8rem;margin-bottom:10px">🤖</div>
                <div style="font-weight:600;color:#e2e8f0;margin-bottom:6px">Analyse</div>
                <div style="color:#475569;font-size:0.82rem">Rules + ML scans every order in seconds</div>
            </div>
            <div style="background:#0d1f3c;border:1px solid #1e3a6b;border-radius:12px;
            padding:20px;text-align:center">
                <div style="font-size:1.8rem;margin-bottom:10px">✅</div>
                <div style="font-weight:600;color:#e2e8f0;margin-bottom:6px">Act</div>
                <div style="color:#475569;font-size:0.82rem">Clear report tells you exactly what to do</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
raw_df = st.session_state.raw_df

# Top bar with tabs
tb1, tb2, tb3 = st.columns([5, 1, 1])
with tb1:
    st.markdown("""
    <div style="padding:16px 0 8px;display:flex;align-items:center;gap:12px">
        <span style="font-size:1.5rem">🔍</span>
        <span style="font-size:1.3rem;font-weight:700;color:#ffffff">ShipScan</span>
        <span style="color:#334155;font-size:0.8rem;margin-left:4px">— Analysis Dashboard</span>
    </div>
    """, unsafe_allow_html=True)
with tb2:
    show_contact = st.button("📧 Contact", use_container_width=True)
with tb3:
    if st.button("← Home", use_container_width=True):
        st.session_state.show_dashboard = False
        st.session_state.raw_df = None
        st.rerun()

if show_contact:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1f3c,#080f1e);
    border:1px solid #1e3a6b;border-radius:14px;padding:32px;
    max-width:600px;margin:0 auto 24px">
        <h3 style="color:#60a5fa;margin:0 0 8px">Get In Touch</h3>
        <p style="color:#64748b;margin:0 0 24px;font-size:0.9rem">
        Questions, feedback, or want a custom analysis for your business?</p>

        <div style="display:flex;flex-direction:column;gap:14px">
            <div style="background:#080f1e;border:1px solid #1e3a6b;
            border-radius:10px;padding:16px;display:flex;align-items:center;gap:14px">
                <span style="font-size:1.5rem">📧</span>
                <div>
                    <div style="color:#94a3b8;font-size:0.75rem;
                    text-transform:uppercase;letter-spacing:0.08em">Email</div>
                    <div style="color:#60a5fa;font-size:1rem;font-weight:600">
                    hello@shipscan.in</div>
                </div>
            </div>
            <div style="background:#080f1e;border:1px solid #1e3a6b;
            border-radius:10px;padding:16px;display:flex;align-items:center;gap:14px">
                <span style="font-size:1.5rem">🌐</span>
                <div>
                    <div style="color:#94a3b8;font-size:0.75rem;
                    text-transform:uppercase;letter-spacing:0.08em">Website</div>
                    <div style="color:#60a5fa;font-size:1rem;font-weight:600">
                    shipscan.in</div>
                </div>
            </div>
            <div style="background:#080f1e;border:1px solid #1e3a6b;
            border-radius:10px;padding:16px">
                <div style="color:#94a3b8;font-size:0.75rem;
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">
                Free Audit Offer</div>
                <div style="color:#e2e8f0;font-size:0.88rem;line-height:1.6">
                We offer a <strong style="color:#34d399">free first analysis</strong>
                for new sellers. Upload your last month's orders and we'll send you
                a complete fraud risk report — no charge, no obligation.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

# Settings bar
sc1, sc2, sc3, sc4 = st.columns([2,2,2,1])
with sc1:
    high_risk_threshold = st.slider("Alert if high-risk exceeds (%)", 1, 30, 10)
with sc2:
    amount_threshold = st.number_input("High-amount flag (Rs.)", value=10000, step=1000)
with sc3:
    period_options = {
        "All data": None,
        "Last 1 month": 30,
        "Last 3 months": 90,
        "Last 6 months": 180,
        "Last 1 year": 365,
    }
    period_label = st.selectbox("Time period", list(period_options.keys()), index=0)
    period_days  = period_options[period_label]
with sc4:
    st.markdown(f"""
    <div style="background:#080f1e;border:1px solid #1e3a6b;border-radius:8px;
    padding:10px 14px;margin-top:28px;text-align:center">
        <div style="color:#60a5fa;font-size:1rem;font-weight:700;
        font-family:'JetBrains Mono'">{len(raw_df):,}</div>
        <div style="color:#475569;font-size:0.72rem">rows loaded</div>
    </div>
    """, unsafe_allow_html=True)

# Apply time period filter BEFORE running analysis
filtered_raw = raw_df.copy()
if period_days is not None:
    try:
        filtered_raw["timestamp"] = pd.to_datetime(filtered_raw["timestamp"], errors="coerce")
        cutoff = filtered_raw["timestamp"].max() - pd.Timedelta(days=period_days)
        filtered_raw = filtered_raw[filtered_raw["timestamp"] >= cutoff]
        st.info(f"Showing {period_label.lower()}: {len(filtered_raw):,} of {len(raw_df):,} transactions")
    except Exception:
        pass  # If timestamp parsing fails, use all data

# Run analysis on filtered data
with st.spinner("Running fraud detection — rules + ML..."):
    t0 = time.time()
    df_hash = hash(str(filtered_raw.shape) + str(filtered_raw.columns.tolist()) + period_label)
    scored, metrics = run_pipeline_cached(df_hash, filtered_raw)
    elapsed = time.time() - t0

st.caption(f"Analysis complete in {elapsed:.1f}s — {metrics.get('mode','ML')} mode")

high_risk_df   = scored[scored["risk_label"]=="High"]
medium_risk_df = scored[scored["risk_label"]=="Medium"]
low_risk_df    = scored[scored["risk_label"]=="Low"]
high_risk_pct  = len(high_risk_df)/len(scored)*100
amount_at_risk = high_risk_df["amount"].sum()

if high_risk_pct >= high_risk_threshold:
    st.error(f"🚨 HIGH ALERT — {high_risk_pct:.1f}% of transactions are HIGH RISK. {len(high_risk_df):,} orders worth Rs.{amount_at_risk:,.0f} need immediate review.")

# ── METRICS — uniform cards with bright accent borders ────────────────────────
st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)

m1,m2,m3,m4,m5 = st.columns(5)

with m1:
    st.markdown(f"""
    <div class="stat-card" style="border-left:4px solid #3b82f6">
        <div class="card-label">Total Orders</div>
        <div class="card-value">{len(scored):,}</div>
        <div class="card-sub">analysed this session</div>
    </div>""", unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="stat-card" style="border-left:4px solid #ef4444">
        <div class="card-label">🚨 High Risk</div>
        <div class="card-value" style="color:#ef4444">{len(high_risk_df):,}</div>
        <div class="card-sub">{high_risk_pct:.1f}% of total</div>
    </div>""", unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="stat-card" style="border-left:4px solid #f59e0b">
        <div class="card-label">⚠️ Medium Risk</div>
        <div class="card-value" style="color:#f59e0b">{len(medium_risk_df):,}</div>
        <div class="card-sub">{len(medium_risk_df)/len(scored)*100:.1f}% of total</div>
    </div>""", unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="stat-card" style="border-left:4px solid #10b981">
        <div class="card-label">✅ Low Risk</div>
        <div class="card-value" style="color:#10b981">{len(low_risk_df):,}</div>
        <div class="card-sub">{len(low_risk_df)/len(scored)*100:.1f}% of total</div>
    </div>""", unsafe_allow_html=True)

with m5:
    st.markdown(f"""
    <div class="stat-card" style="border-left:4px solid #a78bfa">
        <div class="card-label">💰 Amount at Risk</div>
        <div class="card-value" style="color:#a78bfa;font-size:1.3rem">Rs.{amount_at_risk:,.0f}</div>
        <div class="card-sub">from high-risk orders</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── CHARTS ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Visual Analysis</div>', unsafe_allow_html=True)

r1c1, r1c2 = st.columns(2)
with r1c1:
    risk_counts = scored["risk_label"].value_counts().reset_index()
    risk_counts.columns = ["Risk","Count"]
    fig = px.pie(risk_counts, names="Risk", values="Count", color="Risk",
                 color_discrete_map=RISK_COLOURS, hole=0.6, title="Risk Distribution")
    fig.update_traces(textfont_color="#ffffff", textfont_size=13,
                      marker=dict(line=dict(color="#060b18", width=2)))
    style_chart(fig)
    fig.update_layout(legend=dict(font=dict(color="#94a3b8",size=12)))
    st.plotly_chart(fig, use_container_width=True)

with r1c2:
    scored["date"] = scored["timestamp"].dt.date
    daily = scored.groupby(["date","risk_label"]).size().reset_index(name="count")
    fig2 = px.area(daily, x="date", y="count", color="risk_label",
                   color_discrete_map=RISK_COLOURS, title="Transaction Volume Over Time",
                   labels={"count":"Orders","date":"Date","risk_label":"Risk"})
    style_chart(fig2)
    st.plotly_chart(fig2, use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1:
    sample = scored.sample(min(600,len(scored)), random_state=42)
    fig3 = px.scatter(sample, x="amount", y="fraud_score_pct", color="risk_label",
                      color_discrete_map=RISK_COLOURS, opacity=0.75,
                      title="Amount vs Fraud Score",
                      labels={"amount":"Order Amount (Rs.)","fraud_score_pct":"Fraud Score (%)","risk_label":"Risk"},
                      hover_data=["transaction_id","user_id"])
    fig3.add_hline(y=60, line_dash="dash", line_color="#ef4444", line_width=1,
                   annotation_text="High risk", annotation_font_color="#ef4444")
    fig3.add_hline(y=30, line_dash="dash", line_color="#f59e0b", line_width=1,
                   annotation_text="Medium risk", annotation_font_color="#f59e0b")
    style_chart(fig3)
    st.plotly_chart(fig3, use_container_width=True)

with r2c2:
    user_risk = (scored.groupby("user_id")
                 .agg(avg_score=("fraud_score_pct","mean"),
                      high_risk_count=("risk_label",lambda x:(x=="High").sum()))
                 .sort_values("avg_score",ascending=False).head(10).reset_index())
    fig4 = px.bar(user_risk, x="user_id", y="avg_score", color="avg_score",
                  color_continuous_scale=["#10b981","#f59e0b","#ef4444"],
                  title="Top 10 Riskiest Users", text="high_risk_count",
                  labels={"avg_score":"Avg Fraud Score (%)","user_id":"User"})
    fig4.update_traces(
        texttemplate="%{text} high", textposition="outside",
        textfont=dict(color="#94a3b8", size=10), cliponaxis=False
    )
    style_chart(fig4, top=70)
    fig4.update_layout(
        showlegend=False, coloraxis_showscale=False,
        yaxis=dict(range=[0, user_risk["avg_score"].max()*1.3])
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── FLAGGED TRANSACTIONS ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">Flagged Transactions</div>', unsafe_allow_html=True)

cf1, cf2, cf3 = st.columns(3)
with cf1:
    filter_risk = st.multiselect(
        "Risk level",
        ["High","Medium","Low"],
        default=["High","Medium"]
    )
with cf2:
    methods = scored["payment_method"].unique().tolist() if "payment_method" in scored.columns else []
    filter_method = st.multiselect("Payment method", methods, default=[])
with cf3:
    min_score = st.slider("Min fraud score (%)", 0, 100, 30)

filtered = scored[scored["risk_label"].isin(filter_risk)]
if filter_method:
    filtered = filtered[filtered["payment_method"].isin(filter_method)]
filtered = filtered[filtered["fraud_score_pct"]>=min_score].sort_values("fraud_score_pct",ascending=False)
st.caption(f"Showing {len(filtered):,} transactions matching filters")

display_cols = ["transaction_id","user_id","amount","timestamp","payment_method","location","fraud_score_pct","risk_label"]
display_cols = [c for c in display_cols if c in filtered.columns]

def colour_risk(val):
    return {"High":"background-color:#2d0a0a;color:#fca5a5;font-weight:600",
            "Medium":"background-color:#2d1a00;color:#fcd34d;font-weight:600",
            "Low":"background-color:#002d1a;color:#6ee7b7;font-weight:600"}.get(str(val),"")

styled = (filtered[display_cols].head(200).style
          .map(colour_risk, subset=["risk_label"])
          .format({"amount":"Rs.{:,.0f}","fraud_score_pct":"{:.1f}%"}))
st.dataframe(styled, use_container_width=True, height=420)

dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        "⬇️ Download as Excel",
        data=df_to_excel(filtered[display_cols]),
        file_name="shipscan_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
with dl2:
    def generate_html_report(df, scored_df, metrics):
        """
        Generate a beautiful HTML report that can be printed as PDF
        from any browser using Ctrl+P → Save as PDF.
        """
        h = scored_df[scored_df["risk_label"]=="High"]
        m = scored_df[scored_df["risk_label"]=="Medium"]
        l = scored_df[scored_df["risk_label"]=="Low"]
        amount_risk = h["amount"].sum()

        # Build high-risk rows
        top = df[df["risk_label"]=="High"].head(20)
        rows_html = ""
        for _, row in top.iterrows():
            reasons = row.get("rule_reasons", [])
            reason_text = "<br>".join([f"• {r}" for r in reasons]) if reasons else "ML pattern detection"
            score = row.get("fraud_score_pct", 0)
            colour = "#dc2626" if score >= 60 else "#d97706"
            rows_html += f"""
            <tr>
                <td>{row.get("transaction_id","—")}</td>
                <td>{row.get("user_id","—")}</td>
                <td>Rs.{row.get("amount",0):,.0f}</td>
                <td style="color:{colour};font-weight:700">{score:.0f}%</td>
                <td style="color:{colour};font-weight:700">High</td>
                <td style="font-size:0.78rem;color:#64748b">{reason_text}</td>
            </tr>"""

        n_new = scored_df[(scored_df["is_first_txn"]==1)&(scored_df["amount"]>5000)]
        shared_ips = scored_df[scored_df["ip_user_count"]>=3]["ip_address"].unique()[:5]
        ip_list = "".join([f"<li>{ip}</li>" for ip in shared_ips])

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>ShipScan Fraud Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter',sans-serif; background:#ffffff; color:#1e293b; padding:40px; }}
  .header {{ background:linear-gradient(135deg,#0f1e35,#1a3a6b); color:white; padding:32px; border-radius:12px; margin-bottom:28px; }}
  .header h1 {{ font-size:1.8rem; font-weight:700; margin-bottom:4px; }}
  .header p {{ color:#94a3b8; font-size:0.9rem; }}
  .grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:28px; }}
  .card {{ border-radius:10px; padding:18px; border:1px solid #e2e8f0; }}
  .card .label {{ font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:#64748b; margin-bottom:6px; }}
  .card .value {{ font-size:1.6rem; font-weight:700; font-family:monospace; }}
  .red {{ border-left:4px solid #dc2626; }} .red .value {{ color:#dc2626; }}
  .amber {{ border-left:4px solid #d97706; }} .amber .value {{ color:#d97706; }}
  .green {{ border-left:4px solid #059669; }} .green .value {{ color:#059669; }}
  .blue {{ border-left:4px solid #3b82f6; }} .blue .value {{ color:#3b82f6; }}
  h2 {{ font-size:1rem; font-weight:700; color:#1e293b; margin-bottom:12px; padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.82rem; margin-bottom:28px; }}
  th {{ background:#f8fafc; padding:10px 12px; text-align:left; font-size:0.72rem; text-transform:uppercase; color:#64748b; letter-spacing:0.05em; border-bottom:1px solid #e2e8f0; }}
  td {{ padding:10px 12px; border-bottom:1px solid #f1f5f9; vertical-align:top; }}
  tr:hover {{ background:#f8fafc; }}
  .actions {{ background:#f8fafc; border-radius:10px; padding:20px; margin-bottom:28px; }}
  .action-item {{ display:flex; align-items:flex-start; gap:10px; margin-bottom:10px; }}
  .action-num {{ background:#3b82f6; color:white; border-radius:50%; width:22px; height:22px; display:flex; align-items:center; justify-content:center; font-size:0.72rem; font-weight:700; flex-shrink:0; margin-top:1px; }}
  .improvements {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
  .imp-item {{ background:white; border:1px solid #e2e8f0; border-radius:8px; padding:12px; font-size:0.82rem; }}
  .footer {{ text-align:center; color:#94a3b8; font-size:0.75rem; margin-top:28px; padding-top:16px; border-top:1px solid #e2e8f0; }}
  @media print {{ body {{ padding:20px; }} }}
</style>
</head>
<body>

<div class="header">
  <h1>🔍 ShipScan — Fraud Analysis Report</h1>
  <p>Generated by ShipScan &nbsp;|&nbsp; shipscan.in &nbsp;|&nbsp; contact@shipscan.in</p>
</div>

<div class="grid">
  <div class="card blue"><div class="label">Total Orders</div><div class="value">{len(scored_df):,}</div></div>
  <div class="card red"><div class="label">🚨 High Risk</div><div class="value">{len(h):,}</div><div style="font-size:0.75rem;color:#64748b;margin-top:4px">{len(h)/len(scored_df)*100:.1f}% of total</div></div>
  <div class="card amber"><div class="label">⚠️ Medium Risk</div><div class="value">{len(m):,}</div></div>
  <div class="card green"><div class="label">💰 Amount at Risk</div><div class="value" style="font-size:1.1rem">Rs.{amount_risk:,.0f}</div><div style="font-size:0.75rem;color:#64748b;margin-top:4px">from high-risk orders</div></div>
</div>

<h2>Top High-Risk Transactions</h2>
<table>
  <thead><tr><th>Transaction ID</th><th>User</th><th>Amount</th><th>Fraud Score</th><th>Risk</th><th>Why Flagged</th></tr></thead>
  <tbody>{rows_html}</tbody>
</table>

<h2>What Should You Do?</h2>
<div class="actions">
  <div class="action-item"><div class="action-num">1</div><div>Review all <strong>{len(h)} high-risk orders</strong> before shipping or settling payment.</div></div>
  {"<div class='action-item'><div class='action-num'>2</div><div>Verify <strong>" + str(len(n_new)) + " new users</strong> who placed high-value first orders — call before dispatching.</div></div>" if len(n_new) else ""}
  {"<div class='action-item'><div class='action-num'>3</div><div>Block or rate-limit these suspicious IPs: <ul style='margin-top:6px;padding-left:16px'>" + ip_list + "</ul></div></div>" if len(shared_ips) else ""}
</div>

<h2>Process Improvements</h2>
<div class="improvements">
  <div class="imp-item">🔐 Add OTP verification for orders above Rs.10,000</div>
  <div class="imp-item">📞 Manual review queue for new COD buyers</div>
  <div class="imp-item">📍 Block high-RTO pin codes from COD orders</div>
  <div class="imp-item">📱 Device fingerprinting for fraud ring detection</div>
  <div class="imp-item">🔄 Retrain ML model monthly with new data</div>
  <div class="imp-item">🏠 Reject landmark-only delivery addresses for COD</div>
</div>

<div class="footer">
  Generated by ShipScan &nbsp;•&nbsp; shipscan.in &nbsp;•&nbsp;
  To print as PDF: Ctrl+P → Save as PDF in your browser
</div>

</body></html>"""
        return html

    html_report = generate_html_report(filtered, scored, metrics)
    st.download_button(
        "⬇️ Download Report (PDF-ready HTML)",
        data=html_report.encode("utf-8"),
        file_name="shipscan_report.html",
        mime="text/html",
        use_container_width=True
    )
    st.caption("Open the downloaded file in Chrome and press Ctrl+P → Save as PDF")

# ── TRANSACTION EXPLAINER ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">Transaction Explainer</div>', unsafe_allow_html=True)

high_ids = high_risk_df["transaction_id"].tolist()
if high_ids:
    chosen_id = st.selectbox(
        "Select a high-risk transaction to explain",
        options=high_ids[:50],
        format_func=lambda x: f"{x} — Rs.{scored.loc[scored['transaction_id']==x,'amount'].values[0]:,.0f}"
    )
    chosen = scored[scored["transaction_id"]==chosen_id].iloc[0]
    score  = chosen["fraud_score_pct"]
    colour = "#ef4444" if score>=60 else "#f59e0b" if score>=30 else "#10b981"

    ec1, ec2 = st.columns([1,2])
    with ec1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d1f3c,#080f1e);
        border:2px solid {colour};border-radius:14px;padding:28px;text-align:center;
        box-shadow:0 0 30px {colour}22">
            <div style="font-size:3.8rem;font-weight:800;color:{colour};
            font-family:'JetBrains Mono',monospace;line-height:1">{score:.0f}%</div>
            <div style="color:#475569;font-size:0.8rem;margin:6px 0 16px;
            text-transform:uppercase;letter-spacing:0.1em">Fraud Probability</div>
            {risk_badge(str(chosen["risk_label"]))}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        for k,v in {
            "User": chosen.get("user_id","—"),
            "Amount": f"Rs.{chosen['amount']:,.0f}",
            "Method": chosen.get("payment_method","—"),
            "Location": chosen.get("location","—"),
            "Time": str(chosen.get("timestamp","—"))[:16],
            "IP": chosen.get("ip_address","—")
        }.items():
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
            padding:8px 0;border-bottom:1px solid #0d1f3c">
                <span style="color:#475569;font-size:0.82rem">{k}</span>
                <span style="color:#e2e8f0;font-size:0.82rem;font-family:'JetBrains Mono';
                max-width:60%;overflow:hidden;text-overflow:ellipsis;
                white-space:nowrap" title="{v}">{v}</span>
            </div>""", unsafe_allow_html=True)

    with ec2:
        reasons = chosen.get("rule_reasons",[])
        if reasons:
            st.markdown("**Why flagged:**")
            for r in reasons:
                st.markdown(f'<div class="reason-box">• {r}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(59,130,246,0.08);border-left:3px solid #3b82f6;
            border-radius:0 6px 6px 0;padding:12px 16px;color:#94a3b8;font-size:0.88rem">
            Flagged by ML pattern detection — statistically unusual combination of signals.
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>**Feature Snapshot**", unsafe_allow_html=True)
        features_data = {
            "Txns last 1h":     int(chosen.get("txn_count_1h",0)),
            "Txns last 24h":    int(chosen.get("txn_count_24h",0)),
            "User avg spend":   f"Rs.{chosen.get('avg_amount_user',0):,.0f}",
            "Deviation":        f"{chosen.get('amount_deviation',0):.2f}σ",
            "IP shared by":     f"{int(chosen.get('ip_user_count',1))} users",
            "Device shared by": f"{int(chosen.get('device_user_count',1))} users",
            "Location match":   "No ⚠️" if chosen.get("location_mismatch",0) else "Yes ✅",
            "Night txn":        "Yes ⚠️" if chosen.get("is_night",0) else "No ✅",
            "First ever txn":   "Yes ⚠️" if chosen.get("is_first_txn",0) else "No ✅",
        }
        for feat,val in features_data.items():
            val_str = str(val)
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
            padding:7px 0;border-bottom:1px solid #080f1e">
                <span style="color:#475569;font-size:0.82rem">{feat}</span>
                <span style="color:#e2e8f0;font-size:0.82rem;font-family:'JetBrains Mono'"
                title="{val_str}">{val_str}</span>
            </div>""", unsafe_allow_html=True)

# ── INSIGHTS ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">What Should You Do?</div>', unsafe_allow_html=True)

n_velocity = scored[scored["txn_count_1h"]>5].shape[0]
n_new_high = scored[(scored["is_first_txn"]==1)&(scored["amount"]>5000)].shape[0]
shared_ips = scored[scored["ip_user_count"]>=3]["ip_address"].unique()[:5]
top_users  = scored[scored["risk_label"]=="High"]["user_id"].value_counts().head(5).index.tolist()

ins1, ins2, ins3 = st.columns(3)

with ins1:
    st.markdown('<div class="insight-card"><h4>🚨 Immediate Actions</h4>', unsafe_allow_html=True)
    if len(high_risk_df):
        st.error(f"**{len(high_risk_df)} high-risk orders** — review before shipping")
    if n_new_high:
        st.warning(f"**{n_new_high} new users** placed high-value first orders")
    if n_velocity:
        st.warning(f"**{n_velocity} velocity patterns** — consider OTP")
    if not len(high_risk_df) and not n_new_high and not n_velocity:
        st.success("No critical immediate actions needed")
    st.markdown("</div>", unsafe_allow_html=True)

with ins2:
    st.markdown('<div class="insight-card"><h4>🔒 Block / Restrict</h4>', unsafe_allow_html=True)
    if len(shared_ips):
        st.markdown('<p style="color:#64748b;font-size:0.82rem;margin:0 0 8px">Suspicious IPs — hover to see full address:</p>', unsafe_allow_html=True)
        for ip in shared_ips:
            st.markdown(f'<div class="item-box" title="{ip}">{ip}</div>', unsafe_allow_html=True)
    if top_users:
        st.markdown('<p style="color:#64748b;font-size:0.82rem;margin:12px 0 8px">Monitor these users:</p>', unsafe_allow_html=True)
        for u in top_users:
            st.markdown(f'<div class="item-box" title="{u}">{u}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with ins3:
    st.markdown("""
    <div class="insight-card">
        <h4>📋 Process Improvements</h4>
        <div style="display:flex;flex-direction:column;gap:10px;margin-top:4px">
            <div style="background:#080f1e;border-radius:8px;padding:10px 14px;
            border-left:3px solid #3b82f6">
                <div style="color:#e2e8f0;font-size:0.85rem">Add OTP for orders above Rs.10,000</div>
            </div>
            <div style="background:#080f1e;border-radius:8px;padding:10px 14px;
            border-left:3px solid #8b5cf6">
                <div style="color:#e2e8f0;font-size:0.85rem">Manual review for new COD buyers</div>
            </div>
            <div style="background:#080f1e;border-radius:8px;padding:10px 14px;
            border-left:3px solid #10b981">
                <div style="color:#e2e8f0;font-size:0.85rem">Block high-RTO pin codes</div>
            </div>
            <div style="background:#080f1e;border-radius:8px;padding:10px 14px;
            border-left:3px solid #f59e0b">
                <div style="color:#e2e8f0;font-size:0.85rem">Device fingerprinting for fraud rings</div>
            </div>
            <div style="background:#080f1e;border-radius:8px;padding:10px 14px;
            border-left:3px solid #ef4444">
                <div style="color:#e2e8f0;font-size:0.85rem">Retrain ML model monthly</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── MODEL DETAILS ─────────────────────────────────────────────────────────────
# ── MODEL ACCURACY — always visible, not hidden in expander ──────────────────
st.markdown('<div class="section-title">Model Accuracy & Technical Details</div>', unsafe_allow_html=True)

mode = metrics.get("mode","unknown")
has_labels = "precision" in metrics

if has_labels:
    # Supervised mode — show real accuracy metrics with explanation
    precision = metrics["precision"]
    recall    = metrics["recall"]
    f1        = metrics["f1"]

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d2d1a,#0a1f12);
    border:1px solid #1a5c32;border-radius:12px;padding:20px;margin-bottom:16px">
        <div style="color:#34d399;font-size:0.78rem;text-transform:uppercase;
        letter-spacing:0.1em;margin-bottom:12px">
        Supervised ML — trained on your fraud labels (is_fraud column detected)
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:16px">
            <div style="text-align:center">
                <div style="font-size:2rem;font-weight:800;color:#34d399;
                font-family:'JetBrains Mono'">{precision:.0%}</div>
                <div style="color:#64748b;font-size:0.78rem;margin-top:4px">Precision</div>
                <div style="color:#334155;font-size:0.72rem;margin-top:2px">
                Of flagged orders, this % are genuinely risky</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:2rem;font-weight:800;color:#60a5fa;
                font-family:'JetBrains Mono'">{recall:.0%}</div>
                <div style="color:#64748b;font-size:0.78rem;margin-top:4px">Recall</div>
                <div style="color:#334155;font-size:0.72rem;margin-top:2px">
                Of all real fraud, this % was caught</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:2rem;font-weight:800;color:#a78bfa;
                font-family:'JetBrains Mono'">{f1:.0%}</div>
                <div style="color:#64748b;font-size:0.78rem;margin-top:4px">F1 Score</div>
                <div style="color:#334155;font-size:0.72rem;margin-top:2px">
                Balance of precision and recall</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:2rem;font-weight:800;color:#f59e0b;
                font-family:'JetBrains Mono'">{(precision+recall)/2:.0%}</div>
                <div style="color:#64748b;font-size:0.78rem;margin-top:4px">Overall</div>
                <div style="color:#334155;font-size:0.72rem;margin-top:2px">
                Avg accuracy on this dataset</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Plain language explanation of what these numbers mean
    if precision >= 0.8:
        prec_msg = "Excellent — very few false alarms. Sellers can trust flagged orders are genuinely risky."
    elif precision >= 0.6:
        prec_msg = "Good — some false alarms but most flags are real. Worth reviewing all flagged orders."
    else:
        prec_msg = "Moderate — retrain with more labeled data to improve. Still catching real fraud."

    if recall >= 0.7:
        rec_msg = "Strong — catching most fraud. Very few risky orders slipping through undetected."
    elif recall >= 0.4:
        rec_msg = "Moderate — catching a good portion of fraud. Some may slip through."
    else:
        rec_msg = "Low — missing some fraud. Add more training data with is_fraud labels to improve."

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">
        <div style="background:#080f1e;border:1px solid #1e3a6b;border-radius:8px;padding:14px">
            <div style="color:#94a3b8;font-size:0.78rem;margin-bottom:4px">What precision means for you:</div>
            <div style="color:#e2e8f0;font-size:0.85rem">{prec_msg}</div>
        </div>
        <div style="background:#080f1e;border:1px solid #1e3a6b;border-radius:8px;padding:14px">
            <div style="color:#94a3b8;font-size:0.78rem;margin-bottom:4px">What recall means for you:</div>
            <div style="color:#e2e8f0;font-size:0.85rem">{rec_msg}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Unsupervised — explain anomaly detection mode
    st.markdown("""
    <div style="background:#1a1a0d;border:1px solid #4a4a1a;border-radius:12px;
    padding:20px;margin-bottom:16px">
        <div style="color:#fbbf24;font-size:0.78rem;text-transform:uppercase;
        letter-spacing:0.1em;margin-bottom:8px">
        Anomaly Detection Mode — no fraud labels found in your data
        </div>
        <p style="color:#94a3b8;font-size:0.85rem;margin:0;line-height:1.7">
        The model is detecting statistically unusual transactions without knowing
        which ones are definitively fraudulent. To get precision/recall accuracy scores,
        add an <strong style="color:#fbbf24">is_fraud</strong> column to your CSV
        with 1 for known fraud and 0 for legitimate orders.
        Even without labels, the model catches genuine anomalies — velocity attacks,
        shared IPs, unusual amounts — through the rules engine.
        </p>
    </div>
    """, unsafe_allow_html=True)

with st.expander("Technical Details — How Scoring Works"):
    st.markdown(f"""
    <p style="color:#475569;font-size:0.82rem;line-height:1.8">
    <strong style="color:#94a3b8">Detection mode:</strong> {mode}<br>
    <strong style="color:#94a3b8">Formula:</strong>
    fraud_score = (0.45 × rule_score) + (0.55 × ml_probability)<br>
    <strong style="color:#94a3b8">Rules engine:</strong> velocity attacks, shared IP/device,
    high amounts, night transactions, location mismatch, new-user high-value orders<br>
    <strong style="color:#94a3b8">ML model:</strong> RandomForest when is_fraud labels exist,
    IsolationForest for anomaly detection without labels
    </p>
    """, unsafe_allow_html=True)
