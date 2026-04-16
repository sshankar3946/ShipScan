import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import time

from data_generator import generate_dataset
from utils import run_feature_pipeline, load_file
from model import run_detection

st.set_page_config(page_title="ShipScan — Fraud Detection", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: #e8eaf0; }
.stApp { background: #0a0e1a; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0f1624 !important; border-right: 1px solid #1e2d4a; }
[data-testid="stSidebar"] * { color: #c8d0e0 !important; }

/* ── Inputs ── */
input, textarea {
    background-color: #1a2540 !important;
    color: #e8eaf0 !important;
    border: 1px solid #2a3f60 !important;
    border-radius: 6px !important;
}

/* ── Number input ── */
[data-testid="stNumberInput"] > div {
    background-color: #1a2540 !important;
    border: 1px solid #2a3f60 !important;
    border-radius: 6px !important;
}
[data-testid="stNumberInput"] input {
    background-color: #1a2540 !important;
    color: #e8eaf0 !important;
}
[data-testid="stNumberInput"] button {
    background-color: #1e3a5f !important;
    color: #60a5fa !important;
    border: none !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] section {
    background-color: #111827 !important;
    border: 2px dashed #1e3a5f !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] * { color: #94a3b8 !important; }
[data-testid="stFileUploader"] button {
    background-color: #1e3a5f !important;
    color: #60a5fa !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: #111827 !important;
    border: 2px dashed #1e3a5f !important;
    border-radius: 10px !important;
}

/* ── Selectbox ── */
[data-baseweb="select"] > div {
    background-color: #1a2540 !important;
    border-color: #2a3f60 !important;
    color: #e8eaf0 !important;
}
[data-baseweb="select"] span { color: #e8eaf0 !important; }

/* ── Dropdown options ── */
[data-baseweb="popover"], [data-baseweb="menu"] {
    background-color: #1a2540 !important;
    border: 1px solid #2a3f60 !important;
}
[data-baseweb="menu"] li { background-color: #1a2540 !important; color: #e8eaf0 !important; }
[data-baseweb="menu"] li:hover { background-color: #1e3a5f !important; }
[role="option"] { background-color: #1a2540 !important; color: #e8eaf0 !important; }
[data-baseweb="tag"] { background-color: #1e3a5f !important; color: #60a5fa !important; }

/* ── SLIDER — complete fix ── */
[data-testid="stSlider"] { padding: 10px 0 !important; }
[data-testid="stSlider"] [data-baseweb="slider"] {
    margin-top: 12px !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
    background-color: #1e3a5f !important;
    height: 6px !important;
    border-radius: 3px !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[style*="background"] {
    background-color: #3b82f6 !important;
}
[data-testid="stSlider"] [role="slider"] {
    background-color: #3b82f6 !important;
    border: 3px solid #60a5fa !important;
    width: 18px !important;
    height: 18px !important;
    box-shadow: 0 0 8px rgba(59,130,246,0.5) !important;
}
[data-testid="stSlider"] p {
    color: #60a5fa !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    background: #1a2540 !important;
    border: 1px solid #2a3f60 !important;
    border-radius: 4px !important;
    padding: 2px 8px !important;
    display: inline-block !important;
}
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {
    color: #64748b !important;
    font-size: 0.72rem !important;
}

/* ── Radio ── */
[data-testid="stRadio"] label { color: #c8d0e0 !important; }

/* ── Labels ── */
label, [data-testid="stWidgetLabel"] p { color: #94a3b8 !important; }

/* ── Text ── */
p, li, .stMarkdown { color: #e8eaf0 !important; }
h1, h2, h3, h4 { color: #ffffff !important; font-weight: 700 !important; }
small, .stCaption { color: #64748b !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg,#111827,#1a2540) !important;
    border-radius: 12px !important; padding: 20px !important;
    border: 1px solid #1e3a5f !important;
    border-left: 4px solid #3b82f6 !important;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size:0.8rem !important; text-transform:uppercase; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-family:'IBM Plex Mono',monospace !important; font-size:1.8rem !important; }

/* ── CODE BLOCKS — dark background fix ── */
code {
    background: #1a2540 !important;
    color: #60a5fa !important;
    border: 1px solid #2a3f60 !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
}
[data-testid="stCode"] {
    background: #111827 !important;
    border: 1px solid #2a3f60 !important;
    border-radius: 8px !important;
}
[data-testid="stCode"] pre {
    background: #111827 !important;
    color: #60a5fa !important;
}
[data-testid="stCode"] code {
    background: transparent !important;
    color: #60a5fa !important;
    border: none !important;
}
pre {
    background: #111827 !important;
    border: 1px solid #2a3f60 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    color: #60a5fa !important;
}

/* ── Alert boxes with hover ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    cursor: default !important;
}
[data-testid="stAlert"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.4) !important;
}

/* ── Buttons ── */
.stDownloadButton button {
    background: #1e3a5f !important;
    color: #60a5fa !important;
    border: 1px solid #2563eb !important;
    border-radius: 8px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    color: #94a3b8 !important;
    background: #0f1624 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
}
.streamlit-expanderContent {
    background: #0f1624 !important;
    border: 1px solid #1e3a5f !important;
}

/* ── Custom cards ── */
.section-title {
    font-size:1rem; font-weight:600; color:#94a3b8 !important;
    text-transform:uppercase; letter-spacing:0.1em;
    border-bottom:1px solid #1e3a5f; padding-bottom:8px; margin:32px 0 20px 0;
}
.insight-card {
    background: linear-gradient(135deg,#0f1e35,#162840);
    border: 1px solid #1e3a5f; border-radius:12px; padding:20px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}
.insight-card:hover {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 24px rgba(59,130,246,0.15) !important;
}
.mapper-box {
    background: #0f1624; border: 1px solid #1e3a5f;
    border-radius:10px; padding:20px; margin:16px 0;
}

/* ── Risk badges ── */
.badge-high   { background:#dc2626; color:white !important; padding:3px 12px; border-radius:20px; font-size:0.8rem; font-weight:700; display:inline-block; }
.badge-medium { background:#d97706; color:white !important; padding:3px 12px; border-radius:20px; font-size:0.8rem; font-weight:700; display:inline-block; }
.badge-low    { background:#059669; color:white !important; padding:3px 12px; border-radius:20px; font-size:0.8rem; font-weight:700; display:inline-block; }

/* ── IP display boxes ── */
.ip-box {
    background: #111827;
    border: 1px solid #2a3f60;
    border-radius: 6px;
    padding: 8px 14px;
    margin: 6px 0;
    font-family: 'IBM Plex Mono', monospace;
    color: #60a5fa;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

RISK_COLOURS = {"High":"#dc2626","Medium":"#d97706","Low":"#059669"}
CHART_BG = "rgba(0,0,0,0)"
CHART_GRID = "#1e3a5f"
CHART_TEXT = "#94a3b8"

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

def show_column_mapper(df):
    st.markdown("""
    <div class="mapper-box">
        <h4 style="color:#60a5fa;margin:0 0 8px 0">Column Mapping Required</h4>
        <p style="color:#94a3b8">Your file uses different column names. Match them below.</p>
    </div>
    """, unsafe_allow_html=True)
    auto_mapping = auto_map_columns(list(df.columns))
    file_columns = ["-- skip --"] + list(df.columns)
    final_mapping = {}
    col1, col2 = st.columns(2)
    required_cols = list(COLUMN_ALIASES.keys())
    with col1:
        for req in required_cols[:4]:
            auto = auto_mapping.get(req,"-- skip --")
            idx = file_columns.index(auto) if auto in file_columns else 0
            sel = st.selectbox(f"{req}", file_columns, index=idx, key=f"map_{req}")
            if sel != "-- skip --": final_mapping[req] = sel
    with col2:
        for req in required_cols[4:]:
            auto = auto_mapping.get(req,"-- skip --")
            idx = file_columns.index(auto) if auto in file_columns else 0
            sel = st.selectbox(f"{req}", file_columns, index=idx, key=f"map_{req}")
            if sel != "-- skip --": final_mapping[req] = sel
    missing = [c for c in ["transaction_id","user_id","amount","timestamp"] if c not in final_mapping]
    if missing:
        st.warning(f"Please map these essential columns: {missing}")
        return None
    df_mapped = df.rename(columns={v: k for k, v in final_mapping.items()})
    for col, default in {"payment_method":"Unknown","device_id":"Unknown","ip_address":"0.0.0.0","location":"Unknown"}.items():
        if col not in df_mapped.columns: df_mapped[col] = default
    return df_mapped

def style_chart(fig, top_margin=50):
    fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=dict(color=CHART_TEXT, family="IBM Plex Sans"),
        margin=dict(t=top_margin, b=20, l=20, r=20), height=340,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=CHART_TEXT))
    )
    fig.update_xaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID, color=CHART_TEXT)
    fig.update_yaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID, color=CHART_TEXT)
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

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px">
        <div style="font-size:2.5rem">🔍</div>
        <div style="font-size:1.3rem;font-weight:700;color:#ffffff">ShipScan</div>
        <div style="font-size:0.72rem;color:#475569;letter-spacing:0.1em">FRAUD DETECTION ENGINE</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("**Data Source**")
    data_source = st.radio("input", ["Upload my file","Use sample data"], index=1, label_visibility="collapsed")
    uploaded_file = None
    if data_source == "Upload my file":
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"])
        st.caption("Any column names accepted — auto-mapped")
    st.divider()
    st.markdown("**Settings**")
    high_risk_threshold = st.slider("Alert if high-risk exceeds (%)", 1, 30, 10)
    amount_threshold = st.number_input("High-amount flag (Rs.)", value=10000, step=1000)
    st.divider()
    st.markdown('<div style="font-size:0.72rem;color:#475569;line-height:1.8">v1.0 — Python + Streamlit</div>', unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:8px 0 24px">
    <h1 style="margin:0;font-size:2rem;background:linear-gradient(90deg,#60a5fa,#a78bfa);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent">🔍 ShipScan</h1>
    <p style="color:#64748b;margin:4px 0 0;font-size:0.9rem">
    AI-powered fraud detection for eCommerce and UPI transactions</p>
</div>
""", unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
raw_df = None

if data_source == "Use sample data":
    with st.spinner("Generating sample transactions..."):
        raw_df = get_sample_data()
    st.success(f"Sample dataset loaded — {len(raw_df):,} transactions")

elif uploaded_file is not None:
    try:
        name = uploaded_file.name.lower()
        raw_df_orig = pd.read_csv(uploaded_file) if name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.success(f"File loaded — {len(raw_df_orig):,} rows, {len(raw_df_orig.columns)} columns")
        st.caption(f"Columns found: {', '.join(raw_df_orig.columns.tolist())}")
        auto_mapping = auto_map_columns(list(raw_df_orig.columns))
        essential = ["transaction_id","user_id","amount","timestamp"]
        if all(c in auto_mapping for c in essential):
            raw_df = apply_column_mapping(raw_df_orig, auto_mapping)
            for col, default in {"payment_method":"Unknown","device_id":"Unknown","ip_address":"0.0.0.0","location":"Unknown"}.items():
                if col not in raw_df.columns: raw_df[col] = default
            st.info("Columns mapped automatically")
        else:
            raw_df = show_column_mapper(raw_df_orig)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f1624,#162840);border:1px solid #1e3a5f;
    border-radius:16px;padding:40px;text-align:center;margin:20px 0">
        <div style="font-size:3rem;margin-bottom:16px">📡</div>
        <h2 style="color:#ffffff">Upload your transaction data to begin</h2>
        <p style="color:#64748b;max-width:500px;margin:0 auto">
        Upload any CSV or Excel file. We accept any column names and map them automatically.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if raw_df is None:
    st.stop()

# ── DETECTION ────────────────────────────────────────────────────────────────
with st.spinner("Running fraud detection — rules + ML analysis..."):
    t0 = time.time()
    df_hash = hash(str(raw_df.shape) + str(raw_df.columns.tolist()))
    scored, metrics = run_pipeline_cached(df_hash, raw_df)
    elapsed = time.time() - t0

st.caption(f"Analysis complete in {elapsed:.1f}s — {metrics.get('mode','ML')} mode")

high_risk_df   = scored[scored["risk_label"]=="High"]
medium_risk_df = scored[scored["risk_label"]=="Medium"]
low_risk_df    = scored[scored["risk_label"]=="Low"]
high_risk_pct  = len(high_risk_df)/len(scored)*100
amount_at_risk = high_risk_df["amount"].sum()

if high_risk_pct >= high_risk_threshold:
    st.error(f"HIGH ALERT — {high_risk_pct:.1f}% of transactions are HIGH RISK. {len(high_risk_df):,} transactions worth Rs.{amount_at_risk:,.0f} need immediate review.")

# ── METRICS ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Summary Metrics</div>', unsafe_allow_html=True)
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Total Transactions", f"{len(scored):,}")
c2.metric("High Risk", f"{len(high_risk_df):,}", f"{high_risk_pct:.1f}%", delta_color="inverse")
c3.metric("Medium Risk", f"{len(medium_risk_df):,}")
c4.metric("Low Risk", f"{len(low_risk_df):,}")
c5.metric("Amount at Risk", f"Rs.{amount_at_risk:,.0f}")

# ── CHARTS ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Visual Analysis</div>', unsafe_allow_html=True)
r1c1, r1c2 = st.columns(2)

with r1c1:
    risk_counts = scored["risk_label"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level","Count"]
    fig = px.pie(risk_counts, names="Risk Level", values="Count", color="Risk Level",
                 color_discrete_map=RISK_COLOURS, hole=0.55, title="Risk Distribution")
    fig.update_traces(textfont_color="#ffffff", textfont_size=13)
    fig.update_layout(paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                      font=dict(color=CHART_TEXT), title_font=dict(color="#ffffff",size=14),
                      height=340, margin=dict(t=40,b=20,l=20,r=20),
                      legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=CHART_TEXT)))
    st.plotly_chart(fig, use_container_width=True)

with r1c2:
    scored["date"] = scored["timestamp"].dt.date
    daily = scored.groupby(["date","risk_label"]).size().reset_index(name="count")
    fig2 = px.line(daily, x="date", y="count", color="risk_label",
                   color_discrete_map=RISK_COLOURS, markers=True, title="Transactions Over Time",
                   labels={"count":"Transactions","date":"Date","risk_label":"Risk"})
    style_chart(fig2)
    fig2.update_layout(title_font=dict(color="#ffffff",size=14))
    st.plotly_chart(fig2, use_container_width=True)

r2c1, r2c2 = st.columns(2)
with r2c1:
    sample = scored.sample(min(500,len(scored)), random_state=42)
    fig3 = px.scatter(sample, x="amount", y="fraud_score_pct", color="risk_label",
                      color_discrete_map=RISK_COLOURS, opacity=0.7, title="Amount vs Fraud Score",
                      labels={"amount":"Amount (Rs.)","fraud_score_pct":"Fraud Score (%)","risk_label":"Risk"},
                      hover_data=["transaction_id","user_id"])
    fig3.add_hline(y=60, line_dash="dash", line_color="#dc2626",
                   annotation_text="High risk", annotation_font_color="#dc2626")
    fig3.add_hline(y=30, line_dash="dash", line_color="#d97706",
                   annotation_text="Medium risk", annotation_font_color="#d97706")
    style_chart(fig3)
    fig3.update_layout(title_font=dict(color="#ffffff",size=14))
    st.plotly_chart(fig3, use_container_width=True)

with r2c2:
    user_risk = (scored.groupby("user_id")
                 .agg(avg_score=("fraud_score_pct","mean"),
                      high_risk_count=("risk_label",lambda x:(x=="High").sum()))
                 .sort_values("avg_score",ascending=False).head(10).reset_index())
    fig4 = px.bar(user_risk, x="user_id", y="avg_score", color="avg_score",
                  color_continuous_scale=["#059669","#d97706","#dc2626"],
                  title="Top 10 Riskiest Users", text="high_risk_count",
                  labels={"avg_score":"Avg Fraud Score (%)","user_id":"User"})
    # FIX: outside labels + extra top margin so text is not clipped
    fig4.update_traces(
        texttemplate="%{text} high",
        textposition="outside",
        textfont=dict(color=CHART_TEXT, size=11),
        cliponaxis=False
    )
    style_chart(fig4, top_margin=70)  # extra top space for labels
    fig4.update_layout(
        title_font=dict(color="#ffffff",size=14),
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(range=[0, user_risk["avg_score"].max() * 1.25])  # 25% extra headroom
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── FLAGGED TABLE ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Flagged Transactions</div>', unsafe_allow_html=True)
cf1, cf2, cf3 = st.columns(3)
with cf1:
    filter_risk = st.multiselect("Risk level", ["High","Medium","Low"], default=["High","Medium"])
with cf2:
    methods = scored["payment_method"].unique().tolist() if "payment_method" in scored.columns else []
    filter_method = st.multiselect("Payment method", methods, default=[])
with cf3:
    min_score = st.slider("Min fraud score (%)", 0, 100, 30)

filtered = scored[scored["risk_label"].isin(filter_risk)]
if filter_method: filtered = filtered[filtered["payment_method"].isin(filter_method)]
filtered = filtered[filtered["fraud_score_pct"]>=min_score].sort_values("fraud_score_pct",ascending=False)
st.caption(f"Showing {len(filtered):,} transactions")

display_cols = ["transaction_id","user_id","amount","timestamp","payment_method","location","fraud_score_pct","risk_label"]
display_cols = [c for c in display_cols if c in filtered.columns]

def colour_risk(val):
    return {"High":"background-color:#1a0808;color:#fca5a5;font-weight:bold",
            "Medium":"background-color:#1a1200;color:#fcd34d;font-weight:bold",
            "Low":"background-color:#001a0e;color:#6ee7b7;font-weight:bold"}.get(str(val),"")

styled = (filtered[display_cols].head(200).style
          .map(colour_risk, subset=["risk_label"])
          .format({"amount":"Rs.{:,.0f}","fraud_score_pct":"{:.1f}%"}))
st.dataframe(styled, use_container_width=True, height=400)
st.download_button("Download Results (Excel)", data=df_to_excel(filtered[display_cols]),
                   file_name="shipscan_results.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ── EXPLAINER ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Transaction Explainer</div>', unsafe_allow_html=True)
high_ids = high_risk_df["transaction_id"].tolist()
if high_ids:
    chosen_id = st.selectbox("Select high-risk transaction to explain", options=high_ids[:50],
                              format_func=lambda x: f"{x} — Rs.{scored.loc[scored['transaction_id']==x,'amount'].values[0]:,.0f}")
    chosen = scored[scored["transaction_id"]==chosen_id].iloc[0]
    score = chosen["fraud_score_pct"]
    colour = "#dc2626" if score>=60 else "#d97706" if score>=30 else "#059669"

    ec1, ec2 = st.columns([1,2])
    with ec1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#111827,#1a2540);border:1px solid {colour};
        border-radius:12px;padding:24px;text-align:center">
            <div style="font-size:3.5rem;font-weight:900;color:{colour};
            font-family:'IBM Plex Mono'">{score:.0f}%</div>
            <div style="color:#64748b;font-size:0.85rem;margin:4px 0 16px">Fraud Probability</div>
            {risk_badge(str(chosen["risk_label"]))}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        for k,v in {"User":chosen.get("user_id","—"),"Amount":f"Rs.{chosen['amount']:,.0f}",
                     "Method":chosen.get("payment_method","—"),"Location":chosen.get("location","—"),
                     "Time":str(chosen.get("timestamp","—"))[:16],"IP":chosen.get("ip_address","—")}.items():
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e3a5f"><span style="color:#64748b;font-size:0.85rem">{k}</span><span style="color:#e8eaf0;font-size:0.85rem">{v}</span></div>', unsafe_allow_html=True)

    with ec2:
        st.markdown("**Why was this flagged?**")
        reasons = chosen.get("rule_reasons",[])
        if reasons:
            for r in reasons:
                st.markdown(f'<div style="background:#1a0808;border-left:3px solid #dc2626;border-radius:4px;padding:10px 14px;margin:6px 0"><span style="color:#fca5a5;font-size:0.9rem">• {r}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#0f1624;border-left:3px solid #3b82f6;border-radius:4px;padding:10px 14px"><span style="color:#94a3b8">Flagged by ML pattern detection.</span></div>', unsafe_allow_html=True)
        st.markdown("<br>**Feature Snapshot**", unsafe_allow_html=True)
        for feat,val in {"Transactions last 1h":int(chosen.get("txn_count_1h",0)),
                          "Transactions last 24h":int(chosen.get("txn_count_24h",0)),
                          "User avg spend":f"Rs.{chosen.get('avg_amount_user',0):,.0f}",
                          "Amount deviation":f"{chosen.get('amount_deviation',0):.2f} sigma",
                          "IP shared by N users":int(chosen.get("ip_user_count",1)),
                          "Device shared by N users":int(chosen.get("device_user_count",1)),
                          "Location mismatch":"Yes" if chosen.get("location_mismatch",0) else "No",
                          "Night transaction":"Yes" if chosen.get("is_night",0) else "No",
                          "First ever transaction":"Yes" if chosen.get("is_first_txn",0) else "No"}.items():
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1e2d4a"><span style="color:#64748b;font-size:0.85rem">{feat}</span><span style="color:#e8eaf0;font-size:0.85rem">{val}</span></div>', unsafe_allow_html=True)

# ── INSIGHTS ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">What Should You Do?</div>', unsafe_allow_html=True)
n_velocity = scored[scored["txn_count_1h"]>5].shape[0]
n_new_high = scored[(scored["is_first_txn"]==1)&(scored["amount"]>5000)].shape[0]
shared_ips = scored[scored["ip_user_count"]>=3]["ip_address"].unique()[:5]
top_users  = scored[scored["risk_label"]=="High"]["user_id"].value_counts().head(5).index.tolist()

ins1,ins2,ins3 = st.columns(3)
with ins1:
    st.markdown('<div class="insight-card"><h4 style="color:#60a5fa">Immediate Actions</h4>', unsafe_allow_html=True)
    if len(high_risk_df): st.error(f"{len(high_risk_df)} high-risk transactions — review before shipping")
    if n_new_high: st.warning(f"{n_new_high} new users placed high-value first orders — verify first")
    if n_velocity: st.warning(f"{n_velocity} velocity patterns detected — consider OTP friction")
    st.markdown("</div>", unsafe_allow_html=True)

with ins2:
    st.markdown('<div class="insight-card"><h4 style="color:#60a5fa">Block / Restrict</h4>', unsafe_allow_html=True)
    if len(shared_ips):
        st.markdown("**Suspicious IPs to block:**")
        # FIX: custom dark IP boxes instead of st.code()
        for ip in shared_ips:
            st.markdown(f'<div class="ip-box">{ip}</div>', unsafe_allow_html=True)
    if top_users:
        st.markdown("<br>**Monitor these users:**", unsafe_allow_html=True)
        for u in top_users:
            st.markdown(f'<div class="ip-box">{u}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with ins3:
    st.markdown("""
    <div class="insight-card">
        <h4 style="color:#60a5fa">Process Improvements</h4>
        <p style="color:#94a3b8;font-size:0.9rem;line-height:1.9">
        • Add OTP for orders above Rs.10,000<br>
        • Manual review for new COD buyers<br>
        • Block high-RTO pin codes for COD<br>
        • Device fingerprinting for fraud rings<br>
        • Retrain ML model monthly with new data
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── MODEL DETAILS ────────────────────────────────────────────────────────────
with st.expander("Model and Technical Details"):
    st.write(f"**Detection mode:** {metrics.get('mode','unknown')}")
    if "precision" in metrics:
        m1,m2,m3 = st.columns(3)
        m1.metric("Precision", f"{metrics['precision']:.1%}")
        m2.metric("Recall", f"{metrics['recall']:.1%}")
        m3.metric("F1 Score", f"{metrics['f1']:.1%}")
    st.markdown("""
    <div style="background:#111827;border:1px solid #2a3f60;border-radius:8px;padding:14px;margin-top:12px">
        <code style="color:#60a5fa;background:transparent;border:none;font-size:0.9rem">
        fraud_score = (0.45 x rule_score) + (0.55 x ml_probability)
        </code>
    </div>
    <p style="color:#94a3b8;font-size:0.85rem;margin-top:12px">
    Rules engine: velocity attacks, shared IP/device, high amounts, night transactions, location mismatch.<br>
    ML: RandomForest when fraud labels exist, IsolationForest for anomaly detection without labels.
    </p>
    """, unsafe_allow_html=True)
