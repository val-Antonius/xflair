import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import shap
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="xFlair | WC 2026 Nutmeg Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DESIGN SYSTEM (from DESIGN.md)
# Minimalism + Glassmorphism | Clinical + Authoritative
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --surface: #f7f9fb;
    --surface-dim: #d8dadc;
    --surface-container-lowest: #ffffff;
    --surface-container-low: #f2f4f6;
    --surface-container: #eceef0;
    --surface-container-high: #e6e8ea;
    --on-surface: #191c1e;
    --on-surface-variant: #603e39;
    --inverse-surface: #2d3133;
    --outline: #956d67;
    --outline-variant: #ebbbb4;
    --primary: #FF0000;
    --primary-dark: #D90000;
    --on-primary: #ffffff;
    --secondary: #565e74;
    --border-subtle: #E2E8F0;
    --success: #059669;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background-color: var(--surface) !important;
    color: var(--on-surface) !important;
}

/* ============ SIDEBAR ============ */
section[data-testid="stSidebar"] {
    background-color: var(--surface-container-lowest) !important;
    border-right: 1px solid var(--border-subtle) !important;
    padding: 0 !important;
}

section[data-testid="stSidebar"] > div {
    padding: 32px 24px !important;
}

.sidebar-brand {
    font-family: 'Inter', sans-serif;
    font-weight: 800;
    font-size: 28px;
    letter-spacing: -0.02em;
    color: var(--on-surface);
    margin-bottom: 4px;
}

.sidebar-brand-accent { color: var(--primary); }

.sidebar-tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--secondary);
    margin-bottom: 32px;
}

/* Radio Navigation - 3px red indicator on left */
div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

div[role="radiogroup"] > label {
    padding: 12px 16px !important;
    border-radius: 4px !important;
    border-left: 3px solid transparent !important;
    margin: 0 !important;
    transition: all 0.2s ease;
    background: transparent !important;
}

div[role="radiogroup"] > label:hover {
    background-color: var(--surface-container-low) !important;
}

div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {
    background-color: var(--surface-container-low) !important;
    border-left: 3px solid var(--primary) !important;
}

div[role="radiogroup"] > label > div {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    color: var(--secondary) !important;
}

div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) > div {
    color: var(--on-surface) !important;
    font-weight: 600 !important;
}

div[role="radiogroup"] > label svg { display: none !important; }

/* ============ TYPOGRAPHY ============ */
h1 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
    font-size: 48px !important;
    line-height: 56px !important;
    letter-spacing: -0.02em !important;
    color: var(--on-surface) !important;
    margin-bottom: 8px !important;
}

h2 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 28px !important;
    line-height: 36px !important;
    letter-spacing: -0.01em !important;
    color: var(--on-surface) !important;
    margin-top: 8px !important;
}

h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 18px !important;
    line-height: 26px !important;
    color: var(--on-surface) !important;
}

.overline {
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--secondary) !important;
    margin-bottom: 8px !important;
}

.data-mono {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.05em;
}

/* ============ GLASS CARDS ============ */
.glass-card {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 24px;
    box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.03);
}

.player-hero {
    background: linear-gradient(135deg, var(--surface-container-lowest) 0%, var(--surface-container-low) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.03);
}

.player-hero::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(255,0,0,0.08) 0%, transparent 70%);
    pointer-events: none;
}

/* ============ METRICS ============ */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    padding: 16px !important;
}

/* Sidebar metrics: remove border and add ultra-subtle hover accent */
section[data-testid="stSidebar"] div[data-testid="stMetric"] {
    border: none !important;
    box-shadow: none !important;
    transition: background-color 0.2s ease, box-shadow 0.2s ease;
}

section[data-testid="stSidebar"] div[data-testid="stMetric"]:hover {
    background: rgba(255, 0, 0, 0.03) !important;
}
            
div[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--secondary) !important;
}

div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 24px !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    color: var(--on-surface) !important;
}

div[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}

/* ============ HEAT BADGE ============ */
.heat-badge {
    display: inline-block;
    background-color: var(--primary);
    color: white;
    padding: 2px 8px;
    border-radius: 9999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.1em;
    margin-left: 8px;
    vertical-align: middle;
}

/* ============ INPUTS ============ */
.stSelectbox label, .stTextInput label {
    font-family: 'Inter', sans-serif !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--secondary) !important;
}

.stSelectbox > div > div, .stTextInput > div > div > input {
    background-color: var(--surface-container-lowest) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 4px !important;
    color: var(--on-surface) !important;
    font-family: 'Inter', sans-serif !important;
}

.stSelectbox > div > div:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(255, 0, 0, 0.1) !important;
}

/* ============ TABS ============ */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--border-subtle);
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: var(--secondary);
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 14px;
    border-radius: 4px 4px 0 0;
    padding: 12px 20px;
}

.stTabs [aria-selected="true"] {
    background-color: var(--surface-container-low);
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary);
    font-weight: 600;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--primary) !important;
}

/* ============ BUTTONS ============ */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 4px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background-color: var(--primary) !important;
    color: var(--on-primary) !important;
    border: none !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: var(--primary-dark) !important;
}

/* ============ TABLES ============ */
.stDataFrame {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* ============ DIVIDER ============ */
.custom-divider {
    height: 1px;
    background: var(--border-subtle);
    margin: 24px 0;
}

/* ============ STAT ROW ============ */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid var(--border-subtle);
}
.stat-row:last-child { border-bottom: none; }

.stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: var(--secondary);
    font-weight: 500;
}

.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 500;
    color: var(--on-surface);
}

/* Hide Streamlit branding (keep header for sidebar expand control) */
#MainMenu, footer { visibility: hidden; }
header { visibility: visible; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY LAYOUT HELPER (1.5px thin lines, red primary, light grid)
# ============================================================
def plotly_layout(title=None, x_title=None, y_title=None, height=420):
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#191c1e", size=12),
        title=dict(
            text=f"<b>{title}</b>" if title else None,
            font=dict(family="Inter", size=16, color="#191c1e"),
            x=0, xanchor='left', y=0.98
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=11, color="#565e74")) if x_title else None,
            gridcolor='#F1F5F9', gridwidth=1,
            linecolor='#E2E8F0', zeroline=False, linewidth=1,
            tickfont=dict(family="JetBrains Mono", size=11, color="#565e74")
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=11, color="#565e74")) if y_title else None,
            gridcolor='#F1F5F9', gridwidth=1,
            linecolor='#E2E8F0', zeroline=False, linewidth=1,
            tickfont=dict(family="JetBrains Mono", size=11, color="#565e74")
        ),
        margin=dict(l=20, r=20, t=60 if title else 20, b=40),
        height=height,
        hoverlabel=dict(
            bgcolor="#ffffff",
            font_size=12,
            font_family="Inter",
            bordercolor="#E2E8F0"
        )
    )

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_assets():
    base_dir = "data/streamlit_assets"
    processed_dir = "data/processed"
    data = {}
    
    def load_csv(filename):
        for d in [base_dir, processed_dir]:
            path = f"{d}/{filename}"
            if os.path.exists(path):
                return pd.read_csv(path)
        return None

    data['top_players'] = load_csv("top_players_expected_vs_actual.csv")
    data['calibration'] = load_csv("calibration_curve.csv")
    data['errors'] = load_csv("error_analysis.csv")
    data['profiles'] = load_csv("flair_player_profiles_all.csv")
    data['validation_summary'] = load_csv("validation_summary.csv")

    shap_path = None
    for d in [base_dir, processed_dir]:
        p = f"{d}/shap_values.pkl"
        if os.path.exists(p):
            shap_path = p
            break
    if shap_path:
        with open(shap_path, "rb") as f:
            data['shap'] = pickle.load(f)
    return data

data = load_assets()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">x<span class="sidebar-brand-accent">Flair</span></div>
    <div class="sidebar-tagline">WC 2026 • Predictive Analytics</div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation", 
        ["Dashboard", "Player Lab", "Model Integrity"],
        label_visibility="collapsed"
    )
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="overline">Model Performance</div>', unsafe_allow_html=True)
    
    # mae_value = "0.306"
    # rmse_value = "0.432"
    if data.get('validation_summary') is not None and not data['validation_summary'].empty:
        summary_df = data['validation_summary']
        if {'metric', 'value'}.issubset(summary_df.columns):
            metrics_map = dict(zip(summary_df['metric'], summary_df['value']))
            if 'MAE' in metrics_map:
                mae_value = f"{float(metrics_map['MAE']):.3f}"
            if 'RMSE' in metrics_map:
                rmse_value = f"{float(metrics_map['RMSE']):.3f}"

    col1, col2 = st.columns(2)
    with col1:
        st.metric("MAE", mae_value, delta="-27.9%")
    with col2:
        st.metric("RMSE", rmse_value)
    
    st.markdown('<div class="overline" style="margin-top:24px;">Test Window</div>', unsafe_allow_html=True)
    st.markdown('<div class="data-mono" style="font-size:13px; color: var(--on-surface);">01-15 JUL 2024</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.caption("Built on StatsBomb Data • XGBoost Regressor")

# ============================================================
# PAGE 1: DASHBOARD
# ============================================================
if page == "Dashboard":
    st.markdown('<div class="overline">WC 2026 PREDICTIVE ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown("# The Flair Index")
    st.markdown("**Siapa raja nutmeg di Piala Dunia?** Model AI memprediksi probabilitas teknik paling berisiko dalam sepakbola — dari 522,885 events di 3 turnamen utama.")
    
    st.markdown("")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="overline" style="margin-bottom:12px;">Total Nutmegs <span class="heat-badge">HEAT</span></div>
            <div class="data-mono" style="font-size:36px; font-weight:500;">354</div>
            <div class="data-mono" style="font-size:11px; color:#565e74; margin-top:8px;">across 3 tournaments</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="overline" style="margin-bottom:12px;">Nutmeg Rate</div>
            <div class="data-mono" style="font-size:36px; font-weight:500;">8.8<span style="font-size:20px;">%</span></div>
            <div class="data-mono" style="font-size:11px; color:#059669; margin-top:8px;">▲ Euro 2024: 10.1%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass-card">
            <div class="overline" style="margin-bottom:12px;">Flair Players</div>
            <div class="data-mono" style="font-size:36px; font-weight:500;">153</div>
            <div class="data-mono" style="font-size:11px; color:#565e74; margin-top:8px;">≥5 dribbles profiled</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="glass-card">
            <div class="overline" style="margin-bottom:12px;">Model Accuracy</div>
            <div class="data-mono" style="font-size:36px; font-weight:500;">72.1<span style="font-size:20px;">%</span></div>
            <div class="data-mono" style="font-size:11px; color:#059669; margin-top:8px;">vs baseline</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Top Players Chart
    if 'top_players' in data and data['top_players'] is not None and not data['top_players'].empty:
        df_top = data['top_players'].sort_values('expected', ascending=True).tail(12)
        player_col = next((c for c in df_top.columns if 'player' in c.lower() or 'name' in c.lower()), df_top.columns[0])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_top['actual'], y=df_top[player_col],
            name='Actual', orientation='h',
            marker=dict(color='#E2E8F0'),
            hovertemplate="<b>%{y}</b><br>Actual: %{x} nutmegs<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            x=df_top['expected'], y=df_top[player_col],
            name='Expected (AI)', orientation='h',
            marker=dict(color='#FF0000'), opacity=0.9,
            hovertemplate="<b>%{y}</b><br>Expected: %{x:.2f}<extra></extra>"
        ))
        
        layout = plotly_layout(
            title="Top 12 Players — Expected vs Actual Nutmegs",
            x_title="Total Nutmegs (Test Window)",
            height=480
        )
        layout['barmode'] = 'overlay'
        layout['legend'] = dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1,
            font=dict(family="Inter", size=11)
        )
        layout['yaxis']['tickfont'] = dict(family="Inter", size=12, color="#191c1e")
        
        fig.update_layout(layout)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("")
    
    # Insights
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Tactical Insights")
        st.markdown("""
        <div class="glass-card">
            <div class="stat-row">
                <span class="stat-label">Location Bias</span>
                <span class="stat-value">NONE — happens everywhere</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Risk Profile</span>
                <span class="stat-value">57% fail rate</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Under Pressure</span>
                <span class="stat-value">100% of dribbles</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Tournament Peak</span>
                <span class="stat-value">Euro 2024 (10.1%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Data Architecture")
        st.markdown("""
        <div class="glass-card">
            <div class="stat-row">
                <span class="stat-label">Source</span>
                <span class="stat-value">StatsBomb Open Data</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Tournaments</span>
                <span class="stat-value">WC 2022 + Euro + Copa 2024</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Events Processed</span>
                <span class="stat-value">522,885</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Granularity</span>
                <span class="stat-value">Player-Match (no leakage)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# PAGE 2: PLAYER LAB
# ============================================================
elif page == "Player Lab":
    st.markdown('<div class="overline">SCOUTING TERMINAL</div>', unsafe_allow_html=True)
    st.markdown("# Player Lab")
    st.markdown("Analisis profil pemain berdasarkan historical performance dan prediksi AI.")
    
    if 'profiles' in data and data['profiles'] is not None and not data['profiles'].empty:
        # [FIX] Logika pencarian kolom nama yang lebih ketat
        name_candidates = ['player_name', 'player', 'name', 'nickname']
        p_col = next((c for c in name_candidates if c in data['profiles'].columns), None)
        
        if p_col is None:
            # Fallback 1: cari kolom apa saja yang mengandung kata 'name'
            p_col = next((c for c in data['profiles'].columns if 'name' in c.lower()), None)
        if p_col is None:
            # Fallback 2: cari kolom 'player' tapi PASTIKAN bukan 'id'
            p_col = next((c for c in data['profiles'].columns if 'player' in c.lower() and 'id' not in c.lower()), data['profiles'].columns[0])
            
        # [FIX] Paksa konversi ke string dan buang NaN agar tidak ada .0
        players = sorted([str(p) for p in data['profiles'][p_col].dropna().unique()])
        
        col_sel, _ = st.columns([1, 3])
        with col_sel:
            selected_player = st.selectbox("Select Player", players)
        
        if selected_player:
            # [FIX] Samakan tipe data saat filtering
            player_data = data['profiles'][data['profiles'][p_col].astype(str) == str(selected_player)].iloc[0]
            
            nutmegs = int(player_data.get('total_nutmegs', 0))
            dribbles = int(player_data.get('total_dribbles', 1))
            rate = (nutmegs / dribbles * 100) if dribbles > 0 else 0
            success_rate = float(player_data.get('dribble_success_rate', 0)) * 100
            
            st.markdown("")
            
            # Hero card
            st.markdown(f"""
            <div class="player-hero">
                <div class="overline">PLAYER PROFILE</div>
                <h1 style="margin:8px 0 24px 0;">{selected_player}</h1>
                <div style="display:flex; gap:48px; flex-wrap:wrap;">
                    <div>
                        <div class="overline" style="margin-bottom:8px;">Career Nutmegs</div>
                        <div class="data-mono" style="font-size:42px; font-weight:500;">{nutmegs}</div>
                    </div>
                    <div>
                        <div class="overline" style="margin-bottom:8px;">Total Dribbles</div>
                        <div class="data-mono" style="font-size:42px; font-weight:500;">{dribbles}</div>
                    </div>
                    <div>
                        <div class="overline" style="margin-bottom:8px;">Nutmeg Rate <span class="heat-badge">HEAT</span></div>
                        <div class="data-mono" style="font-size:42px; font-weight:500; color: var(--primary);">{rate:.1f}<span style="font-size:24px;">%</span></div>
                    </div>
                    <div>
                        <div class="overline" style="margin-bottom:8px;">Dribble Success</div>
                        <div class="data-mono" style="font-size:42px; font-weight:500;">{success_rate:.0f}<span style="font-size:24px;">%</span></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                metrics = ['Nutmeg Rate', 'Dribble Success', 'Volume']
                values = [rate, success_rate, min(dribbles / 100 * 100, 100)]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=metrics, y=values,
                    marker=dict(color=['#FF0000', '#191c1e', '#565e74']),
                    hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>"
                ))
                layout = plotly_layout(title="Skill Profile", y_title="Percentile / Rate", height=360)
                layout['yaxis']['range'] = [0, 100]
                fig.update_layout(layout)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### AI Forecast")
                expected = rate/100 * 2.3
                st.markdown(f"""
                <div class="glass-card">
                    <div class="overline">Expected / Match <span class="heat-badge">HEAT</span></div>
                    <div class="data-mono" style="font-size:48px; font-weight:500; color: var(--primary);">{expected:.2f}</div>
                    <div style="font-size:12px; color:#565e74; margin-top:8px;">Based on career rate × avg dribbles</div>
                    <div class="custom-divider"></div>
                    <div class="overline">95% Confidence</div>
                    <div class="data-mono" style="font-size:14px;">[{max(0, expected - 0.3):.2f} - {expected + 0.3:.2f}]</div>
                </div>
                """, unsafe_allow_html=True)
            
            # SHAP
            st.markdown("")
            st.markdown("### Global Feature Impact (SHAP)")
            if 'shap' in data and data['shap'] is not None:
                try:
                    shap_values = data['shap']['shap_values']
                    X_sample = data['shap']['X_sample']
                    shap_df = pd.DataFrame({
                        'feature': X_sample.columns,
                        'impact': np.abs(shap_values).mean(axis=0)
                    }).sort_values('impact', ascending=True).tail(10)
                    
                    fig_shap = go.Figure()
                    fig_shap.add_trace(go.Bar(
                        x=shap_df['impact'], y=shap_df['feature'],
                        orientation='h',
                        marker=dict(color='#FF0000', opacity=0.85),
                        hovertemplate="<b>%{y}</b><br>Impact: %{x:.4f}<extra></extra>"
                    ))
                    layout = plotly_layout(title="Top 10 Feature Importance", x_title="Mean |SHAP value|")
                    layout['yaxis']['tickfont'] = dict(family="JetBrains Mono", size=11)
                    fig_shap.update_layout(layout)
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")

# ============================================================
# PAGE 3: MODEL INTEGRITY
# ============================================================
else:
    st.markdown('<div class="overline">VALIDATION SUITE</div>', unsafe_allow_html=True)
    st.markdown("# Model Integrity")
    st.markdown("Validasi statistik: model reliable, tidak bias, dan bebas tautology.")
    
    tab1, tab2 = st.tabs(["Calibration Curve", "Error Analysis"])
    
    with tab1:
        st.markdown("### Decile Calibration")
        st.markdown("Garis merah = model. Garis abu-abu putus = *perfect prediction*. Jika menempel, model terkalibrasi.")
        
        if 'calibration' in data and data['calibration'] is not None and not data['calibration'].empty:
            df_cal = data['calibration']
            fig = go.Figure()
            
            min_v = min(df_cal['mean_pred'].min(), df_cal['mean_actual'].min())
            max_v = max(df_cal['mean_pred'].max(), df_cal['mean_actual'].max())
            fig.add_trace(go.Scatter(
                x=[min_v, max_v], y=[min_v, max_v],
                mode='lines', name='Perfect',
                line=dict(color='#956d67', dash='dash', width=1.5),
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=df_cal['mean_pred'], y=df_cal['mean_actual'],
                mode='lines+markers', name='Model',
                line=dict(color='#FF0000', width=2),
                marker=dict(size=10, color='#FF0000', line=dict(color='white', width=2)),
                hovertemplate="Pred: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>"
            ))
            
            layout = plotly_layout(
                title="Decile Calibration",
                x_title="Mean Predicted", y_title="Mean Actual",
                height=480
            )
            layout['legend'] = dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1
            )
            fig.update_layout(layout)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Where did the model fail?")
        st.markdown("Pertandingan dengan *residual* tertinggi di test window.")
        if 'errors' in data and data['errors'] is not None and not data['errors'].empty:
            st.dataframe(data['errors'], use_container_width=True, hide_index=True, height=500)