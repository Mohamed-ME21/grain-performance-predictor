import streamlit as st
import pandas as pd
import numpy as np
import io
import warnings
from pathlib import Path
from collections import OrderedDict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import smooth_and_interpolate, parse_uploaded_file
from forward_runner import predict_forward
from reverse_runner import predict_reverse, REVERSE_DIM_LABELS

warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="Grain Performance Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme and glassmorphism styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Base Reset */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    html { font-size: 15px !important; }

    .stApp {
        background: #09090b;
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f0f12 !important;
        border-right: 1px solid #27272a !important;
        min-width: 310px !important;
    }
    [data-testid="stSidebar"] * { color: #a1a1aa !important; }

    [data-testid="stSidebar"] .stRadio > label {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #f4f4f5 !important;
        margin-bottom: 0.6rem !important;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
        font-size: 0.98rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 0.8rem !important;
        background: rgba(24, 24, 27, 0.4) !important;
        border: 1px solid #27272a !important;
        border-radius: 8px !important;
        margin-bottom: 0.5rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
        border-color: #3b82f6 !important;
        background: rgba(59, 130, 246, 0.05) !important;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label > div:last-child {
        padding-left: 0 !important;
        color: #f4f4f5 !important;
    }

    .sidebar-logo-container {
        text-align: center;
        padding: 2rem 1rem 1.2rem;
    }
    .sidebar-logo-text {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 1.5rem;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: 1px;
    }
    .sidebar-brand-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f4f4f5;
        letter-spacing: -0.3px;
        margin-bottom: 2px;
    }
    .sidebar-brand-subtitle {
        font-size: 0.82rem;
        color: #71717a;
        font-weight: 500;
    }
    .sidebar-divider {
        border: none;
        border-top: 1px solid #27272a;
        margin: 1rem 0;
    }

    .step-row { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 1rem; }
    .step-num {
        width: 24px; height: 24px; min-width: 24px; border-radius: 6px;
        background: #27272a;
        color: #f4f4f5; font-size: 0.8rem; font-weight: 700;
        display: flex; align-items: center; justify-content: center;
        border: 1px solid #3f3f46;
    }
    .step-text { color: #a1a1aa; font-size: 0.88rem; font-weight: 500; line-height: 1.4; }

    /* Hero Banner */
    .hero-banner {
        position: relative;
        background: #18181b;
        border: 1px solid #27272a;
        border-top: none;
        border-radius: 0 0 14px 14px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0 !important;
        color: #f4f4f5;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #a1a1aa;
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0 0 1.2rem 0 !important;
        line-height: 1.5;
    }
    .hero-tags {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .hero-tag {
        font-size: 0.75rem;
        font-weight: 600;
        color: #71717a;
        background: #27272a;
        padding: 4px 12px;
        border-radius: 6px;
        border: 1px solid #3f3f46;
        letter-spacing: 0.2px;
    }

    /* Grain Status Strip */
    .status-strip {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.8rem;
    }
    .status-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .status-grain-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f4f4f5;
        letter-spacing: -0.2px;
    }
    .status-grain-desc {
        font-size: 0.85rem;
        color: #a1a1aa;
        line-height: 1.4;
        margin-top: 3px;
    }
    .status-right { margin-left: 1rem; }

    .badge-forward {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        background: rgba(16, 185, 129, 0.08);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .badge-reverse {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        background: rgba(139, 92, 246, 0.08);
        color: #a78bfa;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }

    /* Info Box */
    .info-box {
        background: rgba(59, 130, 246, 0.03);
        border-left: 3px solid #3b82f6;
        border-radius: 0 10px 10px 0;
        padding: 1.1rem 1.4rem;
        color: #a1a1aa;
        font-size: 0.92rem;
        margin-bottom: 1.8rem;
        line-height: 1.6;
    }
    .info-box b { color: #f4f4f5; }

    /* Input Panel */
    .input-panel {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 14px;
        padding: 2rem;
        margin-bottom: 1.8rem;
    }
    .panel-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f4f4f5;
        margin-bottom: 1.4rem;
        letter-spacing: -0.2px;
    }

    label, .stRadio label, .stCheckbox label {
        color: #a1a1aa !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
    }

    .stNumberInput input, .stTextInput input {
        background: #18181b !important;
        border: 1px solid #27272a !important;
        color: #f4f4f5 !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.95rem !important;
        padding: 0.55rem 0.85rem !important;
    }
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15) !important;
    }

    /* Summary Metric Boxes */
    .metric-box {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 10px;
        padding: 1rem 0.8rem;
        text-align: center;
        transition: border-color 0.2s ease;
    }
    .metric-box:hover { border-color: #3f3f46; }
    .metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
        font-weight: 700;
        color: #3b82f6;
    }
    .metric-lbl {
        color: #71717a;
        font-size: 0.72rem;
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 600;
    }

    /* Result Cards */
    .result-card {
        background: #18181b;
        border: 1px solid #27272a;
        border-top: 3px solid #10b981;
        border-radius: 12px;
        padding: 1.6rem 1.2rem;
        text-align: center;
        margin-top: 1rem;
        transition: border-color 0.25s ease;
    }
    .result-card-reverse {
        background: #18181b;
        border: 1px solid #27272a;
        border-top: 3px solid #8b5cf6;
        border-radius: 12px;
        padding: 1.6rem 1.2rem;
        text-align: center;
        margin-top: 1rem;
        transition: border-color 0.25s ease;
    }
    .result-card:hover, .result-card-reverse:hover {
        border-color: #3f3f46;
    }
    .result-label {
        color: #71717a;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .result-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 700;
        color: #f4f4f5;
        margin: 0.5rem 0;
    }
    .result-unit {
        color: #71717a;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    /* Buttons */
    .stButton button {
        background: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.98rem !important;
        padding: 0.7rem 2.2rem !important;
        letter-spacing: 0.2px !important;
        transition: all 0.2s ease !important;
        width: 100%;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15) !important;
    }
    .stButton button:hover {
        background: #3b82f6 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.25) !important;
    }
    .stButton button:active {
        transform: translateY(0) !important;
    }
    .stDownloadButton button {
        background: transparent !important;
        color: #3b82f6 !important;
        border: 1px solid #27272a !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 1.6rem !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    .stDownloadButton button:hover {
        background: rgba(59, 130, 246, 0.05) !important;
        border-color: #3b82f6 !important;
        color: #3b82f6 !important;
    }

    /* Selectbox / FileUploader / DataFrame */
    .stSelectbox > div > div {
        background: #18181b !important;
        border: 1px solid #27272a !important;
        color: #f4f4f5 !important;
        border-radius: 8px !important;
    }
    .stFileUploader {
        background: rgba(24, 24, 27, 0.4) !important;
        border: 2px dashed #27272a !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
    }
    .stDataFrame {
        border: 1px solid #27272a !important;
        border-radius: 12px !important;
    }

    /* Dimensions Table */
    .dim-table {
        width: 100%; border-collapse: collapse; margin-top: 1rem;
        border-radius: 10px; overflow: hidden;
        border: 1px solid #27272a;
    }
    .dim-table th {
        text-align: left; padding: 1rem 1.5rem; color: #a1a1aa;
        background: #18181b;
        border-bottom: 1px solid #27272a; font-size: 0.78rem;
        text-transform: uppercase; letter-spacing: 0.8px; font-weight: 700;
    }
    .dim-table td {
        padding: 1rem 1.5rem; color: #f4f4f5;
        border-bottom: 1px solid #27272a;
        font-family: 'JetBrains Mono', monospace; font-size: 0.95rem;
    }
    .dim-table tr:nth-child(even) td {
        background: rgba(24, 24, 27, 0.4);
    }
    .dim-table tr:hover td {
        background: rgba(59, 130, 246, 0.04);
    }

    /* Footer */
    .enhanced-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.8rem 0;
        color: #71717a;
        font-size: 0.82rem;
        border-top: 1px solid #27272a;
        margin-top: 4rem;
    }
    .footer-highlight { color: #3b82f6; font-weight: 600; }
    @media (max-width: 768px) {
        .enhanced-footer { flex-direction: column; gap: 8px; text-align: center; }
    }
</style>
""", unsafe_allow_html=True)

# Model Registry
GRAIN_TYPES = ["Bates", "C", "Conical", "D", "Finocyl", "Moon", "Road and Tube", "Star", "X"]

GRAIN_DESCRIPTIONS = {
    "Bates":         "Cylindrical grain with a central circular bore - the simplest and most widely used configuration.",
    "C":             "C-shaped slot grain offering tailored thrust profiles through slot width and offset control.",
    "Conical":       "Tapered core grain with forward and aft diameters, producing a progressive thrust profile.",
    "D":             "D-shaped slot grain with offset geometry for customized burning characteristics.",
    "Finocyl":       "Fin-augmented cylindrical grain with radial fins for enhanced surface area.",
    "Moon":          "Offset circular bore (Moon-burner) grain for regressive/neutral thrust profiles.",
    "Road and Tube": "Rod-and-tube configuration with central rod support structure.",
    "Star":          "Star-shaped internal bore offering high initial burning surface area.",
    "X":             "X-shaped cross-slot grain with intersecting slot geometry.",
}

FORWARD_INPUTS = {
    "Bates": [
        {"key": "length",          "label": "Length",          "default": 40.0,  "step": 1.0},
        {"key": "diameter",        "label": "Diameter",        "default": 10.0,  "step": 0.5},
        {"key": "core_diameter",   "label": "Core Diameter",   "default": 3.0,   "step": 0.5},
        {"key": "throat_diameter", "label": "Throat Diameter",  "default": 1.5,   "step": 0.1},
        {"key": "exit_diameter",   "label": "Exit Diameter",    "default": 2.5,   "step": 0.1},
    ],
    "C": [
        {"key": "length",          "label": "Length",           "default": 40.0,  "step": 1.0},
        {"key": "diameter",        "label": "Diameter",         "default": 10.0,  "step": 0.5},
        {"key": "slot_width",      "label": "Slot Width",       "default": 1.5,   "step": 0.1},
        {"key": "slot_offset",     "label": "Slot Offset",      "default": 3.0,   "step": 0.5},
        {"key": "throat_diameter", "label": "Throat Diameter",   "default": 1.2,   "step": 0.1},
        {"key": "exit_diameter",   "label": "Exit Diameter",     "default": 2.0,   "step": 0.1},
    ],
    "Conical": [
        {"key": "length",            "label": "Length",                "default": 40.0,  "step": 1.0},
        {"key": "diameter",          "label": "Diameter",              "default": 10.0,  "step": 0.5},
        {"key": "fwd_core_diameter", "label": "Forward Core Diameter", "default": 3.0,   "step": 0.5},
        {"key": "aft_core_diameter", "label": "Aft Core Diameter",     "default": 5.0,   "step": 0.5},
        {"key": "throat_diameter",   "label": "Throat Diameter",        "default": 1.5,   "step": 0.1},
        {"key": "exit_diameter",     "label": "Exit Diameter",          "default": 2.5,   "step": 0.1},
    ],
    "D": [
        {"key": "length",          "label": "Length",           "default": 40.0,  "step": 1.0},
        {"key": "diameter",        "label": "Diameter",         "default": 10.0,  "step": 0.5},
        {"key": "slot_offset",     "label": "Slot Offset",      "default": 2.0,   "step": 0.5},
        {"key": "throat_diameter", "label": "Throat Diameter",   "default": 1.5,   "step": 0.1},
        {"key": "exit_diameter",   "label": "Exit Diameter",     "default": 2.5,   "step": 0.1},
    ],
    "Finocyl": [
        {"key": "diameter",        "label": "Diameter",         "default": 16.0,  "step": 1.0},
        {"key": "length",          "label": "Length",           "default": 40.0,  "step": 1.0},
        {"key": "core_diameter",   "label": "Core Diameter",    "default": 3.0,   "step": 0.5},
        {"key": "num_fins",        "label": "Number of Fins",   "default": 6.0,   "step": 1.0},
        {"key": "fin_length",      "label": "Fin Length",        "default": 2.0,   "step": 0.5},
        {"key": "fin_width",       "label": "Fin Width",         "default": 0.6,   "step": 0.1},
        {"key": "throat_diameter", "label": "Throat Diameter",   "default": 1.5,   "step": 0.1},
        {"key": "exit_diameter",   "label": "Exit Diameter",     "default": 2.25,  "step": 0.1},
    ],
    "Moon": [
        {"key": "length",          "label": "Length",            "default": 40.0,  "step": 1.0},
        {"key": "outer_diameter",  "label": "Outer Diameter",    "default": 10.0,  "step": 0.5},
        {"key": "inner_diameter",  "label": "Inner Diameter",    "default": 3.0,   "step": 0.5},
        {"key": "core_offset",     "label": "Core Offset",       "default": 2.0,   "step": 0.5},
        {"key": "throat_diameter", "label": "Throat Diameter",    "default": 1.2,   "step": 0.1},
        {"key": "exit_diameter",   "label": "Exit Diameter",      "default": 2.0,   "step": 0.1},
    ],
    "Road and Tube": [
        {"key": "length",            "label": "Length",            "default": 60.0,  "step": 1.0},
        {"key": "diameter",          "label": "Diameter",          "default": 14.0,  "step": 0.5},
        {"key": "core_diameter",     "label": "Core Diameter",     "default": 4.0,   "step": 0.5},
        {"key": "rod_diameter",      "label": "Rod Diameter",      "default": 1.0,   "step": 0.1},
        {"key": "support_diameter",  "label": "Support Diameter",  "default": 0.0,   "step": 0.1},
        {"key": "throat_diameter",   "label": "Throat Diameter",   "default": 2.0,   "step": 0.1},
        {"key": "exit_diameter",     "label": "Exit Diameter",     "default": 3.0,   "step": 0.1},
    ],
    "Star": [
        {"key": "length",           "label": "Length",              "default": 40.0,  "step": 1.0},
        {"key": "diameter",         "label": "Outer Diameter",      "default": 10.0,  "step": 0.5},
        {"key": "num_points",       "label": "Number of Points",    "default": 5.0,   "step": 2.0},
        {"key": "point_length",     "label": "Point Length",         "default": 2.0,   "step": 0.5},
        {"key": "point_base_width", "label": "Point Base Width",     "default": 1.0,   "step": 0.1},
        {"key": "throat_diameter",  "label": "Throat Diameter",      "default": 1.5,   "step": 0.1},
        {"key": "exit_diameter",    "label": "Exit Diameter",        "default": 2.5,   "step": 0.1},
    ],
    "X": [
        {"key": "length",          "label": "Length",           "default": 40.0,  "step": 1.0},
        {"key": "diameter",        "label": "Diameter",         "default": 10.0,  "step": 0.5},
        {"key": "slot_length",     "label": "Slot Length",       "default": 3.0,   "step": 0.5},
        {"key": "slot_width",      "label": "Slot Width",        "default": 1.0,   "step": 0.1},
        {"key": "throat_diameter", "label": "Throat Diameter",   "default": 1.5,   "step": 0.1},
        {"key": "exit_diameter",   "label": "Exit Diameter",     "default": 2.5,   "step": 0.1},
    ],
}

def create_performance_plotly(time_steps, thrust, pressure, title_prefix="Predicted"):
    """Create a styled dual-panel Plotly chart matching the Slate/Zinc dark UI theme."""
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=(f"{title_prefix} Thrust Curve", f"{title_prefix} Chamber Pressure"),
        horizontal_spacing=0.12
    )
    
    # Thrust curve
    fig.add_trace(
        go.Scatter(
            x=time_steps, y=thrust,
            mode='lines',
            name='Thrust (N)',
            line=dict(color='#ef4444', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.06)'
        ),
        row=1, col=1
    )
    
    # Pressure curve
    fig.add_trace(
        go.Scatter(
            x=time_steps, y=pressure,
            mode='lines',
            name='Pressure (MPa)',
            line=dict(color='#3b82f6', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.06)'
        ),
        row=1, col=2
    )
    
    # Update layout to match dark theme
    fig.update_layout(
        paper_bgcolor='#09090b',
        plot_bgcolor='#18181b',
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(color='#a1a1aa', family='Plus Jakarta Sans, sans-serif')
    )
    
    # Axis styling
    fig.update_xaxes(
        title_text="Time (s)", 
        gridcolor='#27272a', 
        linecolor='#27272a',
        zerolinecolor='#27272a',
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Thrust (N)", 
        gridcolor='#27272a', 
        linecolor='#27272a',
        zerolinecolor='#27272a',
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text="Time (s)", 
        gridcolor='#27272a', 
        linecolor='#27272a',
        zerolinecolor='#27272a',
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="Pressure (MPa)", 
        gridcolor='#27272a', 
        linecolor='#27272a',
        zerolinecolor='#27272a',
        row=1, col=2
    )
    
    # Subtitle styling
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=13, color='#f4f4f5', weight='bold')
        
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo-container">
        <div class="sidebar-logo-text">GP</div>
        <div class="sidebar-brand-name">Grain Predictor</div>
        <div class="sidebar-brand-subtitle">Solid Propellant Analytics</div>
    </div>
    <hr style='border-color:#27272a; margin: 0 0 1rem;'>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Prediction Mode",
        ["Forward (Dims ➔ Performance)", "Reverse (Performance ➔ Dims)"],
        key="prediction_mode",
    )
    is_forward = "Forward" in mode

    st.markdown("<hr style='border-color:#27272a; margin:1rem 0;'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='font-size:0.82rem; color:#71717a; text-transform:uppercase;
                letter-spacing:1px; margin-bottom:0.6rem; font-weight:700;'>
        Select {"Forward" if is_forward else "Reverse"} Configuration
    </div>
    """, unsafe_allow_html=True)

    selected_grain = st.selectbox(
        "Select Grain Type",
        GRAIN_TYPES,
        format_func=lambda x: f"{x} Grain",
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#27272a; margin:1rem 0;'>", unsafe_allow_html=True)

    # How-to steps
    if is_forward:
        st.markdown("""
        <div style='font-size:0.95rem; color:#a1a1aa;'>
        <b style='color:#f4f4f5; font-size:1.05rem;'>How to use</b><br><br>
        <div class='step-row'><div class='step-num'>1</div>
        <div class='step-text'>Choose a grain model above</div></div>
        <div class='step-row'><div class='step-num'>2</div>
        <div class='step-text'>Enter dimensions in <b>cm</b></div></div>
        <div class='step-row'><div class='step-num'>3</div>
        <div class='step-text'>Click <b>Run Prediction</b></div></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='font-size:0.95rem; color:#a1a1aa;'>
        <b style='color:#f4f4f5; font-size:1.05rem;'>How to use</b><br><br>
        <div class='step-row'><div class='step-num'>1</div>
        <div class='step-text'>Choose a grain model above</div></div>
        <div class='step-row'><div class='step-num'>2</div>
        <div class='step-text'>Download the template file</div></div>
        <div class='step-row'><div class='step-num'>3</div>
        <div class='step-text'>Upload your Excel / CSV data</div></div>
        <div class='step-row'><div class='step-num'>4</div>
        <div class='step-text'>Click <b>Run Prediction</b></div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#27272a; margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.85rem; color:#71717a; text-align:center; font-weight:500;'>
        9 forward &amp; 9 reverse models<br>
        Dimensions in <b style='color:#3b82f6;'>cm</b>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN CONTENT ---
st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">Rocket Motor Grain Prediction System</h1>
    <p class="hero-subtitle">AI-powered performance prediction for solid rocket motor grain geometries</p>
    <div class="hero-tags">
        <span class="hero-tag">Forward Models</span>
        <span class="hero-tag">Reverse Models</span>
        <span class="hero-tag">Unit: cm</span>
        <span class="hero-tag">Engine: Keras / TF</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Status Strip
badge_cls  = "badge-forward" if is_forward else "badge-reverse"
badge_text = "Forward Mode: Dims ➔ Performance" if is_forward else "Reverse Mode: Performance ➔ Dims"
description = GRAIN_DESCRIPTIONS.get(selected_grain, "")

st.markdown(f"""
<div class="status-strip">
    <div class="status-left">
        <div class="status-grain-info">
            <div class="status-grain-name">{selected_grain} Grain Configuration</div>
            <div class="status-grain-desc">{description}</div>
        </div>
    </div>
    <div class="status-right">
        <span class="{badge_cls}">{badge_text}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- FORWARD MODE UI ---
if is_forward:
    st.markdown("""
    <div class='info-box'>
        Enter the grain dimensions below. All geometric dimensions are in <b>cm</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='input-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Input Parameters</div>", unsafe_allow_html=True)

    inputs_cfg = FORWARD_INPUTS[selected_grain]
    n_cols = 2 if len(inputs_cfg) <= 6 else 3
    cols = st.columns(n_cols)
    input_values = OrderedDict()

    for idx, inp in enumerate(inputs_cfg):
        with cols[idx % n_cols]:
            val = st.number_input(
                f"{inp['label']} (cm)",
                value=float(inp["default"]),
                step=float(inp["step"]),
                format="%.2f",
                key=f"fwd_{selected_grain}_{inp['key']}",
            )
            input_values[inp["key"]] = val

    st.markdown("</div>", unsafe_allow_html=True)

    # Summary strip
    st.markdown("<div style='margin-bottom:1rem;'>", unsafe_allow_html=True)
    summary_cols = st.columns(len(inputs_cfg))
    for i, inp in enumerate(inputs_cfg):
        with summary_cols[i]:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-val'>{input_values[inp['key']]:.1f}</div>
                <div class='metric-lbl'>{inp['label']}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Run button
    run_col, _ = st.columns([1, 2])
    with run_col:
        run_fwd = st.button("Run Prediction", key="run_forward")

    if run_fwd:
        with st.spinner("Loading model and running prediction..."):
            try:
                result = predict_forward(selected_grain, input_values)

                st.markdown("<hr style='border-color:#27272a; margin: 2rem 0;'>", unsafe_allow_html=True)

                # Scalar metrics
                m_cols = st.columns(4)
                metrics = [
                    ("ISP", f"{result['isp']:.2f}", "s"),
                    ("Total Impulse", f"{result['total_impulse']:.2f}", "N·s"),
                    ("Burn Time", f"{result['burn_time']:.3f}", "s"),
                    ("Max Thrust", f"{result['max_thrust']:.2f}", "N"),
                ]
                for i, (name, val, unit) in enumerate(metrics):
                    with m_cols[i]:
                        st.markdown(f"""
                        <div class='result-card'>
                            <div class='result-label'>{name}</div>
                            <div class='result-value'>{val}</div>
                            <div class='result-unit'>{unit}</div>
                        </div>
                        """, unsafe_allow_html=True)

                if result.get("peak_pressure") is not None:
                    st.markdown(f"""
                    <div style='text-align:center; margin-top:1.5rem;'>
                        <span style='color:#a1a1aa; font-size:0.95rem; font-weight:500;'>Peak Pressure: </span>
                        <span style='color:#3b82f6; font-family:JetBrains Mono; font-size:1.2rem; font-weight:700;'>
                            {result['peak_pressure']:.3f} MPa
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                # Curves plot
                if result.get("thrust_curve") is not None and result.get("time_steps") is not None:
                    st.markdown("<hr style='border-color:#27272a; margin: 2rem 0;'>", unsafe_allow_html=True)
                    fig = create_performance_plotly(
                        result["time_steps"], result["thrust_curve"], result["pressure_curve"],
                        title_prefix="Predicted"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("ℹ️ Note: This model outputs scalar metrics only (no curve data).")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# --- REVERSE MODE UI ---
else:
    st.markdown("""
    <div class='info-box'>
        Upload an Excel (.xlsx) or CSV file with performance data.<br>
        Required columns: <b>Time (s)</b>, <b>Thrust (N)</b>, <b>Pressure (MPa)</b>.<br>
        Optional column: <b>ISP (s)</b>.
    </div>
    """, unsafe_allow_html=True)

    # Template download
    template_df = pd.DataFrame({
        "Time (s)": np.linspace(0, 2.0, 50),
        "Thrust (N)": np.sin(np.pi * np.linspace(0, 2.0, 50) / 2.0) * 500,
        "Pressure (MPa)": np.sin(np.pi * np.linspace(0, 2.0, 50) / 2.0) * 5,
    })

    tmpl_c1, tmpl_c2, _ = st.columns([1, 1, 2])
    with tmpl_c1:
        csv_bytes = template_df.to_csv(index=False).encode()
        st.download_button(
            "Download CSV Template", data=csv_bytes,
            file_name=f"template_{selected_grain}_reverse.csv",
            mime="text/csv", use_container_width=True,
        )
    with tmpl_c2:
        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
            template_df.to_excel(writer, index=False, sheet_name="Input")
        xlsx_buf.seek(0)
        st.download_button(
            "Download Excel Template", data=xlsx_buf.getvalue(),
            file_name=f"template_{selected_grain}_reverse.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.markdown("<hr style='border-color:#27272a; margin: 2rem 0;'>", unsafe_allow_html=True)

    # File uploader
    st.markdown("<div class='input-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Upload Performance Data</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your Excel or CSV file here",
        type=["xlsx", "xls", "csv"],
        help="File must contain columns: Time (s), Thrust (N), Pressure (MPa).",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    df_input = None
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df_input = pd.read_csv(uploaded)
            else:
                df_input = pd.read_excel(uploaded)
            st.success(f"File loaded — **{len(df_input)} rows & {len(df_input.columns)} columns**")
            with st.expander("Preview uploaded data", expanded=True):
                st.dataframe(df_input.head(15), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not read the file: {e}")

    # Run button
    run_col, _ = st.columns([1, 2])
    with run_col:
        run_rev = st.button("Run Prediction", key="run_reverse", disabled=(df_input is None))

    if run_rev and df_input is not None:
        with st.spinner("Processing data and running inverse prediction..."):
            try:
                t, thrust, pressure, isp_val = parse_uploaded_file(df_input)
                result_dims = predict_reverse(selected_grain, t, thrust, pressure, isp_val)

                st.markdown("<hr style='border-color:#27272a; margin: 2rem 0;'>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='panel-title' style='margin-bottom:1rem;'>
                    Predicted {selected_grain} Grain Dimensions (cm)
                </div>
                """, unsafe_allow_html=True)

                # Dimensions table
                dim_html = "<table class='dim-table'><tr><th>Dimension</th><th>Predicted Value</th><th>Unit</th></tr>"
                for name, val in result_dims.items():
                    unit = "—" if "Number" in name else "cm"
                    dim_html += f"<tr><td>{name}</td><td>{val:.4f}</td><td>{unit}</td></tr>"
                dim_html += "</table>"
                st.markdown(dim_html, unsafe_allow_html=True)

                # Download results
                res_df = pd.DataFrame(list(result_dims.items()), columns=["Dimension", "Predicted Value (cm)"])
                dl_c1, dl_c2, _ = st.columns([1, 1, 2])
                with dl_c1:
                    st.download_button(
                        "Download Results (CSV)",
                        data=res_df.to_csv(index=False).encode(),
                        file_name=f"results_{selected_grain}_reverse.csv",
                        mime="text/csv", use_container_width=True,
                    )
                with dl_c2:
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as w:
                        res_df.to_excel(w, index=False, sheet_name="Results")
                    buf.seek(0)
                    st.download_button(
                        "Download Results (Excel)",
                        data=buf.getvalue(),
                        file_name=f"results_{selected_grain}_reverse.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

                # Show processed input curves
                st.markdown("<hr style='border-color:#27272a; margin: 2rem 0;'>", unsafe_allow_html=True)
                _, t_100, p_100 = smooth_and_interpolate(t, thrust, pressure, 100)
                x_new = np.linspace(t[0], t[-1], 100)
                fig = create_performance_plotly(x_new, t_100, p_100, title_prefix="Input (Processed)")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    elif df_input is None and not run_rev:
        st.markdown("""
        <div style='text-align:center; padding:3rem;'>
            <div style='font-size:1rem; color:#71717a; font-weight:500;'>Upload a performance data file to enable prediction</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="enhanced-footer">
    <div>Grain Performance Predictor</div>
    <div class="footer-center">9 Forward &amp; 9 Reverse Models</div>
    <div>All geometric dimensions in <span class="footer-highlight">cm</span></div>
</div>
""", unsafe_allow_html=True)
