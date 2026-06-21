import os
import joblib
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# Lazy-loaded heavy imports (deferred to first use)
_tf = None
_savgol = None
_interp1d = None
_trapezoid_fn = None

BASE_DIR = Path(__file__).resolve().parent

def _get_tf():
    global _tf
    if _tf is None:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        _tf = tf
    return _tf

def _get_savgol():
    global _savgol
    if _savgol is None:
        from scipy.signal import savgol_filter
        _savgol = savgol_filter
    return _savgol

def _get_interp1d():
    global _interp1d
    if _interp1d is None:
        from scipy.interpolate import interp1d
        _interp1d = interp1d
    return _interp1d

def _trapezoid(y, x):
    """Compatibility wrapper for np.trapz / np.trapezoid."""
    global _trapezoid_fn
    if _trapezoid_fn is None:
        _trapezoid_fn = getattr(np, "trapezoid", np.trapz)
    return _trapezoid_fn(y, x)

def apply_keras_dense_patch():
    """Dynamically patch Dense layer to bypass 'quantization_config' argument."""
    try:
        tf = _get_tf()
        from tensorflow.keras.layers import Dense
        original_init = Dense.__init__
        def patched_init(self, *args, **kwargs):
            if 'quantization_config' in kwargs:
                kwargs.pop('quantization_config')
            original_init(self, *args, **kwargs)
        Dense.__init__ = patched_init
    except Exception:
        pass

@st.cache_resource(show_spinner=False)
def load_forward_assets(grain_type):
    """Load and cache forward model + scalers for the given grain type."""
    tf = _get_tf()
    apply_keras_dense_patch()
    d = BASE_DIR / "Forward Models"
    assets = {}

    if grain_type == "Bates":
        p = d / "Bates Models"
        assets["model"]  = tf.keras.models.load_model(str(p / "Bates_model.keras"), compile=False)
        assets["s_X"]    = joblib.load(str(p / "scaler_X.pkl"))
        assets["s_yt"]   = joblib.load(str(p / "scaler_thrust.pkl"))
        assets["s_yp"]   = joblib.load(str(p / "scaler_pressure.pkl"))
        assets["s_ys"]   = joblib.load(str(p / "scaler_scalars.pkl"))

    elif grain_type == "C":
        p = d / "C Models"
        assets["model"]    = tf.keras.models.load_model(str(p / "Co_Forward_Model.keras"), compile=False)
        assets["s_X"]      = joblib.load(str(p / "co_fwd_scaler_X.pkl"))
        assets["s_ys"]     = joblib.load(str(p / "co_fwd_scaler_scalars.pkl"))
        assets["max_vals"] = joblib.load(str(p / "co_fwd_max_values.pkl"))

    elif grain_type == "Conical":
        p = d / "Conical Models"
        assets["model"]    = tf.keras.models.load_model(str(p / "Conical_Forward_Model.keras"), compile=False)
        assets["s_X"]      = joblib.load(str(p / "con_fwd_scaler_X.pkl"))
        assets["s_ys"]     = joblib.load(str(p / "con_fwd_scaler_scalars.pkl"))
        assets["max_vals"] = joblib.load(str(p / "con_fwd_max_values.pkl"))

    elif grain_type == "D":
        p = d / "D Models"
        assets["model"] = tf.keras.models.load_model(str(p / "grain_d_model.keras"), compile=False)
        with open(str(p / "grain_d_scalers.pkl"), "rb") as f:
            sc = pickle.load(f)
        assets["s_X"]  = sc["s_X"]
        assets["s_yt"] = sc["s_yt"]
        assets["s_yp"] = sc["s_yp"]
        assets["s_ys"] = sc["s_ys"]

    elif grain_type == "Finocyl":
        p = d / "Finocyl Models"
        assets["model"]       = tf.keras.models.load_model(str(p / "finocyl_forward_v3_final.keras"), compile=False)
        assets["s_X"]         = joblib.load(str(p / "fin_v3_scaler_X.pkl"))
        assets["s_ys"]        = joblib.load(str(p / "fin_v3_scaler_ys.pkl"))
        assets["norm_params"] = joblib.load(str(p / "fin_v3_norm_params.pkl"))

    elif grain_type == "Moon":
        p = d / "Moon Models"
        assets["model"]    = tf.keras.models.load_model(str(p / "Moon_Forward_Model.keras"), compile=False)
        assets["s_X"]      = joblib.load(str(p / "moon_fwd_scaler_X.pkl"))
        assets["s_ys"]     = joblib.load(str(p / "moon_fwd_scaler_scalars.pkl"))
        assets["max_vals"] = joblib.load(str(p / "moon_fwd_max_values.pkl"))

    elif grain_type == "Road and Tube":
        p = d / "Road and Tube"
        assets["model"] = tf.keras.models.load_model(str(p / "rocket_motor_golden_rod_improved_fixed.keras"), compile=False)
        
        # Robust loading of separate scalers in Road and Tube forward
        scaler_X_path = p / "scaler_X (1).pkl" 
        scaler_t_path = p / "scaler_t (1).pkl" 
        scaler_p_path = p / "scaler_p (1).pkl" 
        scaler_Ys_path = p / "scaler_Ys (1).pkl" 
        
        assets["scaler_X"]  = joblib.load(str(scaler_X_path))
        assets["scaler_t"]  = joblib.load(str(scaler_t_path))
        assets["scaler_p"]  = joblib.load(str(scaler_p_path))
        assets["scaler_ys"] = joblib.load(str(scaler_Ys_path))

    elif grain_type == "Star":
        p = d / "Star Model"
        assets["model"]  = tf.keras.models.load_model(str(p / "Star_model.keras"), compile=False)
        assets["s_X"]    = joblib.load(str(p / "scaler_X.pkl"))
        assets["s_yt"]   = joblib.load(str(p / "scaler_thrust.pkl"))
        assets["s_yp"]   = joblib.load(str(p / "scaler_pressure.pkl"))
        assets["s_ys"]   = joblib.load(str(p / "scaler_scalars.pkl"))

    elif grain_type == "X":
        p = d / "X Models"
        assets["model"] = tf.keras.models.load_model(str(p / "rocket_motor_X_fixed.keras"), compile=False)
        assets["scaler_X"]    = joblib.load(str(p / "scaler_X (1).pkl"))
        assets["scaler_ys"]   = joblib.load(str(p / "scaler_ys (1).pkl"))
        assets["constants"]   = joblib.load(str(p / "constants.pkl"))
        

    return assets

@st.cache_resource(show_spinner=False)
def load_reverse_assets(grain_type):
    """Load and cache reverse model + scalers for the given grain type."""
    tf = _get_tf()
    apply_keras_dense_patch()
    d = BASE_DIR / "Reverse Models"
    assets = {}

    if grain_type == "Bates":
        p = d / "Bates Models"
        assets["model"] = tf.keras.models.load_model(str(p / "Bates_Inverse_Model_Final.keras"), compile=False)
        assets["s_X"]   = joblib.load(str(p / "scaler_dimensions_inv_final.pkl"))
        assets["s_yt"]  = joblib.load(str(p / "scaler_thrust_inv_final.pkl"))
        assets["s_yp"]  = joblib.load(str(p / "scaler_pressure_inv_final.pkl"))
        assets["s_ys"]  = joblib.load(str(p / "scaler_scalars_inv_final.pkl"))

    elif grain_type == "C":
        p = d / "C Models"
        assets["model"] = tf.keras.models.load_model(str(p / "C_Reverse.keras"), compile=False)
        assets["s_xs"]  = joblib.load(str(p / "c_rev_scaler_scalars.pkl"))
        assets["s_Y"]   = joblib.load(str(p / "c_rev_scaler_dims.pkl"))

    elif grain_type == "Conical":
        p = d / "Conical Models"
        assets["model"]    = tf.keras.models.load_model(str(p / "Conical_Reverse_Model.keras"), compile=False)
        assets["max_vals"] = joblib.load(str(p / "conical_rev_max_values.pkl"))
        assets["s_xs"]     = joblib.load(str(p / "conical_rev_scaler_scalars.pkl"))
        assets["s_Y"]      = joblib.load(str(p / "conical_rev_scaler_dims.pkl"))

    elif grain_type == "D":
        p = d / "D Models"
        assets["model"] = tf.keras.models.load_model(str(p / "GrainD_inverse_model.keras"), compile=False)
        assets["s_X"]   = joblib.load(str(p / "grainD_inv_scaler_dims.pkl"))
        assets["s_yt"]  = joblib.load(str(p / "grainD_inv_scaler_thrust.pkl"))
        assets["s_yp"]  = joblib.load(str(p / "grainD_inv_scaler_pressure.pkl"))
        assets["s_ys"]  = joblib.load(str(p / "grainD_inv_scaler_scalars.pkl"))

    elif grain_type == "Finocyl":
        p = d / "Finocyl Models"
        assets["model"]    = tf.keras.models.load_model(str(p / "Finocyl_Inverse_Model_v2.keras"), compile=False)
        assets["scaler_Y"] = joblib.load(str(p / "finocyl_inv_scaler_Y_v2.pkl"))
        assets["scaler_S"] = joblib.load(str(p / "finocyl_inv_scaler_S_v2.pkl"))

    elif grain_type == "Moon":
        p = d / "Moon Models"
        assets["model"]    = tf.keras.models.load_model(str(p / "Moon_Reverse_Model.keras"), compile=False)
        assets["max_vals"] = joblib.load(str(p / "moon_rev_max_values.pkl"))
        assets["s_xs"]     = joblib.load(str(p / "moon_rev_scaler_scalars.pkl"))
        assets["s_Y"]      = joblib.load(str(p / "moon_rev_scaler_dims.pkl"))

    elif grain_type == "Road and Tube":
        p = d / "Road and Tube"
        assets["model"]    = tf.keras.models.load_model(str(p / "inverse_rocket_model_fixed.keras"), compile=False)
        assets["scaler_X"] = joblib.load(str(p / "scaler_X_inverse.pkl"))
        assets["scaler_y"] = joblib.load(str(p / "scaler_y_inverse.pkl"))

    elif grain_type == "Star":
        p = d / "Star Models"
        assets["model"] = tf.keras.models.load_model(str(p / "Star_inverse_model.keras"), compile=False)
        assets["s_X"]   = joblib.load(str(p / "star_inv_scaler_dimensions.pkl"))
        assets["s_yt"]  = joblib.load(str(p / "star_inv_scaler_thrust.pkl"))
        assets["s_yp"]  = joblib.load(str(p / "star_inv_scaler_pressure.pkl"))
        assets["s_ys"]  = joblib.load(str(p / "star_inv_scaler_scalars.pkl"))

    elif grain_type == "X":
        p = d / "X Models"
        assets["model"] = tf.keras.models.load_model(str(p / "best_X_model_fixed.h5"), compile=False)
        assets["s_X"]   = joblib.load(str(p / "scaler_X_inverse.pkl"))
        assets["s_yt"]  = joblib.load(str(p / "scaler_y_inverse.pkl"))

    return assets

def smart_detect_columns(df):
    """Auto-detect time, thrust, pressure columns from any header naming."""
    col_map = {}
    for col in df.columns:
        c = str(col).lower().replace(" ", "").replace("_", "").replace("(", "").replace(")", "")
        if "time" in c or c == "t":
            col_map["time"] = col
        elif "thrust" in c or c == "f":
            col_map["thrust"] = col
        elif "pressure" in c or c == "p":
            col_map["pressure"] = col
        elif "isp" in c:
            col_map["isp"] = col
    return col_map

def parse_uploaded_file(df):
    """Parse an uploaded DataFrame: detect columns, clean, sort, extract arrays."""
    col_map = smart_detect_columns(df)
    required = ["time", "thrust", "pressure"]
    missing = [r for r in required if r not in col_map]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Detected columns: {list(df.columns)}. "
            f"Expected columns with 'Time', 'Thrust', 'Pressure' in the header."
        )
    df = df.sort_values(col_map["time"]).dropna(
        subset=[col_map["time"], col_map["thrust"], col_map["pressure"]]
    )
    t = df[col_map["time"]].values.astype(float)
    thrust = df[col_map["thrust"]].values.astype(float)
    pressure = df[col_map["pressure"]].values.astype(float)
    isp_val = float(df[col_map["isp"]].iloc[0]) if "isp" in col_map else None
    return t, thrust, pressure, isp_val

def smooth_and_interpolate(t, thrust, pressure, n_points=100):
    """Apply savgol smoothing and interpolate to n_points."""
    savgol = _get_savgol()
    interp1d = _get_interp1d()
    if len(thrust) > 7:
        thrust = savgol(thrust, 7, 3)
        pressure = savgol(pressure, 7, 3)
    x_new = np.linspace(t[0], t[-1], n_points)
    t_interp = interp1d(t, thrust, kind="linear", fill_value="extrapolate")(x_new)
    p_interp = interp1d(t, pressure, kind="linear", fill_value="extrapolate")(x_new)
    return x_new, t_interp, p_interp

def extract_scalar_features_finocyl(t, thrust, press):
    """Extract 6 scalar features for Finocyl inverse model (log-scaled)."""
    max_thrust    = np.max(thrust)
    max_press     = np.max(press)
    burn_time     = t[-1] 
    avg_thrust    = _trapezoid(thrust, t) / (burn_time + 1e-8)
    total_impulse = _trapezoid(thrust, t)
    idx_90    = np.argmax(thrust >= 0.9 * max_thrust)
    rise_time = t[idx_90] / (burn_time + 1e-8) if max_thrust > 0 else 0.5
    return np.array([
        np.log1p(max_thrust), np.log1p(max_press), np.log1p(burn_time),
        np.log1p(avg_thrust), np.log1p(total_impulse), rise_time,
    ], dtype=np.float32)

def compute_scalars_c(t, thrust, pressure):
    """Compute 5 scalar features for C reverse model."""
    return [
        float(np.max(thrust)),
        float(np.mean(thrust)),
        float(_trapezoid(thrust, t)),
        float(np.max(pressure)),
        float(t[-1]),
    ]
