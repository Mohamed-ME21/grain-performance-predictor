import numpy as np
from scipy.signal import savgol_filter
from collections import OrderedDict
from utils import (
    load_reverse_assets, _get_tf, _get_savgol, _get_interp1d,
    _trapezoid, smooth_and_interpolate,
    extract_scalar_features_finocyl, compute_scalars_c
)

REVERSE_DIM_LABELS = {
    "Bates":         ["Length", "Outer Diameter", "Core Diameter", "Throat Diameter", "Exit Diameter"],
    "C":             ["Length", "Diameter", "Slot Width", "Slot Offset", "Throat Diameter", "Exit Diameter"],
    "Conical":       ["Length", "Diameter", "Fwd Core Diameter", "Aft Core Diameter", "Throat Diameter", "Exit Diameter"],
    "D":             ["Length", "Diameter", "Slot Offset", "Throat Diameter", "Exit Diameter"],
    "Finocyl":       ["Diameter", "Length", "Core Diameter", "Number of Fins", "Fin Length", "Fin Width", "Throat Diameter", "Exit Diameter"],
    "Moon":          ["Length", "Diameter", "Core Diameter", "Core Offset", "Throat Diameter", "Exit Diameter"],
    "Road and Tube": ["Length", "Diameter", "Core Diameter", "Rod Diameter", "Support Diameter", "Throat Diameter", "Exit Diameter"],
    "Star":          ["Length", "Outer Diameter", "Number of Points", "Point Length", "Point Base Width", "Throat Diameter", "Exit Diameter"],
    "X":             ["Length", "Diameter", "Slot Length", "Slot Width", "Throat Diameter", "Exit Diameter"],
}

# Default Isp fallback values (s) used by each grain's reverse model.
# Exposed so the UI can display the exact value in the tooltip/label.
# Models that do not use Isp as a scalar feature have None.
REVERSE_DEFAULT_ISP = {
    "Bates":         170.1542764,
    "C":             None,           # C model derives all info from curves
    "Conical":       170.0,
    "D":             168.7509,
    "Finocyl":       174.8347,
    "Moon":          178.0197433,
    "Road and Tube": 157.3015,
    "Star":          170.0,
    "X":             175.1104,
}

def _rev_bates(t, thrust, pressure, isp_val):
    tf = _get_tf()
    # Fixed seed so MC-Dropout gives reproducible results across runs/machines
    tf.random.set_seed(42)
    np.random.seed(42)
    a = load_reverse_assets("Bates")
    _, t_100, p_100 = smooth_and_interpolate(t, thrust, pressure, 100)
    burn_time  = t[-1]
    max_thrust = float(np.max(thrust))
    total_imp  = float(_trapezoid(thrust, t))
    isp        = isp_val if isp_val else 170.1542764
    scalars = np.array([isp, total_imp, burn_time, max_thrust])
    t_sc = a["s_yt"].transform(t_100.reshape(1, -1))
    p_sc = a["s_yp"].transform(p_100.reshape(1, -1))
    s_sc = a["s_ys"].transform(scalars.reshape(1, -1))
    
    # MC Dropout
    t_tensor = tf.constant(t_sc, dtype=tf.float32)
    p_tensor = tf.constant(p_sc, dtype=tf.float32)
    s_tensor = tf.constant(s_sc, dtype=tf.float32)
    preds_list = []
    for _ in range(50):
        out = a["model"]([t_tensor, p_tensor, s_tensor], training=True)
        preds_list.append(out.numpy())
    pred_mean = np.stack(preds_list).mean(axis=0)
    dims = a["s_X"].inverse_transform(pred_mean)[0]
    dims = np.maximum(dims, 0.1)
    return dims

def _rev_c(t, thrust, pressure, isp_val):
    a = load_reverse_assets("C")
    interp1d = _get_interp1d()
    x_new = np.linspace(t[0], t[-1], 200)
    t_interp = interp1d(t, thrust, fill_value="extrapolate")(x_new)
    p_interp = interp1d(t, pressure, fill_value="extrapolate")(x_new)
    
    # Normalize curves
    t_interp = t_interp / (np.max(np.abs(t_interp)) + 1e-8)
    p_interp = p_interp / (np.max(np.abs(p_interp)) + 1e-8)
    curves = np.stack([t_interp, p_interp], axis=-1).reshape(1, 200, 2).astype(np.float32)
    
    scalars = compute_scalars_c(t, thrust, pressure)
    scalars_s = a["s_xs"].transform([scalars])
    pred_s = a["model"].predict([curves, scalars_s], verbose=0)
    dims = a["s_Y"].inverse_transform(pred_s)[0]
    
    BOUNDS_MIN = np.array([20, 6, 0.5, 1, 0.2, 1.2])
    BOUNDS_MAX = np.array([120, 20, 4, 7, 2.0, 3.0])
    dims = np.clip(dims, BOUNDS_MIN, BOUNDS_MAX)
    return dims

def _rev_conical(t, thrust, pressure, isp_val):
    a = load_reverse_assets("Conical")
    interp1d = _get_interp1d()
    # if len(thrust) > 7:
    #     thrust = savgol(thrust, 7, 3)
    #     pressure = savgol(pressure, 7, 3)
    x_new = np.linspace(t[0], t[-1], 100)
    t_100 = interp1d(t, thrust, kind="linear", fill_value="extrapolate")(x_new)
    p_100 = interp1d(t, pressure, kind="linear", fill_value="extrapolate")(x_new)
    xt_max = a["max_vals"]["xt_max"]
    xp_max = a["max_vals"]["xp_max"]
    
    burn_time     = t[-1] 
    max_thrust    = float(np.max(thrust))
    total_impulse = float(_trapezoid(thrust, t))
    isp = isp_val if isp_val else 170.0
    scalars = np.array([[isp, total_impulse, burn_time, max_thrust]])
    
    t_scaled = (t_100 / xt_max).reshape(1, -1)
    p_scaled = (p_100 / xp_max).reshape(1, -1)
    s_scaled = a["s_xs"].transform(scalars)
    
    pred = a["model"].predict([t_scaled, p_scaled, s_scaled], verbose=0)
    dims = a["s_Y"].inverse_transform(pred)[0]
    dims = np.maximum(dims, 0.01)
    return dims

def _rev_d(t, thrust, pressure, isp_val):
    a = load_reverse_assets("D")
    _, t_100, p_100 = smooth_and_interpolate(t, thrust, pressure, 100)
    burn_time     = t[-1] 
    max_thrust    = float(np.max(thrust))
    total_impulse = float(_trapezoid(thrust, t))
    isp           = isp_val if isp_val else 168.7509
    scalars = np.array([isp, total_impulse, burn_time, max_thrust])
    
    t_sc = a["s_yt"].transform(t_100.reshape(1, -1))
    p_sc = a["s_yp"].transform(p_100.reshape(1, -1))
    s_sc = a["s_ys"].transform(scalars.reshape(1, -1))
    
    pred = a["model"].predict([t_sc, p_sc, s_sc], verbose=0)
    dims = a["s_X"].inverse_transform(pred)[0]
    dims = np.maximum(dims, 0.1)
    return dims

def _rev_finocyl(t, thrust, pressure, isp_val):
    a = load_reverse_assets("Finocyl")
    interp1d = _get_interp1d()
    if len(thrust) > 7:
        thrust   = savgol_filter(thrust,   window_length=7, polyorder=3)
        pressure = savgol_filter(pressure, window_length=7, polyorder=3)
    x_new = np.linspace(t[0], t[-1], 100)
    t_100 = interp1d(t, thrust, kind="linear", fill_value="extrapolate")(x_new)
    p_100 = interp1d(t, pressure, kind="linear", fill_value="extrapolate")(x_new)
    xt_max = a["max_vals"]["xt_max"]
    xp_max = a["max_vals"]["xp_max"]
    
    burn_time     = t[-1] 
    max_thrust    = float(np.max(thrust))
    total_impulse = float(_trapezoid(thrust, t))
    isp = isp_val if isp_val else 174.8347
    scalars = np.array([[isp, total_impulse, burn_time, max_thrust]])
    
    t_scaled = (t_100 / xt_max).reshape(1, -1)
    p_scaled = (p_100 / xp_max).reshape(1, -1)
    s_scaled = a["s_xs"].transform(scalars)
    
    pred = a["model"].predict([t_scaled, p_scaled, s_scaled], verbose=0)
    dims = a["s_Y"].inverse_transform(pred)[0]
    dims = np.maximum(dims, 0.01)
    return dims

def _rev_moon(t, thrust, pressure, isp_val):
    a = load_reverse_assets("Moon")
    interp1d = _get_interp1d()
    # if len(thrust) > 7:
    #     thrust = savgol(thrust, 7, 3)
    #     pressure = savgol(pressure, 7, 3)
    x_new = np.linspace(t[0], t[-1], 100)
    t_100 = interp1d(t, thrust, kind="linear", fill_value="extrapolate")(x_new)
    p_100 = interp1d(t, pressure, kind="linear", fill_value="extrapolate")(x_new)
    xt_max = a["max_vals"]["xt_max"]
    xp_max = a["max_vals"]["xp_max"]
    
    isp = isp_val if isp_val else 178.0197433
    total_impulse = float(_trapezoid(thrust, t))
    burn_time = t[-1]
    max_thrust = float(np.max(thrust))
    scalars = np.array([[isp, total_impulse, burn_time, max_thrust]])
    
    t_scaled = (t_100 / xt_max).reshape(1, -1)
    p_scaled = (p_100 / xp_max).reshape(1, -1)
    s_scaled = a["s_xs"].transform(scalars)
    
    pred = a["model"].predict([t_scaled, p_scaled, s_scaled], verbose=0)
    dims = a["s_Y"].inverse_transform(pred)[0]
    dims = np.maximum(dims, 0.01)
    return dims

# EXIT_THROAT_RATIO used by the Road‑and‑Tube model to derive Exit from Throat
_ROD_EXIT_THROAT_RATIO = 1.5

def _rev_road_tube(t, thrust, pressure, isp_val):
    a = load_reverse_assets("Road and Tube")
    savgol   = _get_savgol()
    interp1d = _get_interp1d()
    thr = savgol(thrust, 7, 3) if len(thrust) > 7 else thrust
    prs = savgol(pressure, 7, 3) if len(pressure) > 7 else pressure

    t_new        = np.linspace(t[0], t[-1], 100)
    thrust_100   = interp1d(t, thr, kind="linear", fill_value="extrapolate")(t_new)
    pressure_100 = interp1d(t, prs, kind="linear", fill_value="extrapolate")(t_new)

    # Scalars — must match training order exactly
    total_impulse = float(_trapezoid(thrust, t))
    isp           = isp_val if isp_val else 157.3015
    max_thrust    = float(np.max(thrust))
    peak_pressure = float(np.max(pressure))
    burn_time     = float(t[-1] - t[0])
    avg_thrust    = float(np.mean(thrust))
    scalars = np.array([[total_impulse, isp, max_thrust,
                         peak_pressure, burn_time, avg_thrust]])

    t_sc = a["s_yt"].transform(thrust_100.reshape(1, -1))
    p_sc = a["s_yp"].transform(pressure_100.reshape(1, -1))
    s_sc = a["s_ys"].transform(scalars)

    pred_sc  = a["model"].predict([t_sc, p_sc, s_sc], verbose=0)
    raw_dims = a["s_X"].inverse_transform(pred_sc)[0]   # 5 dims: L, D, Core, Rod, Throat
    raw_dims = np.maximum(raw_dims, 0.1)

    # Insert Support_Diameter = 0.0 at index 4 (between Rod and Throat)
    # Then append Exit_Diameter = Throat × ratio
    throat_dia = raw_dims[-1]                            # Throat is last of the 5
    exit_dia   = throat_dia * _ROD_EXIT_THROAT_RATIO
    # Final order: L, D, Core, Rod, Support(0), Throat, Exit
    dims = np.concatenate([raw_dims[:4], [0.0], raw_dims[4:], [exit_dia]])
    return dims

def _rev_star(t, thrust, pressure, isp_val):
    a = load_reverse_assets("Star")
    _, t_100, p_100 = smooth_and_interpolate(t, thrust, pressure, 100)
    burn_time     = t[-1] 
    max_thrust    = float(np.max(thrust))
    total_impulse = float(_trapezoid(thrust, t))
    isp           = isp_val if isp_val else 170.0
    scalars = np.array([isp, total_impulse, burn_time, max_thrust])
    
    t_sc = a["s_yt"].transform(t_100.reshape(1, -1))
    p_sc = a["s_yp"].transform(p_100.reshape(1, -1))
    s_sc = a["s_ys"].transform(scalars.reshape(1, -1))
    
    pred = a["model"].predict([t_sc, p_sc, s_sc], verbose=0)
    dims = a["s_X"].inverse_transform(pred)[0]
    dims = np.maximum(dims, 0.1)
    return dims

# EXIT_THROAT_RATIO used by the X model to derive Exit from Throat
_X_EXIT_THROAT_RATIO = 1.5

def _rev_x(t, thrust, pressure, isp_val):
    a = load_reverse_assets("X")
    savgol   = _get_savgol()
    interp1d = _get_interp1d()
    thr = savgol(thrust, 7, 3) if len(thrust) > 7 else thrust
    prs = savgol(pressure, 7, 3) if len(pressure) > 7 else pressure

    t_new        = np.linspace(t[0], t[-1], 100)
    thrust_100   = interp1d(t, thr, kind="linear", fill_value="extrapolate")(t_new)
    pressure_100 = interp1d(t, prs, kind="linear", fill_value="extrapolate")(t_new)

    # Scalars — must match training order exactly
    total_impulse = float(_trapezoid(thrust, t))
    isp           = isp_val if isp_val else 175.1104
    max_thrust    = float(np.max(thrust))
    peak_pressure = float(np.max(pressure))
    burn_time     = float(t[-1] - t[0])
    avg_thrust    = float(np.mean(thrust))
    scalars = np.array([[total_impulse, isp, max_thrust,
                         peak_pressure, burn_time, avg_thrust]])

    t_sc = a["s_yt"].transform(thrust_100.reshape(1, -1))
    p_sc = a["s_yp"].transform(pressure_100.reshape(1, -1))
    s_sc = a["s_ys"].transform(scalars)

    pred_sc   = a["model"].predict([t_sc, p_sc, s_sc], verbose=0)
    pred_dims = a["s_X"].inverse_transform(pred_sc)[0]
    pred_dims = np.maximum(pred_dims, 0.1)

    # Exit diameter is derived from throat diameter (last of the 5 predicted dims)
    throat_dia = pred_dims[-1]
    exit_dia   = throat_dia * _X_EXIT_THROAT_RATIO
    dims = np.append(pred_dims, exit_dia)
    return dims

REV_DISPATCH = {
    "Bates": _rev_bates, "C": _rev_c, "Conical": _rev_conical,
    "D": _rev_d, "Finocyl": _rev_finocyl, "Moon": _rev_moon,
    "Road and Tube": _rev_road_tube, "Star": _rev_star, "X": _rev_x,
}

def predict_reverse(grain_type, t, thrust, pressure, isp_val):
    """Predict geometric dimensions based on performance curves."""
    dims = REV_DISPATCH[grain_type](t, thrust, pressure, isp_val)
    labels = REVERSE_DIM_LABELS[grain_type]
    result = OrderedDict()
    for i, lbl in enumerate(labels):
        if i < len(dims):
            result[lbl] = float(dims[i])
    return result