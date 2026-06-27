import numpy as np
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
    "Finocyl":       None,           # Finocyl uses log-scaled curve features only
    "Moon":          178.0197433,
    "Road and Tube": None,           # Road and Tube concatenates raw curves only
    "Star":          170.0,
    "X":             None,           # X model concatenates raw curves only
}

def _rev_bates(t, thrust, pressure, isp_val):
    tf = _get_tf()
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
    savgol = _get_savgol()
    interp1d = _get_interp1d()
    NUM_POINTS = 128
    thrust  = np.clip(thrust, 0, None)
    pressure = np.clip(pressure, 0, None)
    
    scalars = extract_scalar_features_finocyl(t, thrust, pressure)
    scalars_s = a["scaler_S"].transform(scalars.reshape(1, -1))
    
    t_norm = (t - t[0]) / (t[-1] - t[0])
    t_new = np.linspace(0, 1, NUM_POINTS)
    thr_r = np.clip(interp1d(t_norm, thrust, kind="linear", fill_value="extrapolate")(t_new), 0, None)
    prs_r = np.clip(interp1d(t_norm, pressure, kind="linear", fill_value="extrapolate")(t_new), 0, None)
    
    thr_r /= (np.max(thr_r) + 1e-8)
    prs_r /= (np.max(prs_r) + 1e-8)
    thr_r = np.clip(savgol(thr_r, 15, 3), 0, None)
    prs_r = np.clip(savgol(prs_r, 15, 3), 0, None)
    
    pred_s = a["model"].predict(
        [thr_r.reshape(1, NUM_POINTS), prs_r.reshape(1, NUM_POINTS), scalars_s],
        verbose=0,
    )
    dims = a["scaler_Y"].inverse_transform(pred_s)[0]
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

def _rev_road_tube(t, thrust, pressure, isp_val):
    a = load_reverse_assets("Road and Tube")
    savgol = _get_savgol()
    interp1d = _get_interp1d()
    thr = savgol(thrust, 7, 3) if len(thrust) > 7 else thrust
    prs = savgol(pressure, 7, 3) if len(pressure) > 7 else pressure
    t_new = np.linspace(t[0], t[-1], 100)
    thr_interp = interp1d(t, thr, kind="linear", fill_value="extrapolate")(t_new)
    prs_interp = interp1d(t, prs, kind="linear", fill_value="extrapolate")(t_new)
    
    X_new = np.concatenate([thr_interp, prs_interp]).reshape(1, -1)
    X_scaled = a["scaler_X"].transform(X_new)
    y_pred_scaled = a["model"].predict(X_scaled, verbose=0)
    dims = a["scaler_y"].inverse_transform(y_pred_scaled)[0]
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

def _rev_x(t, thrust, pressure, isp_val):
    a = load_reverse_assets("X")
    savgol = _get_savgol()
    interp1d = _get_interp1d()
    thr = savgol(thrust, 7, 3) if len(thrust) > 7 else thrust
    prs = savgol(pressure, 7, 3) if len(pressure) > 7 else pressure
    t_new = np.linspace(t[0], t[-1], 100)
    thr_interp = interp1d(t, thr, kind="linear", fill_value="extrapolate")(t_new)
    prs_interp = interp1d(t, prs, kind="linear", fill_value="extrapolate")(t_new)
    
    X_input = np.concatenate([thr_interp, prs_interp]).reshape(1, -1)
    X_scaled = a["s_X"].transform(X_input)
    y_pred_scaled = a["model"].predict(X_scaled, verbose=0)
    dims = a["s_yt"].inverse_transform(y_pred_scaled)[0]
    
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
