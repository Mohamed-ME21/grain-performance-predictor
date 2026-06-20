import numpy as np
from utils import load_forward_assets, _trapezoid, _get_savgol

def _fwd_bates(vals):
    a = load_forward_assets("Bates")
    user = np.array([vals], dtype=float)
    user_s = a["s_X"].transform(user)
    preds = a["model"].predict(user_s, verbose=0)
    thrust   = a["s_yt"].inverse_transform(preds[0])[0]
    pressure = a["s_yp"].inverse_transform(preds[1])[0]
    scalars  = a["s_ys"].inverse_transform(preds[2])[0]
    bt = max(scalars[2], 0.01)
    ts = np.linspace(0, bt, len(thrust))
    savgol = _get_savgol()
    if len(thrust) > 15:
        thrust   = savgol(thrust, 15, 3)
        pressure = savgol(pressure, 15, 3)
    return {
        "isp": scalars[0], "total_impulse": _trapezoid(thrust, ts),
        "burn_time": bt, "max_thrust": scalars[3],
        "thrust_curve": thrust, "pressure_curve": pressure, "time_steps": ts,
    }

def _fwd_c(vals):
    a = load_forward_assets("C")
    user = np.array([vals], dtype=float)
    user_s = a["s_X"].transform(user)
    preds = a["model"].predict(user_s, verbose=0)
    yt_max = a["max_vals"]["yt_max"]
    yp_max = a["max_vals"]["yp_max"]
    thrust   = preds[0][0] * yt_max
    pressure = preds[1][0] * yp_max
    scalars  = a["s_ys"].inverse_transform(preds[2])[0]
    isp = max(scalars[0], 0.01)
    bt  = max(scalars[1], 0.01)
    ts = np.linspace(0, bt, len(thrust))
    return {
        "isp": isp, "total_impulse": _trapezoid(thrust, ts),
        "burn_time": bt, "max_thrust": float(np.max(thrust)),
        "thrust_curve": thrust, "pressure_curve": pressure, "time_steps": ts,
    }

def _fwd_conical(vals):
    a = load_forward_assets("Conical")
    user = np.array([vals], dtype=float)
    user_s = a["s_X"].transform(user)
    preds = a["model"].predict(user_s, verbose=0)
    yt_max = a["max_vals"]["yt_max"]
    yp_max = a["max_vals"]["yp_max"]
    thrust   = preds[0][0] * yt_max
    pressure = preds[1][0] * yp_max
    scalars  = a["s_ys"].inverse_transform(preds[2])[0]
    isp = max(scalars[0], 0.01)
    bt  = max(scalars[2], 0.01)
    ts = np.linspace(0, bt, len(thrust))
    savgol = _get_savgol()
    thrust   = np.clip(savgol(thrust, 7, 2), 0, None)
    pressure = np.clip(savgol(pressure, 7, 2), 0, None)
    # Tail cutoff
    peak_idx = np.argmax(thrust)
    tail = thrust[peak_idx:]
    cutoff = np.argmax(tail < 0.02 * thrust.max())
    if cutoff > 0:
        thrust[peak_idx + cutoff:] = 0
        pressure[peak_idx + cutoff:] = 0
    return {
        "isp": isp, "total_impulse": _trapezoid(thrust, ts),
        "burn_time": bt, "max_thrust": float(np.max(thrust)),
        "thrust_curve": thrust, "pressure_curve": pressure, "time_steps": ts,
    }

def _fwd_d(vals):
    a = load_forward_assets("D")
    user = np.array([vals], dtype=float)
    user_s = a["s_X"].transform(user)
    preds = a["model"].predict(user_s, verbose=0)
    thrust   = a["s_yt"].inverse_transform(preds[0])[0]
    pressure = a["s_yp"].inverse_transform(preds[1])[0]
    scalars  = a["s_ys"].inverse_transform(preds[2])[0]
    bt = max(scalars[2], 0.01)
    ts = np.linspace(0, bt, len(thrust))
    return {
        "isp": scalars[0], "total_impulse": _trapezoid(thrust, ts),
        "burn_time": bt, "max_thrust": scalars[3],
        "thrust_curve": thrust, "pressure_curve": pressure, "time_steps": ts,
    }

def _fwd_finocyl(vals):
    a = load_forward_assets("Finocyl")
    N_POINTS = 200
    inp = np.array([vals], dtype=np.float32)
    inp_s = a["s_X"].transform(inp)
    preds = a["model"].predict(inp_s, verbose=0)
    T_MAX = a["norm_params"]["thrust_max"]
    P_MAX = a["norm_params"]["pres_max"]
    thrust   = np.clip(preds[0][0] * T_MAX, 0, None)
    pressure = np.clip(preds[1][0] * P_MAX, 0, None)
    savgol = _get_savgol()
    thrust   = np.clip(savgol(thrust, 11, 3), 0, None)
    pressure = np.clip(savgol(pressure, 11, 3), 0, None)
    sc_log  = a["s_ys"].inverse_transform(preds[2])
    sc_vals = np.expm1(sc_log)[0]
    bt = max(float(sc_vals[0]), 0.1)
    isp = max(float(sc_vals[1]), 1.0)
    ts = np.linspace(0, bt, N_POINTS)
    return {
        "isp": isp, "total_impulse": _trapezoid(thrust, ts),
        "burn_time": bt, "max_thrust": float(np.max(thrust)),
        "peak_pressure": max(float(sc_vals[4]), 0.01) if len(sc_vals) > 4 else float(np.max(pressure)),
        "thrust_curve": thrust, "pressure_curve": pressure, "time_steps": ts,
    }

def _fwd_moon(vals):
    a = load_forward_assets("Moon")
    user = np.array([vals], dtype=float)
    user_s = a["s_X"].transform(user)
    preds = a["model"].predict(user_s, verbose=0)
    yt_max = a["max_vals"]["yt_max"]
    yp_max = a["max_vals"]["yp_max"]
    thrust   = preds[0][0] * yt_max
    pressure = preds[1][0] * yp_max
    scalars  = a["s_ys"].inverse_transform(preds[2])[0]
    isp = max(scalars[0], 0.01)
    bt  = max(scalars[2], 0.01)
    ts = np.linspace(0, bt, len(thrust))
    savgol = _get_savgol()
    thrust   = np.clip(savgol(thrust, 7, 2), 0, None)
    pressure = np.clip(savgol(pressure, 7, 2), 0, None)
    peak_idx = np.argmax(thrust)
    tail = thrust[peak_idx:]
    cutoff = np.argmax(tail < 0.02 * thrust.max())
    if cutoff > 0:
        thrust[peak_idx + cutoff:] = 0
        pressure[peak_idx + cutoff:] = 0
    return {
        "isp": isp, "total_impulse": _trapezoid(thrust, ts),
        "burn_time": bt, "max_thrust": float(np.max(thrust)),
        "thrust_curve": thrust, "pressure_curve": pressure, "time_steps": ts,
    }

def _fwd_road_tube(vals):
    a = load_forward_assets("Road and Tube")
    
    # Feature engineering: append throat_area and exit_area
    throat_dia = vals[5]
    exit_dia = vals[6]
    throat_area = np.pi * (throat_dia / 2.0)**2
    exit_area = np.pi * (exit_dia / 2.0)**2
    
    full_vals = vals + [throat_area, exit_area]
    user = np.array([full_vals], dtype=float)
    user_s = a["scaler_X"].transform(user)
    preds = a["model"].predict(user_s, verbose=0)
    thrust   = a["scaler_t"].inverse_transform(preds[0])[0]
    pressure = a["scaler_p"].inverse_transform(preds[1])[0]
    scalars  = a["scaler_ys"].inverse_transform(preds[2])[0]
    bt = max(scalars[2], 0.01)
    ts = np.linspace(0, bt, len(thrust))
    savgol = _get_savgol()
    if len(thrust) > 15:
        thrust   = np.clip(savgol(thrust, 15, 3), 0, None)
        pressure = np.clip(savgol(pressure, 15, 3), 0, None)
    return {
        "isp": scalars[0], "total_impulse": _trapezoid(thrust, ts),
        "burn_time": bt, "max_thrust": scalars[3],
        "peak_pressure": float(np.max(pressure)),
        "thrust_curve": thrust, "pressure_curve": pressure, "time_steps": ts,
    }

def _fwd_star(vals):
    a = load_forward_assets("Star")
    user = np.array([vals], dtype=float)
    user_s = a["s_X"].transform(user)
    preds = a["model"].predict(user_s, verbose=0)
    thrust   = a["s_yt"].inverse_transform(preds[0])[0]
    pressure = a["s_yp"].inverse_transform(preds[1])[0]
    scalars  = a["s_ys"].inverse_transform(preds[2])[0]
    bt = max(scalars[2], 0.01)
    ts = np.linspace(0, bt, len(thrust))
    savgol = _get_savgol()
    if len(thrust) > 15:
        thrust   = np.clip(savgol(thrust, 15, 3), 0, None)
        pressure = np.clip(savgol(pressure, 15, 3), 0, None)
    return {
        "isp": scalars[0], "total_impulse": _trapezoid(thrust, ts),
        "burn_time": bt, "max_thrust": scalars[3],
        "thrust_curve": thrust, "pressure_curve": pressure, "time_steps": ts,
    }

def _fwd_x(vals):
    a = load_forward_assets("X")
    user = np.array([vals], dtype=float)
    user_s = a["scaler_X"].transform(user)
    preds = a["model"].predict(user_s, verbose=0)
    yt_max = a["constants"]["max_thrust_global"]
    yp_max = a["constants"]["max_pressure_global"]
    thrust   = preds[0][0] * yt_max
    pressure = preds[1][0] * yp_max
    scalars  = a["scaler_ys"].inverse_transform(preds[2])[0]
    isp = max(scalars[0], 0.01)
    bt  = max(scalars[2], 0.01)
    ts = np.linspace(0, bt, len(thrust))
    return {
        "isp": isp, "total_impulse": _trapezoid(thrust, ts),
        "burn_time": bt, "max_thrust": float(np.max(thrust)),
        "peak_pressure": scalars[4],
        "thrust_curve": thrust, "pressure_curve": pressure, "time_steps": ts,
    }

FWD_DISPATCH = {
    "Bates": _fwd_bates, "C": _fwd_c, "Conical": _fwd_conical,
    "D": _fwd_d, "Finocyl": _fwd_finocyl, "Moon": _fwd_moon,
    "Road and Tube": _fwd_road_tube, "Star": _fwd_star, "X": _fwd_x,
}

def predict_forward(grain_type, input_values):
    """Predict performance metrics based on input dimensions and grain type."""
    vals = list(input_values.values())
    return FWD_DISPATCH[grain_type](vals)
