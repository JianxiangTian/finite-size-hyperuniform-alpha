#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def load_table(path: Path, min_cols: int) -> np.ndarray:
    x = np.loadtxt(path, comments="#")
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] < min_cols:
        raise RuntimeError(f"{path} must have at least {min_cols} columns")
    return x.astype(float)


def positive(*xs: np.ndarray) -> np.ndarray:
    m = np.ones(xs[0].shape, dtype=bool)
    for x in xs:
        m &= np.isfinite(x) & (x > 0)
    return m


def fit_loglog(x: np.ndarray, y: np.ndarray):
    lx = np.log10(x)
    ly = np.log10(y)
    a, b = np.polyfit(lx, ly, 1)
    r = ly - (a * lx + b)
    rmse = float(np.sqrt(np.mean(r * r)))
    var = float(np.sum(r * r) / max(1, len(x) - 2))
    return float(a), float(b), rmse, var


def scale(vals):
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.finfo(float).eps
    y = float(np.median(x))
    return y if y > 0 and np.isfinite(y) else np.finfo(float).eps


def moving_average_1d(y: np.ndarray, window: int) -> np.ndarray:
    """Simple centered moving average used only for locating the pre-peak bound."""
    w = max(1, int(window))
    if w <= 1 or y.size < 3:
        return y.astype(float).copy()
    if w % 2 == 0:
        w += 1
    pad = w // 2
    yp = np.pad(y.astype(float), (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(yp, kernel, mode="valid")


def first_local_log_slope_drop(x: np.ndarray, y: np.ndarray, p: dict, min_right: int) -> Optional[int]:
    """
    Find the first point where the low-k branch has crossed toward a plateau.

    This is only a guard against selecting windows in the post-crossover S(k)~O(1) region.
    It does not determine the final fitting window. The final window is still chosen by the
    regularized RMSE + boundary-stability score inside the admissible low-k branch.
    """
    threshold = p.get("sk_plateau_slope_threshold", None)
    if threshold is None:
        return None
    threshold = float(threshold)

    min_y = float(p.get("sk_plateau_min_y", 0.50))
    w = int(p.get("sk_plateau_slope_window", 5))
    if w % 2 == 0:
        w += 1
    if w < 5:
        w = 5
    if len(x) < w:
        return None

    half = w // 2
    for c in range(max(half, min_right), len(x) - half):
        if y[c] < min_y:
            continue
        try:
            a, _, _, _ = fit_loglog(x[c - half:c + half + 1], y[c - half:c + half + 1])
        except Exception:
            continue
        if np.isfinite(a) and a <= threshold:
            return c
    return None


def find_sk_admissible_right_bound(x: np.ndarray, y: np.ndarray, p: dict) -> Tuple[int, str]:
    """
    Determine the largest index allowed for S(k) fitting.

    The default mode restricts candidate windows to the pre-peak or pre-plateau low-k
    branch, so that the regularized fit is not attracted by the later S(k)~1 region.

    Config options under analysis:
        sk_right_bound_mode:
            "pre_peak"  : default. Use the earliest of first local peak, plateau slope drop,
                          optional ka max, and optional index max. Exclude the detected
                          peak/crossover point itself.
            "peak"      : legacy mode. Use the global maximum index as the right bound.
            "all"       : scan all positive points.
            "manual"    : use sk_right_bound_index_1based or sk_right_bound_ka_max.

        sk_peak_smooth_window: default 3.
        sk_peak_min_index_1based: default sk_min_points + 1. Avoids treating the first
                                  few noisy points as a peak.
        sk_plateau_slope_threshold: optional, e.g. 0.15. Disabled by default unless set.
        sk_plateau_min_y: default 0.50. Used with sk_plateau_slope_threshold.
        sk_right_bound_ka_max: optional hard upper bound in ka.
    """
    n = len(x)
    if n == 0:
        raise RuntimeError("empty S(k) data after filtering positive values")

    min_points = int(p.get("sk_min_points", 5))
    min_allowed_right = min(max(min_points - 1, 0), n - 1)
    mode = str(p.get("sk_right_bound_mode", "pre_peak")).lower()

    if mode in ("all", "none", "off"):
        return n - 1, "all_points"

    if mode == "peak":
        peak = int(np.argmax(y))
        right = max(min_allowed_right, min(peak, n - 1))
        return right, f"global_peak_included_at_index_{peak + 1}"

    if mode == "manual":
        bounds = []
        if "sk_right_bound_index_1based" in p:
            bounds.append(int(p["sk_right_bound_index_1based"]) - 1)
        if "sk_right_bound_ka_max" in p:
            kk = float(p["sk_right_bound_ka_max"])
            ind = np.where(x <= kk)[0]
            if ind.size > 0:
                bounds.append(int(ind[-1]))
        if not bounds:
            raise RuntimeError("manual S(k) right bound requested, but no manual bound was provided")
        right = max(min_allowed_right, min(min(bounds), n - 1))
        return right, "manual_bound"

    # Default physical mode: pre-peak or pre-plateau low-k branch.
    bounds: List[Tuple[int, str]] = []

    # Optional hard ka cutoff, useful for systems where the first peak is broad or absent.
    if "sk_right_bound_ka_max" in p:
        kk = float(p["sk_right_bound_ka_max"])
        ind = np.where(x <= kk)[0]
        if ind.size > 0:
            bounds.append((int(ind[-1]), f"ka_max_{kk:g}"))

    # First local peak on a lightly smoothed curve. We exclude the peak itself.
    smooth_w = int(p.get("sk_peak_smooth_window", 3))
    ys = moving_average_1d(y, smooth_w)
    min_peak_1based = int(p.get("sk_peak_min_index_1based", min_points + 1))
    min_peak_idx = max(min_peak_1based - 1, min_allowed_right)
    peak_min_y = float(p.get("sk_peak_min_y", 0.0))

    first_peak = None
    for i in range(max(1, min_peak_idx), n - 1):
        if ys[i] >= ys[i - 1] and ys[i] >= ys[i + 1] and ys[i] >= peak_min_y:
            first_peak = i
            break
    if first_peak is not None:
        bounds.append((max(min_allowed_right, first_peak - 1), f"before_first_local_peak_index_{first_peak + 1}"))

    # Optional plateau/crossover detector based on local log slope becoming small.
    plateau_idx = first_local_log_slope_drop(x, y, p, min_allowed_right)
    if plateau_idx is not None:
        bounds.append((max(min_allowed_right, plateau_idx - 1), f"before_plateau_slope_drop_index_{plateau_idx + 1}"))

    if not bounds:
        # If the curve is still rising and no cutoff was found, use all available points.
        return n - 1, "no_peak_or_plateau_detected"

    right, reason = min(bounds, key=lambda t: t[0])
    right = max(min_allowed_right, min(right, n - 1))
    return right, reason


def analyze_sk(path: Path, p: dict):
    d = load_table(path, 2)
    x, y = d[:, 0], d[:, 1]
    m = positive(x, y)
    x, y = x[m], y[m]

    right, right_reason = find_sk_admissible_right_bound(x, y, p)

    rows = []
    logx = np.log10(x)
    min_points = int(p["sk_min_points"])
    for i in range(0, right - min_points + 2):
        for j in range(i + min_points - 1, right + 1):
            span = float(logx[j] - logx[i])
            if span < float(p["sk_min_log10_span"]):
                continue
            a, b, rmse, _ = fit_loglog(x[i:j + 1], y[i:j + 1])
            # Boundary sensitivity is evaluated for every admissible window,
            # including the shortest five-point windows.  The endpoint-deleted
            # fits then contain four points and remain well defined.
            al = fit_loglog(x[i + 1:j + 1], y[i + 1:j + 1])[0]
            ar = fit_loglog(x[i:j], y[i:j])[0]
            da = max(abs(a - al), abs(a - ar))
            rows.append(dict(
                i_left=i + 1,
                i_right=j + 1,
                n_points=j - i + 1,
                ka_min=x[i],
                ka_max=x[j],
                log10_span=span,
                alpha_SK=a,
                RMSE_log=rmse,
                Delta_alpha=da,
                sk_admissible_right_index=right + 1,
                sk_admissible_right_ka=x[right],
                sk_right_bound_reason=right_reason,
            ))
    if not rows:
        raise RuntimeError(
            "no valid S(k) window inside the admissible pre-peak/pre-plateau low-k range; "
            "try reducing sk_min_points or sk_min_log10_span, or use sk_right_bound_mode='manual'"
        )
    r0 = scale([r["RMSE_log"] for r in rows])
    d0 = scale([r["Delta_alpha"] for r in rows])
    for r in rows:
        r["Q_SK"] = r["RMSE_log"] / r0 + float(p["sk_eta"]) * r["Delta_alpha"] / d0
    return sorted(rows, key=lambda z: z["Q_SK"])[0]


def local_exponent(x: np.ndarray, y: np.ndarray, w: int):
    if w % 2 == 0:
        w += 1
    h = w // 2
    idx, cx, pe, err = [], [], [], []
    for c in range(h, len(x) - h):
        sl = slice(c - h, c + h + 1)
        a, _, rmse, _ = fit_loglog(x[sl], y[sl])
        idx.append(c + 1)
        cx.append(x[c])
        pe.append(a)
        err.append(rmse)
    return np.asarray(idx), np.asarray(cx), np.asarray(pe), np.asarray(err)


def analyze_nv(path: Path, p: dict):
    d = load_table(path, 2)
    x, y = d[:, 0], d[:, 1]
    m = positive(x, y)
    x, y = x[m], y[m]
    n_use = min(int(p["nv_n_use"]), len(x))
    x, y = x[:n_use], y[:n_use]
    idx, cx, pe, err = local_exponent(x, y, int(p["nv_local_window"]))
    rows = []
    mmin = int(p["nv_min_plateau_points"])
    smin, smax = int(p["nv_search_point_min"]), min(int(p["nv_search_point_max"]), n_use)
    for i in range(0, len(cx) - mmin + 1):
        for j in range(i + mmin - 1, len(cx)):
            if idx[i] < smin or idx[j] > smax:
                continue
            span = float(np.log10(cx[j]) - np.log10(cx[i]))
            if span < float(p["nv_min_log10_span"]):
                continue
            rows.append(dict(idx_min=int(idx[i]), idx_max=int(idx[j]), n_center_points=j - i + 1, Rmin_over_a=cx[i], Rmax_over_a=cx[j], log10_span=span, mean_p_eff=float(np.mean(pe[i:j + 1])), std_p_eff=float(np.std(pe[i:j + 1], ddof=1)), mean_local_RMSE=float(np.mean(err[i:j + 1]))))
    if not rows:
        raise RuntimeError("no valid number-variance plateau")
    p0 = scale([r["std_p_eff"] for r in rows])
    e0 = scale([r["mean_local_RMSE"] for r in rows])
    for r in rows:
        r["S_plateau"] = r["std_p_eff"] / p0 + float(p["nv_eta"]) * r["mean_local_RMSE"] / e0
    best = sorted(rows, key=lambda z: z["S_plateau"])[0]
    mp = best["mean_p_eff"]
    if mp <= float(p["nv_p_i_max"]):
        cls, alpha = "Class I-like", np.nan
    elif mp >= float(p["nv_p_iii_min"]):
        cls, alpha = "Class III-like", 2.0 - mp
    else:
        cls, alpha = "Class II-like", np.nan
    best["class"] = cls
    best["alpha_NV"] = alpha
    return best


def slope_var(x: np.ndarray, y: np.ndarray) -> float:
    lx = np.log10(x)
    _, _, _, rv = fit_loglog(x, y)
    vx = float(np.var(lx))
    return rv / (vx * max(1, len(x) - 1)) if vx > 0 else np.inf


def alpha_from_slope(m: float) -> float:
    return -2.0 * m - 2.0


def local_slope(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    if w % 2 == 0:
        w += 1
    h = w // 2
    out = np.full(len(x), np.nan)
    for c in range(h, len(x) - h):
        out[c] = fit_loglog(x[c - h:c + h + 1], y[c - h:c + h + 1])[0]
    return out


def spread_fit(tau, y, i, j, key, val):
    m, b, rmse, rv = fit_loglog(tau[i:j + 1], y[i:j + 1])
    return dict(i0=i + 1, i1=j + 1, n_points=j - i + 1, tau_min=tau[i], tau_max=tau[j], num_decades=float(np.log10(tau[j] / tau[i])), slope_m=m, alpha_SP=alpha_from_slope(m), RMSE_log=rmse, **{key: val})


def analyze_spread(
    path: Path,
    p: dict,
    raw_max_factor_over_trust: float,
):
    d = load_table(path, 3)
    tau, y = d[:, 1], d[:, 2]
    m = positive(tau, y)
    tau, y = tau[m], y[m]
    order = np.argsort(tau)
    tau, y = tau[order], y[order]
    raw_factor = float(raw_max_factor_over_trust)
    if raw_factor <= 0.0:
        raise ValueError("raw_max_factor_over_trust must be positive")
    tau_trust = float(np.max(tau) / raw_factor)
    valid = np.where(tau <= tau_trust)[0]
    gbest, gval = None, np.inf
    for a in range(len(valid)):
        i = int(valid[a])
        if tau[i] < float(p["spread_global_tau0_min"]):
            continue
        for b in range(a + int(p["spread_global_min_points"]) - 1, len(valid)):
            j = int(valid[b])
            if np.log10(tau[j] / tau[i]) < float(p["spread_global_min_fit_decades"]):
                continue
            val = slope_var(tau[i:j + 1], y[i:j + 1])
            if val < gval:
                gval = val
                gbest = spread_fit(tau, y, i, j, "min_slope_var", val)
    if gbest is None:
        raise RuntimeError("no global spreadability window")

    sl = local_slope(tau[valid], y[valid], int(p["spread_plateau_local_slope_window"]))
    ae = np.asarray([alpha_from_slope(s) if np.isfinite(s) else np.nan for s in sl])
    rows = []
    min_points = int(p["spread_plateau_min_points"])
    for a in range(len(valid)):
        i = int(valid[a])
        if tau[i] < float(p["spread_plateau_tau0_min"]):
            continue
        for b in range(a + min_points - 1, len(valid)):
            j = int(valid[b])
            dec = float(np.log10(tau[j] / tau[i]))
            if dec < float(p["spread_plateau_min_fit_decades"]):
                continue
            local = ae[a:b + 1]
            local = local[np.isfinite(local)]
            if len(local) < max(3, min_points // 2):
                continue
            mfit, _, rmse, _ = fit_loglog(tau[i:j + 1], y[i:j + 1])
            rows.append(dict(i0=i + 1, i1=j + 1, n_points=j - i + 1, tau_min=tau[i], tau_max=tau[j], num_decades=dec, slope_m=mfit, alpha_SP=alpha_from_slope(mfit), RMSE_log=rmse, mean_alpha_eff=float(np.mean(local)), std_alpha_eff=float(np.std(local, ddof=1))))
    if not rows:
        raise RuntimeError("no plateau spreadability window")
    r0 = scale([r["RMSE_log"] for r in rows])
    a0 = scale([r["std_alpha_eff"] for r in rows])
    for r in rows:
        r["Q_t"] = r["RMSE_log"] / r0 + float(p["spread_plateau_eta"]) * r["std_alpha_eff"] / a0
    pbest = sorted(rows, key=lambda z: z["Q_t"])[0]
    return gbest, pbest, tau_trust


def f(x):
    try:
        if not np.isfinite(float(x)):
            return "nan"
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def jsonable(value):
    """Convert nested NumPy results into strict, portable JSON values."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(cfg["output_dir"])
    p = cfg["analysis"]

    sk = analyze_sk(out_dir / "SK_ensemble_ka.txt", p)
    nv_path = out_dir / "num_var_ensemble_R_over_a.txt"
    nv_enabled = bool(cfg.get("number_variance", {}).get("enabled", True))
    nv = analyze_nv(nv_path, p) if nv_enabled and nv_path.is_file() else None
    sp_global, sp_plateau, tau_trust = analyze_spread(
        out_dir
        / "routeA_pure_directdiffusion_dualfit_protocol"
        / "sample_ensemble_spreadability.txt",
        p,
        cfg["spreadability"]["raw_max_factor_over_trust"],
    )

    # Frozen default estimator: equal-weight S(k) + plateau spreadability.
    # Number variance is an independent real-space Class-like reference and
    # never contributes to the combined exponent.
    alpha_sk = float(sk["alpha_SK"])
    alpha_sp = float(sp_plateau["alpha_SP"])
    alpha_joint_equal = 0.5 * (alpha_sk + alpha_sp)
    d_method = 0.5 * abs(alpha_sk - alpha_sp)
    labels = ["alpha_SK", "alpha_SP_plateau"]
    ref = cfg.get("alpha_reference", None)
    delta = abs(alpha_joint_equal - float(ref)) if ref is not None else np.nan
    eps = delta / abs(float(ref)) if ref not in (None, 0) else np.nan

    lines = []
    lines.append("Combined alpha analysis")
    lines.append("=" * 72)
    lines.append(f"case_name: {cfg.get('case_name', '')}")
    lines.append(f"input_dir: {cfg['input_dir']}")
    lines.append(f"output_dir: {cfg['output_dir']}")
    lines.append("")
    lines.append("Final summary")
    lines.append("-" * 72)
    lines.append(f"alpha_SK: {f(sk['alpha_SK'])}")
    lines.append(f"S(k) window ka: [{f(sk['ka_min'])}, {f(sk['ka_max'])}]")
    lines.append(f"S(k) admissible right bound ka: {f(sk.get('sk_admissible_right_ka', np.nan))}")
    lines.append(f"S(k) right bound reason: {sk.get('sk_right_bound_reason', '')}")
    if nv is not None:
        lines.append(f"number_variance_role: independent real-space Class-like reference")
        lines.append(f"number_variance_class: {nv['class']}")
        lines.append(f"mean_p_eff: {f(nv['mean_p_eff'])}")
        lines.append(f"alpha_NV: {f(nv['alpha_NV'])}")
    else:
        lines.append("number_variance_role: not evaluated; never used in the joint estimator")
    lines.append(f"alpha_SP_plateau: {f(sp_plateau['alpha_SP'])}")
    lines.append(f"spreadability_plateau_tau: [{f(sp_plateau['tau_min'])}, {f(sp_plateau['tau_max'])}]")
    lines.append(f"alpha_SP_global_baseline: {f(sp_global['alpha_SP'])}")
    lines.append("alpha_SP_global_role: traditional fitting baseline only")
    lines.append(f"joint_contributors: {', '.join(labels)}")
    lines.append("joint_weight_SK: 0.500000")
    lines.append("joint_weight_SP: 0.500000")
    lines.append(f"alpha_joint_equal: {f(alpha_joint_equal)}")
    lines.append(f"d_method: {f(d_method)}")
    lines.append(f"alpha_reference: {f(ref)}")
    lines.append(f"Delta_alpha_joint: {f(delta)}")
    lines.append(f"epsilon_joint: {f(eps)}")
    lines.append(f"epsilon_joint_percent: {f(100 * eps)}")
    lines.append("")
    lines.append("S(k)")
    lines.append(str(sk))
    lines.append("")
    if nv is not None:
        lines.append("Number variance (independent reference)")
        lines.append(str(nv))
        lines.append("")
    lines.append("Spreadability")
    lines.append(str(dict(global_fit=sp_global, plateau_fit=sp_plateau, tau_trust_max=tau_trust)))

    out = out_dir / "combined_alpha_result.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    result = {
        "case_name": cfg.get("case_name", ""),
        "alpha_reference": ref,
        "alpha_SK": alpha_sk,
        "alpha_SP_plateau": alpha_sp,
        "alpha_SP_global_baseline": float(sp_global["alpha_SP"]),
        "alpha_joint_equal": alpha_joint_equal,
        "joint_weight_SK": 0.5,
        "joint_weight_SP": 0.5,
        "d_method": d_method,
        "Delta_alpha_joint": delta,
        "epsilon_joint": eps,
        "joint_contributors": labels,
        "number_variance_role": "independent real-space Class-like reference",
        "S_k": sk,
        "number_variance": nv,
        "spreadability": {
            "plateau_fit": sp_plateau,
            "traditional_global_baseline": sp_global,
            "tau_trust_max": tau_trust,
        },
    }
    json_out = out_dir / "combined_alpha_result.json"
    json_out.write_text(
        json.dumps(jsonable(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"saved {out}")
    print(f"saved {json_out}")


if __name__ == "__main__":
    main()
