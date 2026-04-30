from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.special import voigt_profile
import itertools

C_LIGHT = 299_792_458.0
GAMMA_LASER_THZ = 2e-6   # probe laser linewidth HWHM ~2 MHz


@dataclass
class AnalysisConfig:
    scope_file: str
    wavemeter_file: str
    output_dir: str

    ttl_threshold_v: float = 1.5
    bin_size: int = 10
    freq_clean_kernel: int = 21

    scan_smooth_window: int = 101
    coarse_block: int = 20
    min_ramp_points: int = 150
    rise_threshold_fraction: float = 0.30
    start_backtrack_blocks: int = 1
    end_trim_blocks: int = 1
    drop_edge_ramps: bool = True
    use_falling_ramps: bool = False

    edge_fraction: float = 0.12
    eps: float = 1e-12
    max_shift: int = 8

    display_savgol_window: int = 3
    display_savgol_poly: int = 1

    avg_baseline_poly_order: int = 2
    avg_baseline_exclude_half_width_thz: float = 8e-5

    peak_detect_window: int = 31
    peak_detect_poly: int = 2
    peak_threshold_sigma: float = 0.5
    peak_min_region_points: int = 12
    peak_ratio_window: int = 12
    peak_min_area_fraction: float = 0.3
    peak_min_area_snr: float = 0.4
    peak_min_ratio_bump_snr: float = 0.3
    peak_edge_exclusion_mhz: float = 80.0

    f_ref_thz: float = 355.802555
    expected_peak_positions_thz: Optional[Dict[str, Optional[float]]] = None

    single_fit_peak_index: Optional[int] = 0
    fit_half_width_thz: float = 0.00015
    fit_exclude_half_width_thz: float = 6e-5

    multi_fit_peak_indices: Optional[List[int]] = None
    multi_fit_margin_thz: float = 0.00008
    multi_fit_local_half_width_thz: float = 6e-5
    global_baseline_exclude_half_width_thz: float = 4e-5

    save_per_ramp_fits: bool = True
    per_ramp_min_successful_peaks: int = 2

    max_anchor_offset_mhz: float = 120.0
    local_search_window_mhz: float = 35.0
    local_search_min_snr: float = 3.0
    local_search_edge_fraction: float = 0.25

    save_plots: bool = True
    show_plots: bool = False
    debug: bool = False
    output_prefix: str = ""

    ec_temp: Optional[float] = None
    hl_temp: Optional[float] = None

    probe_distance_m: float = 0.22
    beam_speed_ref_ms: float = 480.0
    beam_speed_ref_temp_c: float = 1000.0
    transition_wavelength_m: float = 421e-9


def default_expected_peaks() -> Dict[str, Optional[float]]:
    return {
        "Dy164": 355.802555,
        "Dy163": 355.802690,
        "Dy162": 355.803030,
        "Dy161": 355.803640,
    }


def thz_to_mhz(x_thz: np.ndarray | float, f_ref_thz: float) -> np.ndarray | float:
    return (np.asarray(x_thz) - f_ref_thz) * 1e6

def expected_positions_in_range(cfg: AnalysisConfig, x_min_mhz: float, x_max_mhz: float) -> List[Tuple[str, float]]:
    expected = cfg.expected_peak_positions_thz or default_expected_peaks()
    out = []
    for lbl, x_thz in expected.items():
        if x_thz is None:
            continue
        x_mhz = float(thz_to_mhz(x_thz, cfg.f_ref_thz))
        if x_min_mhz <= x_mhz <= x_max_mhz:
            out.append((lbl, x_mhz))
    return out

def predicted_positions_from_assignment(
    isotope_assignment: Optional[Dict[str, Any]],
    cfg: AnalysisConfig,
    x_min_mhz: float,
    x_max_mhz: float,
) -> List[Tuple[str, float]]:
    expected = cfg.expected_peak_positions_thz or default_expected_peaks()
    if isotope_assignment is None or len(isotope_assignment) == 0:
        return []

    # Keep only anchors whose assignment offset is reasonably small
    good_anchor_infos = []
    for _, info in isotope_assignment.items():
        if "expected_label" not in info:
            continue
        if "offset_mhz" not in info:
            continue
        if abs(info["offset_mhz"]) <= cfg.max_anchor_offset_mhz:
            good_anchor_infos.append(info)

    # If none pass the cut, fall back to the best single anchor by smallest absolute offset
    if len(good_anchor_infos) == 0:
        best_info = min(
            isotope_assignment.values(),
            key=lambda info: abs(info.get("offset_mhz", np.inf))
        )
        good_anchor_infos = [best_info]

    assigned_isotopes = {
        info["expected_label"] for info in isotope_assignment.values()
        if "expected_label" in info
    }

    predicted = []

    for target_iso, target_thz in expected.items():
        if target_thz is None:
            continue
        if target_iso in assigned_isotopes:
            continue

        preds_thz = []

        for info in good_anchor_infos:
            anchor_iso = info["expected_label"]
            anchor_meas_thz = info["detected_x_thz"]

            anchor_exp_thz = expected.get(anchor_iso, None)
            if anchor_exp_thz is None:
                continue

            delta_thz = target_thz - anchor_exp_thz
            preds_thz.append(anchor_meas_thz + delta_thz)

        if len(preds_thz) == 0:
            continue

        pred_thz = float(np.mean(preds_thz))
        pred_mhz = float(thz_to_mhz(pred_thz, cfg.f_ref_thz))

        if x_min_mhz <= pred_mhz <= x_max_mhz:
            predicted.append((target_iso, pred_mhz))

    return predicted

def prefixed_name(cfg: AnalysisConfig, name: str) -> str:
    prefix = (cfg.output_prefix or "").strip()
    return f"{prefix}_{name}" if prefix else name


def bin_average_1d(arr: np.ndarray, bin_size: int) -> np.ndarray:
    n = len(arr) // bin_size
    if n <= 0:
        raise ValueError("bin_size is too large for array length")
    return arr[: n * bin_size].reshape(n, bin_size).mean(axis=1)


def odd_window(window: int, n: int) -> int:
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < 3:
        return 3 if n >= 3 else 1
    if window % 2 == 0:
        window -= 1
    return max(window, 1)


def load_scope_npz(path: str | Path) -> Dict[str, np.ndarray | float | int]:
    d = np.load(path)
    t = d["time"]
    probe = d["ch1"]
    ref = d["ch2"]
    ttl = d["ch3"]
    scan = d["ch4"]

    mask = np.isfinite(t) & np.isfinite(probe) & np.isfinite(ref) & np.isfinite(ttl) & np.isfinite(scan)

    return {
        "t": t[mask],
        "probe": probe[mask],
        "ref": ref[mask],
        "ttl": ttl[mask],
        "scan": scan[mask],
        "t_shot_launch_pc": float(d["t_shot_launch_pc"]),
        "t_shot_launch_perf_ns": int(d["t_shot_launch_perf_ns"]),
    }


def load_wavemeter_csv(path: str | Path) -> pd.DataFrame:
    wm = pd.read_csv(path)

        # Only filter on 'ok' if that column actually exists
    if "ok" in wm.columns:
        wm = wm[wm["ok"] == True].copy()
    else:
        wm = wm.copy()

    # Require the standard columns used by the rest of the script
    required = ["thz", "perf_ns", "pc_time"]
    missing = [c for c in required if c not in wm.columns]
    if missing:
        raise KeyError(
            f"Wavemeter CSV is missing required columns: {missing}. "
            f"Available columns: {list(wm.columns)}"
        )

    wm["thz"] = pd.to_numeric(wm["thz"], errors="coerce")
    wm = wm.dropna(subset=["thz"]).copy()

    if len(wm) < 2:
        raise RuntimeError("Not enough valid wavemeter samples")

    wm["t_rel_perf"] = (wm["perf_ns"] - wm["perf_ns"].iloc[0]) * 1e-9
    wm["t_rel_pc"] = wm["pc_time"] - wm["pc_time"].iloc[0]
    return wm


def norm_sig(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y)
    if np.sum(m) < 10:
        return y * np.nan
    yy = y.copy()
    yy[~m] = np.nan
    mu = np.nanmean(yy)
    sd = np.nanstd(yy)
    if sd <= 0:
        return yy - mu
    return (yy - mu) / sd


def _interpolate_and_clean_freq(
    t_scope_in_wm: np.ndarray,
    t_raw: np.ndarray,
    scan_raw: np.ndarray,
    wm_t: np.ndarray,
    wm_f: np.ndarray,
    freq_clean_kernel: int,
) -> np.ndarray:
    """
    Given a time array mapped into wavemeter time (t_scope_in_wm), interpolate
    the wavemeter frequency onto the scope timebase, clean outliers, and smooth.
    Extracted as a helper so it can be called for both rising and falling shifts.
    """
    freq_interp = np.full_like(t_scope_in_wm, np.nan, dtype=float)
    valid = (t_scope_in_wm >= wm_t.min()) & (t_scope_in_wm <= wm_t.max())
    freq_interp[valid] = np.interp(t_scope_in_wm[valid], wm_t, wm_f)

    f = freq_interp.copy()
    valid_sf = np.isfinite(f) & np.isfinite(scan_raw)
    f_valid = f[valid_sf]
    scan_valid = scan_raw[valid_sf]
    t_valid = t_raw[valid_sf]

    a, b = np.polyfit(scan_valid, f_valid, 1)
    f_model = a * scan_valid + b

    resid = f_valid - f_model
    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med))
    threshold = 8 * mad if mad > 0 else np.inf

    bad = np.abs(resid - med) > threshold
    bad = np.convolve(bad.astype(int), np.ones(5), mode="same") > 0

    good = ~bad
    if np.sum(good) > 10 and np.any(bad):
        f_valid[bad] = np.interp(t_valid[bad], t_valid[good], f_valid[good])

    f_clean = np.full_like(f, np.nan)
    f_clean[valid_sf] = f_valid

    kernel_len = odd_window(freq_clean_kernel, len(f_clean))
    kernel = np.ones(kernel_len)
    valid2 = np.isfinite(f_clean)
    num = np.convolve(np.where(valid2, f_clean, 0.0), kernel, mode="same")
    den = np.convolve(valid2.astype(float), kernel, mode="same")
    return num / den


def _find_best_shift(
    t_scope_in_wm_guess: np.ndarray,
    scan_n: np.ndarray,
    wm_t: np.ndarray,
    wm_f: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Search for the time shift that maximises correlation between normalised scan
    voltage and interpolated wavemeter frequency.

    Uses a two-stage coarse+fine search on downsampled scope data for speed:
    - Downsample scope to ~3000 points (correlation doesn't need full resolution)
    - Coarse search: 81 steps over ±0.4 s
    - Fine search: 81 steps over ±(coarse_step) around coarse best
    Returns (best_shift, best_score).
    """
    # Downsample for speed — correlation only needs ~1 ms precision
    n = len(t_scope_in_wm_guess)
    stride = max(1, n // 3000)
    t_ds   = t_scope_in_wm_guess[::stride]
    scan_ds = scan_n[::stride]
    mask_ds = mask[::stride] if mask is not None else None

    def _score_at(dt: float) -> float:
        t_test = t_ds + dt
        f_test = np.full_like(t_test, np.nan, dtype=float)
        m = (t_test >= wm_t.min()) & (t_test <= wm_t.max())
        f_test[m] = np.interp(t_test[m], wm_t, wm_f)
        f_n = norm_sig(f_test)
        good = np.isfinite(scan_ds) & np.isfinite(f_n)
        if mask_ds is not None:
            good = good & mask_ds
        if np.sum(good) < 50:
            return -np.inf
        return float(np.corrcoef(scan_ds[good], f_n[good])[0, 1])

    # Coarse search over ±0.4 s
    coarse_grid = np.linspace(-0.4, 0.4, 81)
    coarse_scores = np.array([_score_at(dt) for dt in coarse_grid])
    best_coarse_idx = int(np.argmax(coarse_scores))
    best_coarse = coarse_grid[best_coarse_idx]
    coarse_step = coarse_grid[1] - coarse_grid[0]

    # Fine search ±1 coarse step around the best coarse point
    fine_grid = np.linspace(
        best_coarse - coarse_step,
        best_coarse + coarse_step,
        81,
    )
    fine_scores = np.array([_score_at(dt) for dt in fine_grid])
    best_fine_idx = int(np.argmax(fine_scores))

    best_shift = float(fine_grid[best_fine_idx])
    best_score = float(fine_scores[best_fine_idx])
    return best_shift, best_score


def align_and_interpolate_frequency(scope: Dict[str, Any], wm_ok: pd.DataFrame, cfg: AnalysisConfig) -> Dict[str, np.ndarray | float]:
    t_raw = scope["t"]
    ttl_raw = scope["ttl"]
    scan_raw = scope["scan"]

    ttl_idx = np.where(ttl_raw > cfg.ttl_threshold_v)[0]
    if len(ttl_idx) == 0:
        raise RuntimeError("No TTL pulse found in scope CH3")

    i_ttl = int(ttl_idx[0])
    t_scope_ttl = float(t_raw[i_ttl])

    t_wm_ttl_approx = (
        (scope["t_shot_launch_perf_ns"] - int(wm_ok["perf_ns"].iloc[0])) * 1e-9 + t_scope_ttl
    )

    wm_t = wm_ok["t_rel_perf"].to_numpy(dtype=float)
    wm_f = wm_ok["thz"].to_numpy(dtype=float)

    t_scope0_in_wm_guess = t_wm_ttl_approx - t_scope_ttl
    t_scope_in_wm_guess = t_scope0_in_wm_guess + t_raw

    scan_n = norm_sig(scan_raw)

    # --- Global shift (rising ramps dominate, used for rising ramp processing) ---
    # Restrict correlation search to rising segments of the scan voltage
    dscan = np.gradient(scan_raw)
    rising_mask = dscan > 0

    best_shift, best_score = _find_best_shift(
        t_scope_in_wm_guess, scan_n, wm_t, wm_f, mask=rising_mask
    )

    # --- Falling-specific shift (only needed when use_falling_ramps=True) ---
    falling_mask = dscan < 0
    if cfg.use_falling_ramps:
        best_shift_falling, best_score_falling = _find_best_shift(
            t_scope_in_wm_guess, scan_n, wm_t, wm_f, mask=falling_mask
        )
    else:
        # Reuse rising shift — freq_interp_raw_falling won't be used
        best_shift_falling = best_shift
        best_score_falling = best_score

    # Build frequency arrays for both shifts
    freq_interp_raw = _interpolate_and_clean_freq(
        t_scope_in_wm_guess + best_shift,
        t_raw, scan_raw, wm_t, wm_f,
        cfg.freq_clean_kernel,
    )

    freq_interp_raw_falling = _interpolate_and_clean_freq(
        t_scope_in_wm_guess + best_shift_falling,
        t_raw, scan_raw, wm_t, wm_f,
        cfg.freq_clean_kernel,
    )

    return {
        "freq_interp_raw": freq_interp_raw,
        "freq_interp_raw_falling": freq_interp_raw_falling,
        "t_scope_ttl": t_scope_ttl,
        "t_wm_ttl_approx": t_wm_ttl_approx,
        "best_shift": best_shift,
        "best_score": best_score,
        "best_shift_falling": best_shift_falling,
        "best_score_falling": best_score_falling,
    }


def prepare_binned_scope(
    scope: Dict[str, Any],
    freq_interp_raw: np.ndarray,
    cfg: AnalysisConfig,
    freq_interp_raw_falling: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    t_raw = scope["t"]
    probe_raw = scope["probe"]
    ref_raw = scope["ref"]
    ttl_raw = scope["ttl"]
    scan_raw = scope["scan"]

    t = bin_average_1d(t_raw, cfg.bin_size)
    probe = bin_average_1d(probe_raw, cfg.bin_size)
    ref = bin_average_1d(ref_raw, cfg.bin_size)
    ttl = bin_average_1d(ttl_raw, cfg.bin_size)
    scan = bin_average_1d(scan_raw, cfg.bin_size)
    freq_interp = bin_average_1d(freq_interp_raw, cfg.bin_size)

    result = {
        "t": t,
        "probe": probe,
        "ref": ref,
        "ttl": ttl,
        "scan": scan,
        "freq_interp": freq_interp,
        "t_raw": t_raw,
        "probe_raw": probe_raw,
        "ref_raw": ref_raw,
        "ttl_raw": ttl_raw,
        "scan_raw": scan_raw,
    }

    if freq_interp_raw_falling is not None:
        result["freq_interp_falling"] = bin_average_1d(freq_interp_raw_falling, cfg.bin_size)
    else:
        result["freq_interp_falling"] = freq_interp

    return result


def detect_rising_ramps(t: np.ndarray, scan: np.ndarray, cfg: AnalysisConfig) -> Tuple[np.ndarray, List[slice]]:
    smooth_window = odd_window(cfg.scan_smooth_window, len(scan))
    scan_smooth = savgol_filter(scan, smooth_window, 3 if smooth_window >= 5 else 1)

    n_coarse = len(scan_smooth) // cfg.coarse_block
    if n_coarse < 2:
        raise RuntimeError("Not enough data for coarse ramp detection")

    scan_blocks = scan_smooth[: n_coarse * cfg.coarse_block].reshape(n_coarse, cfg.coarse_block)
    scan_coarse = scan_blocks.mean(axis=1)
    dscan_coarse = np.diff(scan_coarse, prepend=scan_coarse[0])

    positive_d = dscan_coarse[dscan_coarse > 0]
    if len(positive_d) == 0:
        raise RuntimeError("No positive scan derivative points found")

    rise_threshold = cfg.rise_threshold_fraction * np.nanmax(positive_d)
    rising_state = dscan_coarse > rise_threshold

    rising_ramps_coarse: List[Tuple[int, int]] = []
    in_run = False
    start_idx = None
    for i, state in enumerate(rising_state):
        if state and not in_run:
            in_run = True
            start_idx = i
        elif not state and in_run:
            rising_ramps_coarse.append((start_idx, i - 1))
            in_run = False
    if in_run:
        rising_ramps_coarse.append((start_idx, len(rising_state) - 1))

    ramp_slices: List[slice] = []
    for a, b in rising_ramps_coarse:
        start = a * cfg.coarse_block
        end = min((b + 1) * cfg.coarse_block, len(scan_smooth))
        start = max(0, start - cfg.start_backtrack_blocks * cfg.coarse_block)
        end = end - cfg.end_trim_blocks * cfg.coarse_block
        if (end - start) >= cfg.min_ramp_points:
            ramp_slices.append(slice(start, end))

    if cfg.drop_edge_ramps and len(ramp_slices) >= 3:
        lengths = np.array([r.stop - r.start for r in ramp_slices])
        median_len = np.median(lengths)
        keep = []
        for r in ramp_slices:
            this_len = r.stop - r.start
            k = this_len >= 0.8 * median_len
            if r.start <= cfg.coarse_block:
                k = False
            if r.stop >= len(scan_smooth) - cfg.coarse_block:
                k = False
            keep.append(k)
        ramp_slices = [r for r, k in zip(ramp_slices, keep) if k]

    return scan_smooth, ramp_slices


def detect_falling_ramps(t: np.ndarray, scan: np.ndarray, cfg: AnalysisConfig) -> List[slice]:
    """
    Detect falling (negative) ramps in the scan signal.
    Mirrors detect_rising_ramps exactly, thresholding on negative derivatives
    instead of positive ones. Returns ramp slices only (scan_smooth is already
    available from detect_rising_ramps).
    """
    smooth_window = odd_window(cfg.scan_smooth_window, len(scan))
    scan_smooth = savgol_filter(scan, smooth_window, 3 if smooth_window >= 5 else 1)

    n_coarse = len(scan_smooth) // cfg.coarse_block
    if n_coarse < 2:
        raise RuntimeError("Not enough data for coarse ramp detection")

    scan_blocks = scan_smooth[: n_coarse * cfg.coarse_block].reshape(n_coarse, cfg.coarse_block)
    scan_coarse = scan_blocks.mean(axis=1)
    dscan_coarse = np.diff(scan_coarse, prepend=scan_coarse[0])

    negative_d = dscan_coarse[dscan_coarse < 0]
    if len(negative_d) == 0:
        raise RuntimeError("No negative scan derivative points found")

    # rise_threshold_fraction applied to the most-negative value (nanmin is negative)
    fall_threshold = cfg.rise_threshold_fraction * np.nanmin(negative_d)
    falling_state = dscan_coarse < fall_threshold

    falling_ramps_coarse: List[Tuple[int, int]] = []
    in_run = False
    start_idx = None
    for i, state in enumerate(falling_state):
        if state and not in_run:
            in_run = True
            start_idx = i
        elif not state and in_run:
            falling_ramps_coarse.append((start_idx, i - 1))
            in_run = False
    if in_run:
        falling_ramps_coarse.append((start_idx, len(falling_state) - 1))

    ramp_slices: List[slice] = []
    for a, b in falling_ramps_coarse:
        start = a * cfg.coarse_block
        end = min((b + 1) * cfg.coarse_block, len(scan_smooth))
        start = max(0, start - cfg.start_backtrack_blocks * cfg.coarse_block)
        end = end - cfg.end_trim_blocks * cfg.coarse_block
        if (end - start) >= cfg.min_ramp_points:
            ramp_slices.append(slice(start, end))

    if cfg.drop_edge_ramps and len(ramp_slices) >= 3:
        lengths = np.array([r.stop - r.start for r in ramp_slices])
        median_len = np.median(lengths)
        keep = []
        for r in ramp_slices:
            this_len = r.stop - r.start
            k = this_len >= 0.8 * median_len
            if r.start <= cfg.coarse_block:
                k = False
            if r.stop >= len(scan_smooth) - cfg.coarse_block:
                k = False
            keep.append(k)
        ramp_slices = [r for r, k in zip(ramp_slices, keep) if k]

    return ramp_slices


def process_rising_ramp(
    t_r: np.ndarray,
    scan_r: np.ndarray,
    freq_r: np.ndarray,
    probe_r: np.ndarray,
    ref_r: np.ndarray,
    edge_fraction: float = 0.12,
    eps: float = 1e-12,
    max_shift: int = 8,
) -> Dict[str, Any]:
    n0 = len(scan_r)
    n_edge0 = max(10, int(edge_fraction * n0))
    edge_idx0 = np.r_[np.arange(n_edge0), np.arange(n0 - n_edge0, n0)]

    best_shift = 0
    best_score = np.inf

    for s in range(-max_shift, max_shift + 1):
        p_test = np.roll(probe_r, s)
        p_edge = p_test[edge_idx0]
        r_edge = ref_r[edge_idx0]
        k_test = np.mean(p_edge) / np.mean(r_edge)
        resid = p_edge - k_test * r_edge
        score = np.std(resid)
        if score < best_score:
            best_score = score
            best_shift = s

    probe_r = np.roll(probe_r, best_shift)

    valid = (
        np.isfinite(freq_r)
        & np.isfinite(scan_r)
        & np.isfinite(probe_r)
        & np.isfinite(ref_r)
        & np.isfinite(t_r)
    )

    tt = t_r[valid]
    x_scan = scan_r[valid]
    x_freq = freq_r[valid]
    p = probe_r[valid]
    r = ref_r[valid]

    if len(x_scan) < 20:
        raise RuntimeError("Ramp has too few valid points after frequency filtering")

    tt_mid = np.mean(tt)
    p_lin = np.polyfit(tt - tt_mid, x_freq, 1)
    x_freq_linear = p_lin[0] * (tt - tt_mid) + p_lin[1]
    freq_linear_resid_std_mhz = float(np.std(x_freq - x_freq_linear) * 1e6)

    n = len(x_scan)
    n_edge = max(20, int(edge_fraction * n))

    probe_edges = np.r_[p[:n_edge], p[-n_edge:]]
    ref_edges = np.r_[r[:n_edge], r[-n_edge:]]

    k = np.mean(probe_edges) / np.mean(ref_edges)
    ref_scaled = k * r

    ratio = ref_scaled / (p + eps)
    od = np.log(ratio)
    diff = p - ref_scaled

    f_edge = np.r_[x_freq_linear[:n_edge], x_freq_linear[-n_edge:]]
    od_edge = np.r_[od[:n_edge], od[-n_edge:]]
    ratio_edge = np.r_[ratio[:n_edge], ratio[-n_edge:]]
    diff_edge = np.r_[diff[:n_edge], diff[-n_edge:]]

    f0 = np.mean(f_edge)
    f_edge_c = f_edge - f0
    f_c = x_freq_linear - f0

    od_coeff = np.polyfit(f_edge_c, od_edge, 1)
    od_baseline = np.polyval(od_coeff, f_c)
    od_corr = od - od_baseline

    ratio_coeff = np.polyfit(f_edge_c, ratio_edge, 1)
    ratio_baseline = np.polyval(ratio_coeff, f_c)
    ratio_corr = ratio / ratio_baseline

    diff_coeff = np.polyfit(f_edge_c, diff_edge, 1)
    diff_baseline = np.polyval(diff_coeff, f_c)
    diff_corr = diff - diff_baseline

    return {
        "t": tt,
        "scan": x_scan,
        "freq": x_freq_linear,
        "probe": p,
        "ref": r,
        "k": float(k),
        "best_shift": int(best_shift),
        "shift_score": float(best_score),
        "freq_linear_resid_std_mhz": freq_linear_resid_std_mhz,
        "ref_scaled": ref_scaled,
        "ratio": ratio,
        "ratio_baseline": ratio_baseline,
        "ratio_corr": ratio_corr,
        "od": od,
        "od_baseline": od_baseline,
        "od_corr": od_corr,
        "diff": diff,
        "diff_baseline": diff_baseline,
        "diff_corr": diff_corr,
    }


def average_ramps_to_common_axis(processed: List[Dict[str, Any]], cfg: AnalysisConfig) -> Dict[str, Any]:
    x_min = max(np.nanmin(p["freq"]) for p in processed)
    x_max = min(np.nanmax(p["freq"]) for p in processed)
    common_x = np.linspace(x_min, x_max, 1200)

    ratio_corr_matrix = []
    od_corr_matrix = []
    diff_corr_matrix = []

    for p in processed:
        f = np.asarray(p["freq"], dtype=float)
        y_ratio = np.asarray(p["ratio_corr"], dtype=float)
        y_od = np.asarray(p["od_corr"], dtype=float)
        y_diff = np.asarray(p["diff_corr"], dtype=float)

        m = np.isfinite(f) & np.isfinite(y_ratio) & np.isfinite(y_od) & np.isfinite(y_diff)
        f = f[m]
        y_ratio = y_ratio[m]
        y_od = y_od[m]
        y_diff = y_diff[m]

        order = np.argsort(f)
        f = f[order]
        y_ratio = y_ratio[order]
        y_od = y_od[order]
        y_diff = y_diff[order]

        # Collapse duplicate frequency values by averaging, matching the notebook Cell 9 logic
        f_unique, inv = np.unique(f, return_inverse=True)

        ratio_unique = np.zeros_like(f_unique, dtype=float)
        od_unique = np.zeros_like(f_unique, dtype=float)
        diff_unique = np.zeros_like(f_unique, dtype=float)
        counts = np.zeros_like(f_unique, dtype=float)

        for i, j in enumerate(inv):
            ratio_unique[j] += y_ratio[i]
            od_unique[j] += y_od[i]
            diff_unique[j] += y_diff[i]
            counts[j] += 1.0

        ratio_unique /= counts
        od_unique /= counts
        diff_unique /= counts

        ratio_corr_matrix.append(np.interp(common_x, f_unique, ratio_unique))
        od_corr_matrix.append(np.interp(common_x, f_unique, od_unique))
        diff_corr_matrix.append(np.interp(common_x, f_unique, diff_unique))

    ratio_corr_matrix = np.asarray(ratio_corr_matrix)
    od_corr_matrix = np.asarray(od_corr_matrix)
    diff_corr_matrix = np.asarray(diff_corr_matrix)

    ratio_mean = np.mean(ratio_corr_matrix, axis=0)
    ratio_std = np.std(ratio_corr_matrix, axis=0)
    od_mean = np.mean(od_corr_matrix, axis=0)
    od_std = np.std(od_corr_matrix, axis=0)
    diff_mean = np.mean(diff_corr_matrix, axis=0)
    diff_std = np.std(diff_corr_matrix, axis=0)

    win = odd_window(cfg.display_savgol_window, len(common_x))
    poly = min(cfg.display_savgol_poly, win - 1)
    ratio_mean_s = savgol_filter(ratio_mean, win, poly)
    od_mean_s = savgol_filter(od_mean, win, poly)
    diff_mean_s = savgol_filter(diff_mean, win, poly)

    return {
        "common_x": common_x,
        "ratio_corr_matrix": ratio_corr_matrix,
        "od_corr_matrix": od_corr_matrix,
        "diff_corr_matrix": diff_corr_matrix,
        "ratio_mean": ratio_mean,
        "ratio_std": ratio_std,
        "od_mean": od_mean,
        "od_std": od_std,
        "diff_mean": diff_mean,
        "diff_std": diff_std,
        "ratio_mean_s": ratio_mean_s,
        "od_mean_s": od_mean_s,
        "diff_mean_s": diff_mean_s,
    }

def correct_average_baseline(
    avg: Dict[str, Any],
    peak_positions_thz: Optional[List[float]],
    cfg: AnalysisConfig,
) -> Dict[str, Any]:
    """
    Remove slow baseline curvature from the averaged ratio/OD/diff traces using a
    low-order polynomial fit, excluding regions around the detected peaks.
    """
    x = np.asarray(avg["common_x"], dtype=float)

    if peak_positions_thz is None:
        peak_positions_thz = []

    peak_positions_thz = [float(v) for v in peak_positions_thz if np.isfinite(v)]

    baseline_mask = np.isfinite(x)
    for x0 in peak_positions_thz:
        baseline_mask &= np.abs(x - x0) > cfg.avg_baseline_exclude_half_width_thz

    # Fallback to edges if too many points are excluded
    if np.sum(baseline_mask) < 20:
        n_edge = max(20, len(x) // 10)
        baseline_mask = np.zeros_like(x, dtype=bool)
        baseline_mask[:n_edge] = True
        baseline_mask[-n_edge:] = True

    def _correct_one(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float)
        good = np.isfinite(x) & np.isfinite(y) & baseline_mask

        if np.sum(good) < 10:
            baseline = np.full_like(y, np.nanmedian(y))
            y_corr = y - baseline + np.nanmedian(baseline)
            return y_corr, baseline

        x_ref = float(np.mean(x[good]))
        coeff = np.polyfit(x[good] - x_ref, y[good], cfg.avg_baseline_poly_order)
        baseline = np.polyval(coeff, x - x_ref)

        # Preserve approximate overall level
        baseline_offset = float(np.nanmedian(baseline[good]))
        y_corr = y - baseline + baseline_offset
        return y_corr, baseline

    ratio_mean_corr, ratio_baseline = _correct_one(avg["ratio_mean"])
    od_mean_corr, od_baseline       = _correct_one(avg["od_mean"])
    diff_mean_corr, diff_baseline   = _correct_one(avg["diff_mean"])

    # Re-zero the OD trace so off-resonance baseline sits at zero.
    # This corrects for probe/ref imbalance without affecting the ratio trace
    # (which peak detection and fitting use independently).
    od_off_resonance_level = float(np.nanmedian(od_mean_corr[baseline_mask]))
    od_mean_corr = od_mean_corr - od_off_resonance_level

    win = odd_window(cfg.display_savgol_window, len(x))
    poly = min(cfg.display_savgol_poly, win - 1)

    avg_out = dict(avg)
    avg_out["ratio_mean"] = ratio_mean_corr
    avg_out["od_mean"] = od_mean_corr
    avg_out["diff_mean"] = diff_mean_corr

    avg_out["ratio_mean_s"] = savgol_filter(ratio_mean_corr, win, poly)
    avg_out["od_mean_s"] = savgol_filter(od_mean_corr, win, poly)
    avg_out["diff_mean_s"] = savgol_filter(diff_mean_corr, win, poly)

    avg_out["ratio_baseline_poly"] = ratio_baseline
    avg_out["od_baseline_poly"] = od_baseline
    avg_out["diff_baseline_poly"] = diff_baseline
    avg_out["baseline_mask"] = baseline_mask

    return avg_out

def single_ramp_to_avg_like(ramp: Dict[str, Any], cfg: AnalysisConfig, common_x: np.ndarray) -> Dict[str, Any]:
    """
    Convert one processed ramp into an avg-like structure on the supplied common_x axis,
    so the existing peak detection / fit functions can be reused.
    """
    f = np.asarray(ramp["freq"], dtype=float)
    ratio = np.asarray(ramp["ratio_corr"], dtype=float)
    od = np.asarray(ramp["od_corr"], dtype=float)
    diff = np.asarray(ramp["diff_corr"], dtype=float)

    m = np.isfinite(f) & np.isfinite(ratio) & np.isfinite(od) & np.isfinite(diff)
    f = f[m]
    ratio = ratio[m]
    od = od[m]
    diff = diff[m]

    order = np.argsort(f)
    f = f[order]
    ratio = ratio[order]
    od = od[order]
    diff = diff[order]

    f_unique, inv = np.unique(f, return_inverse=True)

    ratio_unique = np.zeros_like(f_unique, dtype=float)
    od_unique = np.zeros_like(f_unique, dtype=float)
    diff_unique = np.zeros_like(f_unique, dtype=float)
    counts = np.zeros_like(f_unique, dtype=float)

    for i, j in enumerate(inv):
        ratio_unique[j] += ratio[i]
        od_unique[j] += od[i]
        diff_unique[j] += diff[i]
        counts[j] += 1.0

    ratio_unique /= counts
    od_unique /= counts
    diff_unique /= counts

    ratio_interp = np.interp(common_x, f_unique, ratio_unique)
    od_interp = np.interp(common_x, f_unique, od_unique)
    diff_interp = np.interp(common_x, f_unique, diff_unique)

    win = odd_window(cfg.display_savgol_window, len(common_x))
    poly = min(cfg.display_savgol_poly, win - 1)

    return {
        "common_x": common_x,
        "ratio_corr_matrix": np.asarray([ratio_interp]),
        "od_corr_matrix": np.asarray([od_interp]),
        "diff_corr_matrix": np.asarray([diff_interp]),
        "ratio_mean": ratio_interp,
        "ratio_std": np.zeros_like(ratio_interp),
        "od_mean": od_interp,
        "od_std": np.zeros_like(od_interp),
        "diff_mean": diff_interp,
        "diff_std": np.zeros_like(diff_interp),
        "ratio_mean_s": savgol_filter(ratio_interp, win, poly),
        "od_mean_s": savgol_filter(od_interp, win, poly),
        "diff_mean_s": savgol_filter(diff_interp, win, poly),
    }

def detect_strong_peaks(avg: Dict[str, Any], cfg: AnalysisConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    x_search = avg["common_x"]
    ratio_search = avg["ratio_mean"]
    ratio_std_search = avg["ratio_std"]
    od_search = avg["od_mean"]
    od_std_search = avg["od_std"]

    ratio_detect = savgol_filter(ratio_search, cfg.peak_detect_window, cfg.peak_detect_poly)
    od_detect = savgol_filter(od_search, cfg.peak_detect_window, cfg.peak_detect_poly)

    # Use median absolute deviation for robust baseline and noise estimation
    # Works correctly whether baseline is at zero or offset
    baseline = np.nanmedian(od_detect)
    noise = np.nanmedian(np.abs(od_detect - baseline)) * 1.4826  # MAD to sigma
    threshold = baseline + cfg.peak_threshold_sigma * noise

    above = od_detect > threshold
    regions = []
    in_region = False
    start = None
    for i, flag in enumerate(above):
        if flag and not in_region:
            start = i
            in_region = True
        elif not flag and in_region:
            end = i - 1
            if end - start + 1 >= cfg.peak_min_region_points:
                regions.append((start, end))
            in_region = False
    if in_region:
        end = len(above) - 1
        if end - start + 1 >= cfg.peak_min_region_points:
            regions.append((start, end))

    candidates = []

    x_min_mhz = float(thz_to_mhz(np.min(x_search), cfg.f_ref_thz))
    x_max_mhz = float(thz_to_mhz(np.max(x_search), cfg.f_ref_thz))

    for start, end in regions:
        region_slice = slice(start, end + 1)
        x_region = x_search[region_slice]
        od_region = od_detect[region_slice]
        od_std_region = od_std_search[region_slice]

        local_idx = int(np.argmax(od_region))
        op = start + local_idx

        x_peak_thz = float(x_search[op])
        x_peak_mhz = float(thz_to_mhz(x_peak_thz, cfg.f_ref_thz))

        # Reject candidate peaks too close to the scan edges.
        # These are often baseline / edge artefacts rather than real absorption lines.
        if (
            x_peak_mhz < x_min_mhz + cfg.peak_edge_exclusion_mhz
            or x_peak_mhz > x_max_mhz - cfg.peak_edge_exclusion_mhz
        ):
            continue

        od_excess = np.clip(od_region - threshold, 0, None)
        od_area = np.trapz(od_excess, x_region)

        mean_std = np.mean(od_std_region)
        region_width = x_region[-1] - x_region[0] if len(x_region) > 1 else 0.0
        od_area_noise = mean_std * max(region_width, 1e-12)
        area_snr = od_area / od_area_noise if od_area_noise > 0 else np.nan

        i_min = max(0, op - cfg.peak_ratio_window)
        i_max = min(len(ratio_search), op + cfg.peak_ratio_window + 1)

        local_ratio = ratio_detect[i_min:i_max]
        local_ratio_std = ratio_std_search[i_min:i_max]
        ratio_val = np.max(local_ratio)
        ratio_noise = np.mean(local_ratio_std)

        ratio_bump = max(0.0, ratio_val - 1.0)
        ratio_bump_snr = ratio_bump / ratio_noise if ratio_noise > 0 else np.nan

        candidates.append({
            "x": x_peak_thz,
            "od": float(od_detect[op]),
            "od_area": float(od_area),
            "area_snr": float(area_snr),
            "ratio": float(ratio_val),
            "ratio_bump": float(ratio_bump),
            "ratio_bump_snr": float(ratio_bump_snr),
            "region_start_x": float(x_search[start]),
            "region_end_x": float(x_search[end]),
        })

    candidates = sorted(candidates, key=lambda d: d["x"])
    if len(candidates) == 0:
        strong_peaks = []
    else:
        candidates_sorted_by_area = sorted(candidates, key=lambda d: d["od_area"], reverse=True)
        max_area = candidates_sorted_by_area[0]["od_area"]
        strong_peaks = [
            p for p in candidates_sorted_by_area
            if (p["od_area"] >= cfg.peak_min_area_fraction * max_area)
            and (p["area_snr"] >= cfg.peak_min_area_snr)
            and (p["ratio_bump_snr"] >= cfg.peak_min_ratio_bump_snr)
        ]
        strong_peaks = sorted(strong_peaks, key=lambda d: d["x"])

    for i, p in enumerate(candidates):
        p["peak_label"] = f"P{i+1}"
    for i, p in enumerate(strong_peaks):
        p["peak_label"] = f"P{i+1}"

    diagnostics = {
        "ratio_detect": ratio_detect,
        "od_detect": od_detect,
        "baseline": float(baseline),
        "noise": float(noise),
        "threshold": float(threshold),
        "regions": regions,
    }
    return candidates, strong_peaks, diagnostics

def find_local_peak_near_prediction(
    x_mhz: np.ndarray,
    y: np.ndarray,
    pred_mhz: float,
    window_mhz: float = 20.0,
    min_snr: float = 3.0,
    edge_fraction: float = 0.25,
):
    """
    Find a real local peak near a predicted position.
    Returns peak dict or None.
    """
    mask = (x_mhz > pred_mhz - window_mhz) & (x_mhz < pred_mhz + window_mhz)

    if np.sum(mask) < 10:
        return None

    x_local = x_mhz[mask]
    y_local = y[mask]

    idx = int(np.argmax(y_local))

    # Reject maxima at the boundary of the search window
    if idx == 0 or idx == len(y_local) - 1:
        return None

    n = len(y_local)
    n_edge = max(3, int(edge_fraction * n))
    y_edge = np.r_[y_local[:n_edge], y_local[-n_edge:]]

    baseline = float(np.median(y_edge))
    noise = float(np.std(y_edge))
    peak_height = float(y_local[idx] - baseline)

    if noise <= 0:
        return None

    snr = peak_height / noise

    if peak_height <= 0:
        return None
    if snr < min_snr:
        return None

    peak = {
        "x_mhz": float(x_local[idx]),
        "y": float(y_local[idx]),
        "source": "local_search",
        "predicted_mhz": pred_mhz,
        "baseline": baseline,
        "noise": noise,
        "peak_height": peak_height,
        "snr": float(snr),
    }

    return peak

def find_shoulder_peak_near_prediction(
    x_mhz: np.ndarray,
    y: np.ndarray,
    pred_mhz: float,
    anchor_mhz: float,
    window_mhz: float = 40.0,
    min_snr: float = 1.5,
):
    """
    Search for a weak peak that sits on the shoulder of a stronger neighbour (e.g. Dy163
    on the shoulder of Dy164). Unlike find_local_peak_near_prediction, this function:
      - Restricts the search window to the side of pred_mhz AWAY from the anchor peak
        to avoid the anchor's tail dominating the local maximum detection
      - Estimates baseline from the far edge of the window only (not both edges)
      - Uses a lower default min_snr since we expect a weak shoulder peak
    anchor_mhz: position of the neighbouring strong peak to avoid (e.g. Dy164)
    """
    # Only search on the far side from the anchor
    if pred_mhz > anchor_mhz:
        mask = (x_mhz > anchor_mhz + 0.4 * (pred_mhz - anchor_mhz)) &                (x_mhz < pred_mhz + window_mhz)
    else:
        mask = (x_mhz > pred_mhz - window_mhz) &                (x_mhz < anchor_mhz - 0.4 * (anchor_mhz - pred_mhz))

    if np.sum(mask) < 8:
        return None

    x_local = x_mhz[mask]
    y_local = y[mask]

    idx = int(np.argmax(y_local))

    # Reject if maximum is at the very edge closest to the anchor
    if pred_mhz > anchor_mhz and idx == 0:
        return None
    if pred_mhz < anchor_mhz and idx == len(y_local) - 1:
        return None

    # Baseline from the far edge only (away from anchor)
    n_edge = max(3, int(0.2 * len(y_local)))
    if pred_mhz > anchor_mhz:
        y_edge = y_local[-n_edge:]   # far end = high freq side
    else:
        y_edge = y_local[:n_edge]    # far end = low freq side

    baseline = float(np.median(y_edge))
    noise = float(np.std(y_edge))
    peak_height = float(y_local[idx] - baseline)

    if noise <= 0 or peak_height <= 0:
        return None

    snr = peak_height / noise
    if snr < min_snr:
        return None

    return {
        "x_mhz":        float(x_local[idx]),
        "y":            float(y_local[idx]),
        "source":       "shoulder_search",
        "predicted_mhz": pred_mhz,
        "baseline":     baseline,
        "noise":        noise,
        "peak_height":  peak_height,
        "snr":          float(snr),
    }


def voigt_with_baseline(
    x: np.ndarray,
    A: float,
    x0: float,
    sigma: float,
    gamma: float,
    c0: float,
    c1: float,
) -> np.ndarray:
    """
    Voigt peak with linear baseline.
    sigma and gamma are in THz.
    c0 is the baseline offset.
    c1 is the baseline slope in ratio units per THz.
    """
    return c0 + c1 * (x - x0) + A * voigt_profile(x - x0, sigma, gamma)

def voigt_fwhm_from_sigma_gamma(sigma: float, gamma: float) -> float:
    """
    Approximate Voigt FWHM in THz from Gaussian sigma and Lorentzian gamma.
    """
    return 0.5346 * (2 * gamma) + np.sqrt(
        0.2166 * (2 * gamma) ** 2 + (2.35482 * sigma) ** 2
    )

def estimate_linear_baseline_excluding_peaks(
    x: np.ndarray,
    y: np.ndarray,
    peak_centres: np.ndarray,
    exclude_half_width_thz: float,
) -> Dict[str, Any]:
    """
    Estimate a linear baseline across the selected fit window while excluding
    regions around the selected peaks.

    Returns a dict containing:
        baseline_fit: baseline evaluated across x
        c0: intercept at x_ref
        c1: slope
        x_ref: reference x used for numerical stability
        baseline_mask: boolean mask of baseline points used
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    peak_centres = np.asarray(peak_centres, dtype=float)

    baseline_mask = np.ones_like(x, dtype=bool)
    for x0 in peak_centres:
        baseline_mask &= np.abs(x - x0) > exclude_half_width_thz

    # Fallback if too many points were excluded
    if np.sum(baseline_mask) < 10:
        n_edge = max(5, len(x) // 8)
        baseline_mask = np.zeros_like(x, dtype=bool)
        baseline_mask[:n_edge] = True
        baseline_mask[-n_edge:] = True

    x_ref = float(np.mean(x))
    coeff = np.polyfit(x[baseline_mask] - x_ref, y[baseline_mask], 1)
    c1 = float(coeff[0])
    c0 = float(coeff[1])

    baseline_fit = c0 + c1 * (x - x_ref)

    return {
        "baseline_fit": baseline_fit,
        "c0": c0,
        "c1": c1,
        "x_ref": x_ref,
        "baseline_mask": baseline_mask,
    }

def global_multi_voigt_with_baseline(
    x: np.ndarray,
    c0: float,
    c1: float,
    *params: float,
) -> np.ndarray:
    """
    Global multi-peak Voigt model with a linear residual baseline,
    intended to be used on a pre-flattened selected fit window.

    Parameter order:
        c0,   (constant baseline offset)
        c1,   (baseline slope, in ratio units per THz, centred on x_ref)
        A1, x01, sigma1, gamma1,
        A2, x02, sigma2, gamma2,
        ...
    """
    x_ref = float(np.mean(x))
    y = c0 + c1 * (x - x_ref)

    if len(params) % 4 != 0:
        raise ValueError("params must be groups of 4: A, x0, sigma, gamma")

    for i in range(0, len(params), 4):
        A = params[i]
        x0 = params[i + 1]
        sigma = params[i + 2]
        gamma = params[i + 3]

        y += A * voigt_profile(x - x0, sigma, gamma)

    return y

def global_multi_voigt_with_fixed_widths(
    x: np.ndarray,
    fixed_sigmas: np.ndarray,
    fixed_gammas: np.ndarray,
    c0: float,
    c1: float,
    *params: float,
) -> np.ndarray:
    """
    Global multi-peak Voigt model with linear residual baseline and
    fixed sigma/gamma for each peak.

    Parameter order:
        c0,   (constant baseline)
        c1,   (baseline slope, centred on x_ref)
        A1, x01,
        A2, x02,
        ...
    """
    x_ref = float(np.mean(x))
    y = c0 + c1 * (x - x_ref)

    if len(params) % 2 != 0:
        raise ValueError("params must be groups of 2: A, x0")

    n_peaks = len(params) // 2
    if len(fixed_sigmas) != n_peaks or len(fixed_gammas) != n_peaks:
        raise ValueError("fixed_sigmas and fixed_gammas must match number of peaks")

    for i in range(n_peaks):
        A = params[2 * i]
        x0 = params[2 * i + 1]
        sigma = fixed_sigmas[i]
        gamma = fixed_gammas[i]

        y += A * voigt_profile(x - x0, sigma, gamma)

    return y

def fit_single_peak(avg: Dict[str, Any], strong_peaks: List[Dict[str, Any]], cfg: AnalysisConfig) -> Optional[Dict[str, Any]]:
    if len(strong_peaks) == 0 or cfg.single_fit_peak_index is False:
        return None

    if cfg.single_fit_peak_index is None:
        peak0 = max(strong_peaks, key=lambda p: p["ratio_bump_snr"])
    else:
        idx = cfg.single_fit_peak_index
        if idx < 0 or idx >= len(strong_peaks):
            raise IndexError(f"single_fit_peak_index must be between 0 and {len(strong_peaks)-1}")
        peak0 = strong_peaks[idx]

    x0_guess = peak0["x"]
    peak_label = peak0["peak_label"]

    x = avg["common_x"]
    y = avg["ratio_mean"]
    mask_fit = (x >= x0_guess - cfg.fit_half_width_thz) & (x <= x0_guess + cfg.fit_half_width_thz)
    x_fit = x[mask_fit]
    y_fit = y[mask_fit]
    if len(x_fit) < 20:
        raise RuntimeError("Single fit window is too small")

    centre_mask = np.abs(x_fit - x0_guess) > cfg.fit_exclude_half_width_thz
    x_edge = x_fit[centre_mask]
    y_edge = y_fit[centre_mask]
    c0_guess = float(np.mean(y_edge))

    A_guess = float(max(y_fit) - c0_guess)

    # Initial guesses in THz
    sigma_guess = 6e-6
    gamma_guess = GAMMA_LASER_THZ
    c1_guess = 0.0

    # Bounds in THz
    lower_bounds = [
        0.0,                 # A
        x0_guess - 3e-5,     # x0
        1e-6,                # sigma
        GAMMA_LASER_THZ * 0.5,   # gamma lower bound
        c0_guess - 0.01,     # c0
        -100.0,              # c1
    ]
    upper_bounds = [
        0.05,                # A
        x0_guess + 5e-5,     # x0 — ±50 MHz to match global fit
        5e-5,                # sigma — raised to match global fit upper bound
        GAMMA_LASER_THZ * 3.0,   # gamma upper bound
        c0_guess + 0.01,     # c0
        100.0,               # c1
    ]

    A_guess = np.clip(A_guess, lower_bounds[0] + 1e-9, upper_bounds[0] - 1e-9)
    sigma_guess = np.clip(sigma_guess, lower_bounds[2] + 1e-9, upper_bounds[2] - 1e-9)
    gamma_guess = np.clip(gamma_guess, lower_bounds[3] + 1e-9, upper_bounds[3] - 1e-9)
    c0_guess = np.clip(c0_guess, lower_bounds[4] + 1e-9, upper_bounds[4] - 1e-9)
    c1_guess = np.clip(c1_guess, lower_bounds[5] + 1e-9, upper_bounds[5] - 1e-9)

    p0 = [A_guess, x0_guess, sigma_guess, gamma_guess, c0_guess, c1_guess]

    popt, pcov = curve_fit(
        voigt_with_baseline,
        x_fit,
        y_fit,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=50000,
    )

    A_fit, x0_fit, sigma_fit, gamma_fit, c0_fit, c1_fit = popt
    y_model = voigt_with_baseline(x_fit, *popt)
    resid = y_fit - y_model

    perr = np.sqrt(np.diag(pcov))
    A_err, x0_err, sigma_err, gamma_err, c0_err, c1_err = perr

    # Approximate Voigt FWHM in THz
    fwhm_voigt_thz = 0.5346 * (2 * gamma_fit) + np.sqrt(
        0.2166 * (2 * gamma_fit) ** 2 + (2.35482 * sigma_fit) ** 2
    )

    resid_rms = float(np.sqrt(np.mean(resid**2)))

    expected_positions = cfg.expected_peak_positions_thz or default_expected_peaks()
    assignment = match_detected_to_expected_single(x0_fit, expected_positions)

    return {
        "peak_label": peak_label,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "baseline_fit": c0_fit + c1_fit * (x_fit - x0_fit),
        "y_model": y_model,
        "resid": resid,
        "centre_thz": float(x0_fit),
        "centre_err_thz": float(x0_err),
        "amplitude": float(A_fit),
        "amplitude_err": float(A_err),
        "sigma_thz": float(sigma_fit),
        "sigma_err_thz": float(sigma_err),
        "gamma_thz": float(gamma_fit),
        "gamma_err_thz": float(gamma_err),
        "baseline_c0": float(c0_fit),
        "baseline_c0_err": float(c0_err),
        "baseline_c1": float(c1_fit),
        "baseline_c1_err": float(c1_err),
        "fwhm_mhz": float(fwhm_voigt_thz * 1e6),
        "fwhm_err_mhz": np.nan,
        "resid_rms": resid_rms,
        "isotope_assignment": assignment,
    }


def match_detected_to_expected_single(detected_x_thz: float, expected_positions_thz: Dict[str, Optional[float]]) -> Dict[str, Any]:
    expected_defined = {k: v for k, v in expected_positions_thz.items() if v is not None}
    if not expected_defined:
        return {"expected_label": None, "offset_mhz": None}
    best_label = None
    best_offset = None
    for lbl, xexp in expected_defined.items():
        off = (detected_x_thz - xexp) * 1e6
        if best_offset is None or abs(off) < abs(best_offset):
            best_offset = off
            best_label = lbl
    return {"expected_label": best_label, "offset_mhz": float(best_offset)}

def velocity_sigma_from_sigma_thz(sigma_thz: float, centre_thz: float) -> float:
    """
    Convert the Gaussian sigma of the Voigt fit to transverse velocity spread in m/s.

    Following the method of both reference theses (Uierlings 2021, Schindler 2011),
    the transverse velocity is derived from the Gaussian FWHM of the absorption peak:

        v_trans = FWHM_G * c / nu0

    where FWHM_G = 2*sqrt(2*ln2) * sigma = 2.3548 * sigma.

    This gives the characteristic transverse velocity (most-probable-speed equivalent
    in the collimated beam) consistent with the design documents.

    Note: the result is labelled velocity_sigma_ms in the output for historical reasons,
    but physically represents the transverse velocity corresponding to FWHM_G.
    """
    sigma_thz = float(sigma_thz)
    centre_thz = float(centre_thz)

    if not np.isfinite(sigma_thz) or not np.isfinite(centre_thz) or centre_thz <= 0:
        return np.nan

    SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))  # 2.3548
    fwhm_g_thz = SIGMA_TO_FWHM * sigma_thz
    return float(C_LIGHT * fwhm_g_thz / centre_thz)

def beam_speed_from_ec_temp(ec_temp_c: Optional[float], cfg: AnalysisConfig) -> float:
    """
    Estimate longitudinal beam speed from EC temperature using the scaling:
        v ~ sqrt(T)

    Uses cfg.beam_speed_ref_ms at cfg.beam_speed_ref_temp_c as the reference point.
    Temperatures are converted to Kelvin internally.
    """
    if ec_temp_c is None:
        return float(cfg.beam_speed_ref_ms)

    t_ref_k = float(cfg.beam_speed_ref_temp_c) + 273.15
    t_k = float(ec_temp_c) + 273.15

    if t_ref_k <= 0 or t_k <= 0:
        return float(cfg.beam_speed_ref_ms)

    return float(cfg.beam_speed_ref_ms * np.sqrt(t_k / t_ref_k))


def spatial_sigma_from_velocity_sigma(
    velocity_sigma_ms: float,
    beam_speed_ms: float,
    probe_distance_m: float,
) -> float:
    """
    Convert transverse velocity sigma to spatial sigma at the probe position:
        sigma_x = (sigma_v / v_z) * z
    """
    if not np.isfinite(velocity_sigma_ms) or not np.isfinite(beam_speed_ms):
        return np.nan
    if beam_speed_ms <= 0 or probe_distance_m <= 0:
        return np.nan

    return float((velocity_sigma_ms / beam_speed_ms) * probe_distance_m)


def gaussian_effective_path_length_from_sigma(spatial_sigma_m: float) -> float:
    """
    Effective path length through a transverse Gaussian beam profile at line centre:

        L_eff = sqrt(2*pi) * sigma_x

    This is the correct line-of-sight integral length for a Gaussian density
    distribution when using the peak optical density.
    """
    if not np.isfinite(spatial_sigma_m) or spatial_sigma_m <= 0:
        return np.nan

    return float(np.sqrt(2.0 * np.pi) * spatial_sigma_m)


def resonant_cross_section_m2(wavelength_m: float) -> float:
    """
    Resonant low-intensity cross section:
        sigma_0 = 3 * lambda^2 / (2*pi)
    """
    wavelength_m = float(wavelength_m)
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        return np.nan

    return float(3.0 * wavelength_m**2 / (2.0 * np.pi))


def peak_od_from_avg(avg: Dict[str, Any], centre_thz: float) -> float:
    """
    Extract the OD at the fitted line centre from the averaged smoothed OD trace.
    """
    x = np.asarray(avg["common_x"], dtype=float)
    y = np.asarray(avg["od_mean_s"], dtype=float)

    if len(x) == 0 or len(y) == 0:
        return np.nan

    if centre_thz < np.min(x) or centre_thz > np.max(x):
        return np.nan

    return float(np.interp(centre_thz, x, y))

def build_physics_results(
    avg: Dict[str, Any],
    multi_fit: Optional[Dict[str, Any]],
    cfg: AnalysisConfig,
) -> Optional[pd.DataFrame]:
    """
    Build a derived-physics table from the averaged global Voigt fit results.

    We use the Gaussian sigma of the Voigt profile, sigma_thz, to compute the
    transverse velocity spread:
        sigma_v = c * sigma_thz / centre_thz
    """
    if multi_fit is None or not multi_fit.get("fit_results"):
        return None

    rows = []

    for r in multi_fit["fit_results"]:
        assignment = r.get("isotope_assignment")
        isotope_label = None if assignment is None else assignment.get("expected_label")

        # Correct sigma for frequency jitter contribution (per-ramp jitter broadens the
        # averaged Gaussian sigma). Subtract jitter in quadrature if available.
        sigma_thz_raw = r["sigma_thz"]
        jitter_sigma_thz = 0.0
        if hasattr(cfg, "_per_ramp_jitter_sigma_thz") and cfg._per_ramp_jitter_sigma_thz is not None:
            jitter_sigma_thz = float(cfg._per_ramp_jitter_sigma_thz)

        sigma_thz_corrected = sigma_thz_raw
        if jitter_sigma_thz > 0 and sigma_thz_raw > jitter_sigma_thz:
            sigma_thz_corrected = float(np.sqrt(sigma_thz_raw**2 - jitter_sigma_thz**2))

        velocity_sigma_ms = velocity_sigma_from_sigma_thz(
            sigma_thz=sigma_thz_corrected,
            centre_thz=r["centre_thz"],
        )

        beam_speed_ms = beam_speed_from_ec_temp(cfg.ec_temp, cfg)

        spatial_sigma_m = spatial_sigma_from_velocity_sigma(
            velocity_sigma_ms=velocity_sigma_ms,
            beam_speed_ms=beam_speed_ms,
            probe_distance_m=cfg.probe_distance_m,
        )

        l_eff_m = gaussian_effective_path_length_from_sigma(spatial_sigma_m)

        sigma0_m2 = resonant_cross_section_m2(cfg.transition_wavelength_m)

        peak_od = peak_od_from_avg(
            avg=avg,
            centre_thz=r["centre_thz"],
        )

        if np.isfinite(peak_od) and np.isfinite(sigma0_m2) and np.isfinite(l_eff_m) and l_eff_m > 0:
            number_density_m3 = peak_od / (sigma0_m2 * l_eff_m)
        else:
            number_density_m3 = np.nan

        beam_diameter_mm = np.nan
        if np.isfinite(spatial_sigma_m):
            beam_diameter_mm = 4.0 * spatial_sigma_m * 1e3

        rows.append({
            "ec_temp": cfg.ec_temp,
            "hl_temp": cfg.hl_temp,
            "peak_label": r["peak_label"],
            "assigned_isotope": isotope_label,
            "centre_thz": r["centre_thz"],
            "offset_mhz": None if assignment is None else assignment.get("offset_mhz"),
            "sigma_thz_raw": r["sigma_thz"],
            "sigma_thz_corrected": sigma_thz_corrected,
            "sigma_mhz_raw": float(r["sigma_thz"]) * 1e6,
            "sigma_mhz_corrected": float(sigma_thz_corrected) * 1e6,
            "jitter_sigma_thz": jitter_sigma_thz,
            "velocity_sigma_ms": velocity_sigma_ms,
            "beam_speed_ms": beam_speed_ms,
            "spatial_sigma_m": spatial_sigma_m,
            "spatial_sigma_mm": spatial_sigma_m * 1e3 if np.isfinite(spatial_sigma_m) else np.nan,
            "beam_diameter_mm": beam_diameter_mm,
            "l_eff_m": l_eff_m,
            "l_eff_mm": l_eff_m * 1e3 if np.isfinite(l_eff_m) else np.nan,
            "sigma0_m2": sigma0_m2,
            "peak_od": peak_od,
            "number_density_m3": number_density_m3,
            "voigt_fwhm_mhz": r["fwhm_mhz"],
            "gamma_thz": r["gamma_thz"],
        })

    return pd.DataFrame(rows)


def save_physics_csv(physics_df: Optional[pd.DataFrame], cfg: AnalysisConfig, outdir: Path) -> Optional[Path]:
    if physics_df is None or physics_df.empty:
        return None

    path = outdir / prefixed_name(cfg, "physics_results.csv")
    physics_df.to_csv(path, index=False)
    return path

def fit_selected_peaks_global(avg: Dict[str, Any], strong_peaks: List[Dict[str, Any]], cfg: AnalysisConfig) -> Optional[Dict[str, Any]]:
    if len(strong_peaks) == 0:
        return None

    peak_indices = cfg.multi_fit_peak_indices
    if not peak_indices:
        peak_indices = list(range(len(strong_peaks)))

    selected_peaks = [strong_peaks[i] for i in peak_indices]
    selected_peaks = sorted(selected_peaks, key=lambda p: p["x"])
    x0_guesses = np.array([p["x"] for p in selected_peaks], dtype=float)
    peak_labels = [p["peak_label"] for p in selected_peaks]

    x = avg["common_x"]
    y = avg["ratio_mean"]

    x_min_fit = np.min(x0_guesses) - cfg.multi_fit_margin_thz
    x_max_fit = np.max(x0_guesses) + cfg.multi_fit_margin_thz
    mask_fit = (x >= x_min_fit) & (x <= x_max_fit)
    x_fit = x[mask_fit]
    y_fit = y[mask_fit]

    if len(x_fit) < 50:
        raise RuntimeError("Global multi-peak fit window is too small")

    # Estimate baseline across the selected fit window while excluding the peaks
    baseline_info = estimate_linear_baseline_excluding_peaks(
        x=x_fit,
        y=y_fit,
        peak_centres=x0_guesses,
        exclude_half_width_thz=cfg.global_baseline_exclude_half_width_thz,
    )

    baseline_prefit = baseline_info["baseline_fit"]

    # Flatten the selected window so the final global fit does not have to fight the slope
    y_fit_flat = y_fit - baseline_prefit + 1.0

    # Residual baseline after pre-flattening — allow a small slope to absorb
    # any curvature the linear pre-fit didn't fully remove
    c0_guess = 1.0
    c1_guess = 0.0

    x_ref = float(np.mean(x_fit))

    p0 = [c0_guess, c1_guess]
    lower_bounds = [0.98, -50.0]
    upper_bounds = [1.02,  50.0]

    for x0 in x0_guesses:
        # Use the local maximum near x0 as amplitude guess, not just the nearest point
        local_mask = np.abs(x_fit - x0) < 5e-5  # within 50 MHz
        if np.sum(local_mask) > 0:
            A0 = max(1e-6, float(np.max(y_fit_flat[local_mask]) - c0_guess))
        else:
            idx0 = int(np.argmin(np.abs(x_fit - x0)))
            A0 = max(1e-6, float(y_fit_flat[idx0] - c0_guess))

        # Probe laser linewidth dominates the Lorentzian component (HWHM ~2 MHz).
        # Constrain gamma to laser linewidth rather than natural linewidth to avoid
        # sigma/gamma degeneracy and ensure correct Doppler (sigma) extraction.
        gamma0 = GAMMA_LASER_THZ      # probe laser linewidth HWHM ~2 MHz (constrained)
        sigma0 = 1.2e-5               # start at expected Doppler sigma ~12 MHz

        p0.extend([A0, x0, sigma0, gamma0])

        lower_bounds.extend([
            0.0,                      # amplitude
            x0 - 5e-5,                # centre — ±50 MHz
            5e-6,                     # sigma — min ~5 MHz (physically motivated)
            GAMMA_LASER_THZ * 0.5,    # gamma lower bound
        ])

        upper_bounds.extend([
            0.05,
            x0 + 5e-5,
            5e-5,                     # sigma — max ~50 MHz
            GAMMA_LASER_THZ * 3.0,    # gamma upper bound
        ])

    popt, pcov = curve_fit(
        global_multi_voigt_with_baseline,
        x_fit,
        y_fit_flat,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=100000,
    )

    y_model_flat = global_multi_voigt_with_baseline(x_fit, *popt)

    # Convert the model back to the original ratio scale for plotting and residuals
    c0_fit = popt[0]
    c1_fit = popt[1]
    baseline_fit = baseline_prefit + (c0_fit - 1.0) + c1_fit * (x_fit - x_ref)
    y_model = baseline_fit + (y_model_flat - c0_fit - c1_fit * (x_fit - x_ref))

    resid = y_fit - y_model
    resid_rms = float(np.sqrt(np.mean(resid**2)))

    perr = np.sqrt(np.diag(pcov))

    c0_fit = popt[0]
    c1_fit = popt[1]
    c0_err = perr[0]
    c1_err = perr[1]

    expected_positions = cfg.expected_peak_positions_thz or default_expected_peaks()

    fit_results = []
    for i, lbl in enumerate(peak_labels):
        base = 2 + 4 * i  # now offset by 2 (c0 and c1) instead of 1

        A_fit = popt[base]
        x0_fit = popt[base + 1]
        sigma_fit = popt[base + 2]
        gamma_fit = popt[base + 3]

        A_err = perr[base]
        x0_err = perr[base + 1]
        sigma_err = perr[base + 2]
        gamma_err = perr[base + 3]

        fwhm_voigt_thz = voigt_fwhm_from_sigma_gamma(sigma_fit, gamma_fit)

        assignment = match_detected_to_expected_single(float(x0_fit), expected_positions)

        fit_results.append({
            "fit_type": "global_independent_width",
            "peak_label": lbl,
            "centre_thz": float(x0_fit),
            "centre_err_thz": float(x0_err),
            "amplitude": float(A_fit),
            "amplitude_err": float(A_err),
            "sigma_thz": float(sigma_fit),
            "sigma_err_thz": float(sigma_err),
            "gamma_thz": float(gamma_fit),
            "gamma_err_thz": float(gamma_err),
            "baseline_c0": float(c0_fit),
            "baseline_c0_err": float(c0_err),
            "baseline_c1": float(c1_fit),
            "baseline_c1_err": float(c1_err),
            "fwhm_mhz": float(fwhm_voigt_thz * 1e6),
            "fwhm_err_mhz": np.nan,
            "isotope_assignment": assignment,
        })

    return {
        "fit_type": "global_independent_width_prefit_baseline",
        "x_fit": x_fit,
        "y_fit": y_fit,
        "y_fit_flat": y_fit_flat,
        "baseline_prefit": baseline_prefit,
        "baseline_fit": baseline_fit,
        "y_model_flat": y_model_flat,
        "y_model": y_model,
        "resid": resid,
        "resid_rms": resid_rms,
        "fit_results": fit_results,
    }

def fit_selected_peaks_global_with_fixed_labels(
    avg: Dict[str, Any],
    selected_peaks: List[Dict[str, Any]],
    cfg: AnalysisConfig,
) -> Optional[Dict[str, Any]]:
    """
    Per-ramp global fit using fixed peak labels and fixed widths taken from the
    averaged global fit. Only amplitude, centre, and residual baseline level are fitted.
    """
    if len(selected_peaks) == 0:
        return None

    selected_peaks = sorted(selected_peaks, key=lambda p: p["x"])
    x0_guesses = np.array([p["x"] for p in selected_peaks], dtype=float)
    peak_labels = [p["peak_label"] for p in selected_peaks]

    # These widths must come from the averaged global fit results
    fixed_sigmas = np.array([p["sigma_thz"] for p in selected_peaks], dtype=float)
    fixed_gammas = np.array([p["gamma_thz"] for p in selected_peaks], dtype=float)

    x = avg["common_x"]
    y = avg["ratio_mean"]

    x_min_fit = np.min(x0_guesses) - cfg.multi_fit_margin_thz
    x_max_fit = np.max(x0_guesses) + cfg.multi_fit_margin_thz
    mask_fit = (x >= x_min_fit) & (x <= x_max_fit)
    x_fit = x[mask_fit]
    y_fit = y[mask_fit]

    if len(x_fit) < 50:
        return None

    baseline_info = estimate_linear_baseline_excluding_peaks(
        x=x_fit,
        y=y_fit,
        peak_centres=x0_guesses,
        exclude_half_width_thz=cfg.global_baseline_exclude_half_width_thz,
    )

    baseline_prefit = baseline_info["baseline_fit"]
    y_fit_flat = y_fit - baseline_prefit + 1.0

    c0_guess = 1.0
    c1_guess = 0.0
    x_ref = float(np.mean(x_fit))

    p0 = [c0_guess, c1_guess]
    lower_bounds = [0.98, -50.0]
    upper_bounds = [1.02,  50.0]

    for x0 in x0_guesses:
        idx0 = int(np.argmin(np.abs(x_fit - x0)))
        A0 = max(1e-6, float(y_fit_flat[idx0] - c0_guess))

        p0.extend([A0, x0])
        lower_bounds.extend([0.0, x0 - 5e-5])
        upper_bounds.extend([0.05, x0 + 5e-5])

    try:
        popt, pcov = curve_fit(
            lambda x_in, *params: global_multi_voigt_with_fixed_widths(
                x_in, fixed_sigmas, fixed_gammas, *params
            ),
            x_fit,
            y_fit_flat,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=100000,
        )
    except Exception:
        return None

    y_model_flat = global_multi_voigt_with_fixed_widths(x_fit, fixed_sigmas, fixed_gammas, *popt)

    c0_fit = popt[0]
    c1_fit = popt[1]
    baseline_fit = baseline_prefit + (c0_fit - 1.0) + c1_fit * (x_fit - x_ref)
    y_model = baseline_fit + (y_model_flat - c0_fit - c1_fit * (x_fit - x_ref))

    resid = y_fit - y_model
    resid_rms = float(np.sqrt(np.mean(resid**2)))

    perr = np.sqrt(np.diag(pcov))
    c0_err = perr[0]
    c1_err = perr[1]

    expected_positions = cfg.expected_peak_positions_thz or default_expected_peaks()

    fit_results = []
    for i, lbl in enumerate(peak_labels):
        base = 2 + 2 * i  # now offset by 2 (c0 and c1)

        A_fit = popt[base]
        x0_fit = popt[base + 1]

        A_err = perr[base]
        x0_err = perr[base + 1]

        sigma_fit = fixed_sigmas[i]
        gamma_fit = fixed_gammas[i]
        fwhm_voigt_thz = voigt_fwhm_from_sigma_gamma(sigma_fit, gamma_fit)

        assignment = match_detected_to_expected_single(float(x0_fit), expected_positions)

        fit_results.append({
            "fit_type": "global_fixed_width_per_ramp",
            "peak_label": lbl,
            "centre_thz": float(x0_fit),
            "centre_err_thz": float(x0_err),
            "centre_mhz": float(thz_to_mhz(x0_fit, cfg.f_ref_thz)),
            "amplitude": float(A_fit),
            "amplitude_err": float(A_err),
            "sigma_thz": float(sigma_fit),
            "sigma_err_thz": 0.0,
            "gamma_thz": float(gamma_fit),
            "gamma_err_thz": 0.0,
            "baseline_c0": float(c0_fit),
            "baseline_c0_err": float(c0_err),
            "baseline_c1": 0.0,
            "baseline_c1_err": 0.0,
            "fwhm_mhz": float(fwhm_voigt_thz * 1e6),
            "fwhm_err_mhz": 0.0,
            "isotope_assignment": assignment,
        })

    return {
        "fit_type": "global_fixed_width_per_ramp",
        "x_fit": x_fit,
        "y_fit": y_fit,
        "y_fit_flat": y_fit_flat,
        "baseline_prefit": baseline_prefit,
        "baseline_fit": baseline_fit,
        "y_model_flat": y_model_flat,
        "y_model": y_model,
        "resid": resid,
        "resid_rms": resid_rms,
        "fit_results": fit_results,
    }

def analyze_per_ramp_fits(
    processed: List[Dict[str, Any]],
    selected_peaks: List[Dict[str, Any]],
    cfg: AnalysisConfig,
    common_x: np.ndarray,
) -> Dict[str, Any]:
    """
    Fit each processed rising ramp individually using the same selected peak labels
    from the averaged analysis. Returns per-ramp fit rows and centre-jitter summary.
    """
    per_ramp_rows = []

    for ramp_idx, ramp in enumerate(processed):
        avg_like = single_ramp_to_avg_like(ramp, cfg, common_x)
        fit = fit_selected_peaks_global_with_fixed_labels(avg_like, selected_peaks, cfg)

        if fit is None:
            continue

        if len(fit["fit_results"]) < cfg.per_ramp_min_successful_peaks:
            continue

        for r in fit["fit_results"]:
            assignment = r.get("isotope_assignment")
            per_ramp_rows.append({
                "ramp_index": ramp_idx,
                "peak_label": r["peak_label"],
                "assigned_isotope": None if assignment is None else assignment["expected_label"],
                "centre_thz": r["centre_thz"],
                "centre_mhz": r["centre_mhz"],
                "offset_mhz": None if assignment is None else assignment["offset_mhz"],
                "amplitude": r["amplitude"],
                "resid_rms": fit["resid_rms"],
            })

    per_ramp_df = pd.DataFrame(per_ramp_rows)

    summary_rows = []
    if not per_ramp_df.empty:
        for peak_label, grp in per_ramp_df.groupby("peak_label"):
            assigned_isotope = grp["assigned_isotope"].iloc[0]

            centre_std_mhz = float(np.std(grp["centre_mhz"], ddof=1)) if len(grp) > 1 else np.nan

            offset_mean_mhz = float(np.mean(grp["offset_mhz"])) if grp["offset_mhz"].notna().any() else np.nan
            offset_std_mhz = float(np.std(grp["offset_mhz"], ddof=1)) if grp["offset_mhz"].notna().sum() > 1 else np.nan

            summary_rows.append({
                "peak_label": peak_label,
                "assigned_isotope": assigned_isotope,
                "n_ramps": int(len(grp)),
                "offset_mean_mhz": offset_mean_mhz,
                "offset_std_mhz": offset_std_mhz,
                "centre_std_mhz": centre_std_mhz,
            })

    per_ramp_summary_df = pd.DataFrame(summary_rows)

    return {
        "per_ramp_df": per_ramp_df,
        "per_ramp_summary_df": per_ramp_summary_df,
    }

def match_isotopes(strong_peaks: List[Dict[str, Any]], cfg: AnalysisConfig) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if len(strong_peaks) < 1:
        return None, None

    expected_positions = cfg.expected_peak_positions_thz or default_expected_peaks()
    expected_defined = {label: x for label, x in expected_positions.items() if x is not None}
    expected_all = [{"label": lbl, "x_thz": x} for lbl, x in expected_defined.items()]
    expected_all = sorted(expected_all, key=lambda d: d["x_thz"])

    detected_all = [{
        "label": p["peak_label"],
        "x_thz": p["x"],
        "x_mhz": float(thz_to_mhz(p["x"], cfg.f_ref_thz)),
        "area_snr": p["area_snr"],
        "ratio_bump_snr": p["ratio_bump_snr"],
    } for p in strong_peaks]
    detected_all = sorted(detected_all, key=lambda d: d["x_thz"])

    n_detected_use = len(detected_all)
    if n_detected_use > len(expected_all):
        return None, None

    results = []
    for detected_combo in itertools.combinations(detected_all, n_detected_use):
        detected_combo = sorted(detected_combo, key=lambda d: d["x_thz"])
        for expected_combo in itertools.combinations(expected_all, n_detected_use):
            expected_combo = sorted(expected_combo, key=lambda d: d["x_thz"])

            det_anchor = detected_combo[0]["x_thz"]
            exp_anchor = expected_combo[0]["x_thz"]

            det_rel_mhz = np.array([(d["x_thz"] - det_anchor) * 1e6 for d in detected_combo])
            exp_rel_mhz = np.array([(e["x_thz"] - exp_anchor) * 1e6 for e in expected_combo])

            spacing_residuals_mhz = det_rel_mhz - exp_rel_mhz
            rms_mhz = float(np.sqrt(np.mean(spacing_residuals_mhz**2)))
            snr_score = float(np.mean([d["area_snr"] + d["ratio_bump_snr"] for d in detected_combo]))

            assignment = {}
            for det, exp in zip(detected_combo, expected_combo):
                assignment[det["label"]] = {
                    "expected_label": exp["label"],
                    "detected_x_thz": det["x_thz"],
                    "expected_x_thz": exp["x_thz"],
                    "detected_x_mhz": float(thz_to_mhz(det["x_thz"], cfg.f_ref_thz)),
                    "expected_x_mhz": float(thz_to_mhz(exp["x_thz"], cfg.f_ref_thz)),
                    "offset_mhz": float((det["x_thz"] - exp["x_thz"]) * 1e6),
                }

            results.append({
                "rms_mhz": rms_mhz,
                "snr_score": snr_score,
                "assignment": assignment,
                "detected_combo": detected_combo,
                "expected_combo": expected_combo,
            })

    if not results:
        return None, None

    results = sorted(results, key=lambda r: (r["rms_mhz"], -r["snr_score"]))
    best = results[0]
    return best, best["assignment"]


def _save_fig(path: Path, show: bool = False) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def make_summary_plots(
    avg: Dict[str, Any],
    strong_peaks: List[Dict[str, Any]],
    isotope_assignment: Optional[Dict[str, Any]],
    local_peaks: List[Dict[str, Any]],
    single_fit: Optional[Dict[str, Any]],
    multi_fit: Optional[Dict[str, Any]],
    diagnostics: Dict[str, Any],
    cfg: AnalysisConfig,
    outdir: Path,
) -> List[Path]:
    paths = []
    x_plot = thz_to_mhz(avg["common_x"], cfg.f_ref_thz)

    # Full ratio plot, debug only
    if cfg.debug:
        plt.figure(figsize=(12, 5))
        plt.plot(x_plot, avg["ratio_mean_s"], color="red", lw=2, label="Mean smoothed ratio")
        y_min = np.min(avg["ratio_mean_s"])
        y_max = np.max(avg["ratio_mean_s"])
        y_range = y_max - y_min
        for p in strong_peaks:
            x_pk = thz_to_mhz(p["x"], cfg.f_ref_thz)
            plt.axvline(x_pk, color="blue", ls="--", alpha=0.6)
            plt.text(x_pk + 10, y_max + 0.05 * y_range, p["peak_label"], va="bottom", ha="left", fontsize=10, color="blue")
        plt.axhline(1.0, color="black", lw=0.8)
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Corrected ratio")
        plt.title("Full-range ratio")
        plt.ylim(y_min - 0.02 * y_range, y_max + 0.12 * y_range)
        path = outdir / prefixed_name(cfg, "full_ratio.png")
        _save_fig(path, cfg.show_plots)
        paths.append(path)

    # Full OD plot, debug only
    if cfg.debug:
        plt.figure(figsize=(12, 5))
        plt.plot(x_plot, avg["od_mean_s"], color="purple", lw=2, label="Averaged signal")
        y_min = np.min(avg["od_mean_s"])
        y_max = np.max(avg["od_mean_s"])
        y_range = y_max - y_min
        for p in strong_peaks:
            x_pk = thz_to_mhz(p["x"], cfg.f_ref_thz)
            plt.axvline(x_pk, color="blue", ls="--", alpha=0.6)
            plt.text(x_pk + 10, y_max + 0.05 * y_range, p["peak_label"], va="bottom", ha="left", fontsize=10, color="blue")
        plt.axhline(0.0, color="black", lw=0.8)
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Absorption signal (OD-like)")
        plt.title("Full-range absorption signal")
        plt.ylim(y_min - 0.02 * y_range, y_max + 0.12 * y_range)
        path = outdir / prefixed_name(cfg, "full_od.png")
        _save_fig(path, cfg.show_plots)
        paths.append(path)

    # Peak detection diagnostic, debug only
    if cfg.debug:
        plt.figure(figsize=(12, 5))
        plt.plot(x_plot, avg["od_mean_s"], label="Averaged signal", alpha=0.5)
        plt.plot(x_plot, diagnostics["od_detect"], label="Detection signal", lw=2)

        y_min = np.min(diagnostics["od_detect"])
        y_max = np.max(diagnostics["od_detect"])
        y_range = y_max - y_min

        found_local_labels = {p["label"] for p in local_peaks}

        for iso_lbl, x_pred in predicted_positions_from_assignment(
            isotope_assignment=isotope_assignment,
            cfg=cfg,
            x_min_mhz=float(np.min(x_plot)),
            x_max_mhz=float(np.max(x_plot)),
        ):
            if iso_lbl in found_local_labels:
                continue

            plt.axvline(x_pred, color="grey", ls=":", alpha=0.8)
            plt.text(
                x_pred + 6,
                y_min + 0.10 * y_range,
                f"pred {iso_lbl}",
                va="bottom",
                ha="left",
                fontsize=7,
                color="grey"
            )

        for p in strong_peaks:
            x0 = thz_to_mhz(p["region_start_x"], cfg.f_ref_thz)
            x1 = thz_to_mhz(p["region_end_x"], cfg.f_ref_thz)
            xc = thz_to_mhz(p["x"], cfg.f_ref_thz)

            plt.axvspan(x0, x1, color="orange", alpha=0.2)
            plt.axvline(xc, color="blue", ls="--", alpha=0.8)

            label_text = f"{p['peak_label']}\n{p['x']:.6f} THz"

            plt.text(
                xc + 8,
                y_max + 0.045 * y_range,
                label_text,
                va="bottom",
                ha="left",
                fontsize=7,
                color="blue"
            )

        for p in local_peaks:
            plt.axvline(p["x_mhz"], color="green", ls="--", alpha=0.9)

            label_text = (
                f"{p['label']}\n"
                f"{p['x_mhz']:.3f} MHz"
            )

            plt.text(
                p["x_mhz"] + 8,
                y_max + 0.02 * y_range,
                label_text,
                va="bottom",
                ha="left",
                fontsize=7,
                color="green"
            )

        plt.axhline(0.0, color="black", lw=0.8)
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Absorption signal (OD-like)")
        plt.title("Peak detection")
        plt.ylim(y_min - 0.02 * y_range, y_max + 0.12 * y_range)
        plt.legend()
        path = outdir / prefixed_name(cfg, "peak_detection.png")
        _save_fig(path, cfg.show_plots)
        paths.append(path)

    if cfg.debug and single_fit is not None:
        x_fit_plot = thz_to_mhz(single_fit["x_fit"], cfg.f_ref_thz)
        x0_plot = thz_to_mhz(single_fit["centre_thz"], cfg.f_ref_thz)
        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax[0].plot(x_fit_plot, single_fit["y_fit"], "o", ms=3, label="Ratio data in local window")
        ax[0].plot(x_fit_plot, single_fit["y_model"], "-", lw=2, label="1-peak Voigt fit")
        ax[0].axvline(x0_plot, color="blue", ls="--", label=f"{single_fit['peak_label']} centre = {x0_plot:.3f} MHz")
        y_min = np.min(single_fit["y_fit"])
        y_max = np.max(single_fit["y_fit"])
        y_range = y_max - y_min
        x_min_plot = np.min(x_fit_plot); x_max_plot = np.max(x_fit_plot)
        if x0_plot > 0.75 * x_max_plot + 0.25 * x_min_plot:
            x_text = x0_plot - 8; ha_text = "right"
        else:
            x_text = x0_plot + 5; ha_text = "left"
        ax[0].text(x_text, y_max + 0.08 * y_range, single_fit["peak_label"], va="bottom", ha=ha_text, fontsize=10, color="blue")
        ax[0].axhline(1.0, color="black", lw=0.8)
        ax[0].set_ylabel("Corrected ratio")
        ax[0].set_title("Single-peak fit")
        ax[0].legend(loc="best")
        ax[0].set_ylim(y_min - 0.02 * y_range, y_max + 0.15 * y_range)
        ax[1].plot(x_fit_plot, single_fit["resid"], "-", lw=1.5)
        ax[1].axhline(0.0, color="black", lw=0.8)
        ax[1].set_xlabel("Frequency (MHz)")
        ax[1].set_ylabel("Residual")
        ax[1].set_title("Fit residuals")
        fig.subplots_adjust(hspace=0.25)
        path = outdir / prefixed_name(cfg, "single_peak_fit.png")
        _save_fig(path, cfg.show_plots)
        paths.append(path)

    if multi_fit is not None and multi_fit["fit_results"]:
        x_fit_plot = thz_to_mhz(multi_fit["x_fit"], cfg.f_ref_thz)

        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

        ax[0].plot(x_fit_plot, multi_fit["y_fit"], "o", ms=3, label="Ratio data in selected window")
        ax[0].plot(x_fit_plot, multi_fit["baseline_fit"], "--", lw=1.5, label="Shared baseline")
        ax[0].plot(x_fit_plot, multi_fit["y_model"], "-", lw=2.2, label="Global Voigt fit")

        y_min = np.min(multi_fit["y_fit"])
        y_max = np.max(multi_fit["y_fit"])
        y_range = y_max - y_min
        x_min_plot = np.min(x_fit_plot)
        x_max_plot = np.max(x_fit_plot)

        for r in multi_fit["fit_results"]:
            x_local_thz = multi_fit["x_fit"]
            x_local_plot = x_fit_plot

            baseline_local = multi_fit["baseline_fit"]
            y_peak_only = baseline_local + r["amplitude"] * voigt_profile(
                x_local_thz - r["centre_thz"],
                r["sigma_thz"],
                r["gamma_thz"],
            )

            x0_plot = thz_to_mhz(r["centre_thz"], cfg.f_ref_thz)
            ax[0].plot(x_local_plot, y_peak_only, "-", lw=1.5, alpha=0.8, label=f"{r['peak_label']} component")
            ax[0].axvline(x0_plot, color="blue", ls="--", alpha=0.7)

            if x0_plot > 0.75 * x_max_plot + 0.25 * x_min_plot:
                x_text = x0_plot - 8
                ha_text = "right"
            else:
                x_text = x0_plot + 5
                ha_text = "left"

            ax[0].text(
                x_text,
                y_max + 0.08 * y_range,
                r["peak_label"],
                va="bottom",
                ha=ha_text,
                fontsize=10,
                color="blue",
            )

        ax[0].axhline(1.0, color="black", lw=0.8)
        ax[0].set_ylabel("Corrected ratio")
        ax[0].set_title("Global constrained Voigt fit on selected ratio features")
        ax[0].legend(loc="best")
        ax[0].set_ylim(y_min - 0.02 * y_range, y_max + 0.15 * y_range)

        ax[1].plot(x_fit_plot, multi_fit["resid"], "-", lw=1.5)
        ax[1].axhline(0.0, color="black", lw=0.8)
        ax[1].set_xlabel("Frequency (MHz)")
        ax[1].set_ylabel("Residual")
        ax[1].set_title("Fit residuals")

        fig.subplots_adjust(hspace=0.25)

        path = outdir / prefixed_name(cfg, "selected_peak_fits.png")
        _save_fig(path, cfg.show_plots)
        paths.append(path)

    return paths

def make_isotope_assignment_plot(
    avg: Dict[str, Any],
    strong_peaks: List[Dict[str, Any]],
    isotope_assignment: Optional[Dict[str, Any]],
    local_peaks: List[Dict[str, Any]],
    multi_fit: Optional[Dict[str, Any]],
    diagnostics: Dict[str, Any],
    cfg: AnalysisConfig,
    outdir: Path,
) -> Path:
    x_plot = thz_to_mhz(avg["common_x"], cfg.f_ref_thz)

    plt.figure(figsize=(12, 5))
    plt.plot(x_plot, diagnostics["od_detect"], label="Averaged signal", lw=2)

    y_min = np.min(diagnostics["od_detect"])
    y_max = np.max(diagnostics["od_detect"])
    y_range = y_max - y_min

    # Build quick lookup for fitted FWHM by detected peak label
    fwhm_by_label = {}
    if multi_fit is not None and multi_fit.get("fit_results"):
        for r in multi_fit["fit_results"]:
            fwhm_by_label[r["peak_label"]] = r["fwhm_mhz"]

    # Predicted isotope positions from assigned anchors
    # If a local peak has already been found for an isotope, do not also draw the grey predicted line.
    found_local_labels = {p["label"] for p in local_peaks}

    for iso_lbl, x_pred in predicted_positions_from_assignment(
        isotope_assignment=isotope_assignment,
        cfg=cfg,
        x_min_mhz=float(np.min(x_plot)),
        x_max_mhz=float(np.max(x_plot)),
    ):
        if iso_lbl in found_local_labels:
            continue

        plt.axvline(x_pred, color="grey", ls=":", alpha=0.8)
        plt.text(
            x_pred + 6,
            y_min + 0.10 * y_range,
            f"pred {iso_lbl}",
            va="bottom",
            ha="left",
            fontsize=7,
            color="grey"
        )

    for p in strong_peaks:
        xc = thz_to_mhz(p["x"], cfg.f_ref_thz)
        plt.axvline(xc, color="blue", ls="--", alpha=0.8)

        if isotope_assignment is not None and p["peak_label"] in isotope_assignment:
            info = isotope_assignment[p["peak_label"]]
            label_text = (
                f"{p['peak_label']} -> {info['expected_label']}\n"
                f"{p['x']:.6f} THz\n"
                f"offset = {info['offset_mhz']:.3f} MHz"
            )
        else:
            label_text = (
                f"{p['peak_label']}\n"
                f"{p['x']:.6f} THz"
            )

        if p["peak_label"] in fwhm_by_label:
            label_text += f"\nFWHM = {fwhm_by_label[p['peak_label']]:.3f} MHz"

        plt.text(
            xc + 8,
            y_max - 0.08 * y_range,
            label_text,
            va="bottom",
            ha="left",
            fontsize=7,
            color="blue"
        )

    # Local predicted-peak search results, e.g. weak Dy163
    for p in local_peaks:
        plt.axvline(p["x_mhz"], color="green", ls="--", alpha=0.9)

        x_thz = cfg.f_ref_thz + p["x_mhz"] / 1e6
        offset_mhz = p["x_mhz"] - p["predicted_mhz"]

        label_text = (
            f"{p['label']}\n"
            f"{x_thz:.6f} THz\n"
            f"offset = {offset_mhz:.3f} MHz"
        )

        # Add fitted FWHM if this local peak made it into the multi-fit results
        peak_label = p.get("peak_label", None)
        if peak_label in fwhm_by_label:
            label_text += f"\nFWHM = {fwhm_by_label[peak_label]:.3f} MHz"

        plt.text(
            p["x_mhz"] + 8,
            y_max - 0.25 * y_range,
            label_text,
            va="bottom",
            ha="left",
            fontsize=7,
            color="green"
        )

    plt.axhline(0.0, color="black", lw=0.8)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Absorption signal (OD-like)")
    plt.title("Detected peaks with isotope assignment")
    plt.ylim(y_min - 0.02 * y_range, y_max + 0.16 * y_range)

    path = outdir / prefixed_name(cfg, "isotope_assignment.png")
    _save_fig(path, cfg.show_plots)
    return path

def save_csvs(strong_peaks: List[Dict[str, Any]], multi_fit: Optional[Dict[str, Any]], isotopes: Optional[Dict[str, Any]], cfg: AnalysisConfig, outdir: Path) -> Tuple[Optional[Path], Path]:
    strong_path = None

    if cfg.debug:
        rows = []
        for p in strong_peaks:
            assignment = isotopes.get(p["peak_label"]) if isotopes else None
            rows.append({
                "detected_label": p["peak_label"],
                "position_thz": p["x"],
                "position_mhz": float(thz_to_mhz(p["x"], cfg.f_ref_thz)),
                "od_area_snr": p["area_snr"],
                "ratio_bump_snr": p["ratio_bump_snr"],
                "assigned_isotope": None if assignment is None else assignment["expected_label"],
                "offset_mhz": None if assignment is None else assignment["offset_mhz"],
            })
        strong_df = pd.DataFrame(rows)
        strong_path = outdir / prefixed_name(cfg, "strong_peaks.csv")
        strong_df.to_csv(strong_path, index=False)

    fit_rows = []
    if multi_fit is not None:
        fit_type = multi_fit.get("fit_type", "unknown")

        for r in multi_fit["fit_results"]:
            assignment = r.get("isotope_assignment")
            fit_rows.append({
                "ec_temp": cfg.ec_temp,
                "hl_temp": cfg.hl_temp,
                "fit_type": fit_type,
                "peak_label": r["peak_label"],
                "assigned_isotope": None if assignment is None else assignment["expected_label"],
                "centre_thz": r["centre_thz"],
                "centre_mhz": float(thz_to_mhz(r["centre_thz"], cfg.f_ref_thz)),
                "offset_mhz": None if assignment is None else assignment["offset_mhz"],
                "amplitude": r["amplitude"],
                "fwhm_mhz": r["fwhm_mhz"],
                "centre_err_thz": r["centre_err_thz"],
                "fwhm_err_mhz": r["fwhm_err_mhz"],
                "sigma_thz": r["sigma_thz"],
                "sigma_err_thz": r["sigma_err_thz"],
                "gamma_thz": r["gamma_thz"],
                "gamma_err_thz": r["gamma_err_thz"],
            })

    fit_df = pd.DataFrame(fit_rows)
    fit_path = outdir / prefixed_name(cfg, "fit_results.csv")
    fit_df.to_csv(fit_path, index=False)
    return strong_path, fit_path

def save_per_ramp_csvs(per_ramp_analysis: Optional[Dict[str, Any]], cfg: AnalysisConfig, outdir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    if per_ramp_analysis is None:
        return None, None

    per_ramp_df = per_ramp_analysis.get("per_ramp_df")
    per_ramp_summary_df = per_ramp_analysis.get("per_ramp_summary_df")

    per_ramp_path = None
    per_ramp_summary_path = None

    if per_ramp_df is not None and not per_ramp_df.empty:
        per_ramp_path = outdir / prefixed_name(cfg, "per_ramp_fit_results.csv")
        per_ramp_df.to_csv(per_ramp_path, index=False)

    if per_ramp_summary_df is not None and not per_ramp_summary_df.empty:
        per_ramp_summary_path = outdir / prefixed_name(cfg, "per_ramp_fit_summary.csv")
        per_ramp_summary_df.to_csv(per_ramp_summary_path, index=False)

    return per_ramp_path, per_ramp_summary_path

def save_summary_text(
    processed: List[Dict[str, Any]],
    strong_peaks: List[Dict[str, Any]],
    local_peaks: List[Dict[str, Any]],
    isotope_assignment: Optional[Dict[str, Any]],
    single_fit: Optional[Dict[str, Any]],
    multi_fit: Optional[Dict[str, Any]],
    per_ramp_analysis: Optional[Dict[str, Any]],
    physics_df: Optional[pd.DataFrame],
    cfg: AnalysisConfig,
    outdir: Path,
) -> Path:
    lines = []
    lines.append("Absorption scan analysis summary")
    lines.append("")
    lines.append(f"Scope file: {cfg.scope_file}")
    lines.append(f"Wavemeter file: {cfg.wavemeter_file}")
    if cfg.ec_temp is not None:
        lines.append(f"EC temperature: {cfg.ec_temp:.0f} C")
    if cfg.hl_temp is not None:
        lines.append(f"HL temperature: {cfg.hl_temp:.0f} C")
    lines.append(f"Valid rising ramps: {len(processed)}")
    lines.append(f"Strong peaks found: {len(strong_peaks)}")
    lines.append("")
    if isotope_assignment:
        lines.append("Isotope assignment")
        for det_label, info in isotope_assignment.items():
            lines.append(f"{det_label} -> {info['expected_label']} | {info['detected_x_thz']:.9f} THz | offset = {info['offset_mhz']:.3f} MHz")
        lines.append("")
    if local_peaks:
        lines.append("Local peak search")
        for p in local_peaks:
            x_thz = cfg.f_ref_thz + p["x_mhz"] / 1e6
            offset_mhz = p["x_mhz"] - p["predicted_mhz"]
            lines.append(
                f"{p['peak_label']} -> {p['label']} | "
                f"{x_thz:.9f} THz | offset = {offset_mhz:.3f} MHz | "
                f"SNR = {p['snr']:.2f}"
            )
        lines.append("")
    if cfg.debug and single_fit:
        a = single_fit["isotope_assignment"]
        lines.append("Single peak fit")
        lines.append(f"{single_fit['peak_label']} | centre = {single_fit['centre_thz']:.9f} THz | FWHM = {single_fit['fwhm_mhz']:.3f} MHz | offset = {a['offset_mhz']:.3f} MHz")
        lines.append("")
    if multi_fit and multi_fit["fit_results"]:
        lines.append("Global selected peak fit")
        lines.append(f"Fit type = {multi_fit.get('fit_type', 'unknown')}")
        for r in multi_fit["fit_results"]:
            a = r["isotope_assignment"]
            lines.append(
                f"{r['peak_label']} -> {a['expected_label']} | "
                f"{r['centre_thz']:.9f} THz | "
                f"offset = {a['offset_mhz']:.3f} MHz | "
                f"FWHM = {r['fwhm_mhz']:.3f} MHz"
            )
        lines.append(f"Residual RMS = {multi_fit['resid_rms']:.6f}")
        lines.append("")

    if per_ramp_analysis is not None:
        per_ramp_summary_df = per_ramp_analysis.get("per_ramp_summary_df")
        if per_ramp_summary_df is not None and not per_ramp_summary_df.empty:
            lines.append("Per-ramp fit summary")
            for _, row in per_ramp_summary_df.iterrows():
                lines.append(
                    f"{row['peak_label']} -> {row['assigned_isotope']} | "
                    f"n_ramps = {int(row['n_ramps'])} | "
                    f"offset mean = {row['offset_mean_mhz']:.3f} MHz | "
                    f"offset std = {row['offset_std_mhz']:.3f} MHz"
                )

            valid_stds = per_ramp_summary_df["offset_std_mhz"].dropna()

            if len(valid_stds) > 0:
                jitter_mean = float(np.mean(valid_stds))
                jitter_median = float(np.median(valid_stds))

                lines.append("")
                lines.append("System frequency uncertainty")
                lines.append(f"Estimated jitter (mean) = {jitter_mean:.2f} MHz")
                lines.append(f"Estimated jitter (median) = {jitter_median:.2f} MHz")
    if physics_df is not None and not physics_df.empty:
        lines.append("")
        lines.append("==============================")
        lines.append("Quick Experimental Summary")
        lines.append("==============================")
        lines.append("")

        # Sort once by fitted centre frequency so every section uses the same order
        physics_df_sorted = physics_df.sort_values("centre_thz").reset_index(drop=True)

        lines.append("Experimental conditions")
        lines.append("-----------------------")
        if cfg.ec_temp is not None:
            lines.append(f"EC temperature: {cfg.ec_temp:.0f} C")
        if cfg.hl_temp is not None:
            lines.append(f"HL temperature: {cfg.hl_temp:.0f} C")
        lines.append("")

        lines.append("Data quality")
        lines.append("------------")
        lines.append(f"Valid scans: {len(processed)}")

        # Scan centre frequency from the fitted peak centres
        if physics_df is not None and not physics_df.empty:
            centre_vals = pd.to_numeric(physics_df["centre_thz"], errors="coerce").dropna()
            if len(centre_vals) > 0:
                scan_centre_thz = float(np.mean(centre_vals))
                lines.append(f"Scan centre frequency: {scan_centre_thz:.6f} THz")

        if per_ramp_analysis is not None:
            per_ramp_summary_df = per_ramp_analysis.get("per_ramp_summary_df")
            if per_ramp_summary_df is not None and not per_ramp_summary_df.empty:
                valid_stds = per_ramp_summary_df["offset_std_mhz"].dropna()
                if len(valid_stds) > 0:
                    jitter_median = float(np.median(valid_stds))
                    lines.append(f"Frequency uncertainty (shot-to-shot): ±{jitter_median:.1f} MHz")
        lines.append("")

        lines.append("Identified transitions")
        lines.append("----------------------")
        for _, row in physics_df_sorted.iterrows():
            lines.append(
                f"{row['assigned_isotope']}: {row['centre_thz']:.6f} THz, "
                f"offset {row['offset_mhz']:+.1f} MHz"
            )
        lines.append("")

        lines.append("Fitted peak widths, Voigt fit")
        lines.append("-----------------------------")
        for _, row in physics_df_sorted.iterrows():
            lines.append(
                f"{row['assigned_isotope']}: {row['voigt_fwhm_mhz']:.1f} MHz"
            )
        lines.append("")

        lines.append("Atomic beam properties")
        lines.append("----------------------")
        beam_speed_vals = physics_df_sorted["beam_speed_ms"].dropna()
        if len(beam_speed_vals) > 0:
            lines.append(f"Mean beam speed: {beam_speed_vals.iloc[0]:.1f} m/s")
        for _, row in physics_df_sorted.iterrows():
            lines.append(
                f"{row['assigned_isotope']} transverse velocity spread: "
                f"{row['velocity_sigma_ms']:.2f} m/s"
            )
        lines.append("")

        lines.append("Beam size at probe")
        lines.append("------------------")
        for _, row in physics_df_sorted.iterrows():
            lines.append(
                f"{row['assigned_isotope']} estimated beam diameter: "
                f"{row['beam_diameter_mm']:.1f} mm"
            )
        lines.append("")

        lines.append("Absorption and atomic density")
        lines.append("-----------------------------")
        for _, row in physics_df_sorted.iterrows():
            lines.append(
                f"{row['assigned_isotope']} peak optical depth: {row['peak_od']:.5f}"
            )
        lines.append("")
        for _, row in physics_df_sorted.iterrows():
            density_val = row["number_density_m3"]
            if np.isfinite(density_val):
                density_str = f"{density_val:.3g}"
            else:
                density_str = "nan"
            lines.append(
                f"{row['assigned_isotope']} density: {density_str} m^-3"
            )
    path = outdir / prefixed_name(cfg, "summary.txt")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_ramp_comparison_plot(
    avg_rising: Dict[str, Any],
    avg_falling: Dict[str, Any],
    multi_fit_rising: Optional[Dict[str, Any]],
    multi_fit_falling: Optional[Dict[str, Any]],
    cfg: AnalysisConfig,
    outdir: Path,
    output_filename: str = "ramp_comparison.png",
) -> Path:
    """
    Overlay the averaged OD spectra from rising and falling ramps, with fitted
    peak centres marked. Used to validate that both ramp directions agree before
    folding them together.
    """
    x_rising = thz_to_mhz(avg_rising["common_x"], cfg.f_ref_thz)
    x_falling = thz_to_mhz(avg_falling["common_x"], cfg.f_ref_thz)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False,
                             gridspec_kw={"hspace": 0.4})

    # --- Top panel: OD overlay ---
    ax = axes[0]
    ax.plot(x_rising, avg_rising["od_mean_s"], color="royalblue", lw=1.5,
            label=f"Rising ramps (n={len(avg_rising.get('od_corr_matrix', [[]]))})")
    ax.plot(x_falling, avg_falling["od_mean_s"], color="tomato", lw=1.5,
            ls="--", label=f"Falling ramps (n={len(avg_falling.get('od_corr_matrix', [[]]))})")

    # Mark rising fit centres
    if multi_fit_rising and multi_fit_rising.get("fit_results"):
        for r in multi_fit_rising["fit_results"]:
            xc = thz_to_mhz(r["centre_thz"], cfg.f_ref_thz)
            ax.axvline(xc, color="royalblue", ls=":", lw=1.0, alpha=0.8)

    # Mark falling fit centres
    if multi_fit_falling and multi_fit_falling.get("fit_results"):
        for r in multi_fit_falling["fit_results"]:
            xc = thz_to_mhz(r["centre_thz"], cfg.f_ref_thz)
            ax.axvline(xc, color="tomato", ls=":", lw=1.0, alpha=0.8)

    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("OD (baseline corrected)")
    ax.set_title("Rising vs falling ramp comparison — OD spectra")
    ax.legend(loc="best", fontsize=9)

    # --- Bottom panel: peak centre comparison table ---
    ax2 = axes[1]
    ax2.axis("off")

    rows = []
    headers = ["Isotope", "Rising centre (MHz)", "Falling centre (MHz)", "Difference (MHz)"]

    if (multi_fit_rising and multi_fit_rising.get("fit_results") and
            multi_fit_falling and multi_fit_falling.get("fit_results")):

        rising_centres = {
            r["isotope_assignment"]["expected_label"]: thz_to_mhz(r["centre_thz"], cfg.f_ref_thz)
            for r in multi_fit_rising["fit_results"]
            if r.get("isotope_assignment") and r["isotope_assignment"].get("expected_label")
        }
        falling_centres = {
            r["isotope_assignment"]["expected_label"]: thz_to_mhz(r["centre_thz"], cfg.f_ref_thz)
            for r in multi_fit_falling["fit_results"]
            if r.get("isotope_assignment") and r["isotope_assignment"].get("expected_label")
        }

        all_isotopes = sorted(set(rising_centres) | set(falling_centres))
        for iso in all_isotopes:
            rc = rising_centres.get(iso, float("nan"))
            fc = falling_centres.get(iso, float("nan"))
            diff = fc - rc if (np.isfinite(rc) and np.isfinite(fc)) else float("nan")
            rows.append([
                iso,
                f"{rc:.2f}" if np.isfinite(rc) else "—",
                f"{fc:.2f}" if np.isfinite(fc) else "—",
                f"{diff:+.2f}" if np.isfinite(diff) else "—",
            ])

        # Also add OD comparison
        rising_ods = {
            r["isotope_assignment"]["expected_label"]: r.get("amplitude", float("nan"))
            for r in multi_fit_rising["fit_results"]
            if r.get("isotope_assignment") and r["isotope_assignment"].get("expected_label")
        }
        falling_ods = {
            r["isotope_assignment"]["expected_label"]: r.get("amplitude", float("nan"))
            for r in multi_fit_falling["fit_results"]
            if r.get("isotope_assignment") and r["isotope_assignment"].get("expected_label")
        }

    if rows:
        tbl = ax2.table(
            cellText=rows,
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
        ax2.set_title("Fitted peak centre comparison", pad=8, fontsize=10)
    else:
        ax2.text(0.5, 0.5, "Fit results not available for one or both ramp directions",
                 ha="center", va="center", fontsize=9, transform=ax2.transAxes)

    path = outdir / prefixed_name(cfg, output_filename)
    _save_fig(path, cfg.show_plots)
    return path


def correct_falling_ramp_freq_axes(
    falling_processed: List[Dict[str, Any]],
    selected_peaks_for_ramps: List[Dict[str, Any]],
    cfg: AnalysisConfig,
    common_x: np.ndarray,
    target_centre_thz: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Correct the frequency axis of each falling ramp by fitting the peak centres
    using fixed widths from the rising ramp averaged fit, then shifting each
    falling ramp's frequency axis so its anchor peak lands at target_centre_thz.

    selected_peaks_for_ramps: peak positions and widths used as fit guesses.
        Positions should match where peaks actually are in falling ramp data.
        Widths should come from the rising ramp averaged fit.

    target_centre_thz: the frequency (THz) the anchor peak should land at after
        correction. If None, uses selected_peaks_for_ramps[0]["x"] as target
        (i.e. corrects to the falling average position, not the rising position).
        Pass the rising ramp anchor centre here to align falling to rising.

    Returns:
        corrected_falling: list of ramp dicts with shifted freq arrays
        offsets_mhz: per-ramp frequency offset applied (MHz), nan if ramp failed
    """
    corrected_falling = []
    offsets_mhz = []

    if len(selected_peaks_for_ramps) == 0:
        return falling_processed, [0.0] * len(falling_processed)

    anchor_peak_label = selected_peaks_for_ramps[0]["peak_label"]

    # target_centre_thz is where we WANT the anchor peak to land (rising average)
    # anchor position in selected_peaks_for_ramps[0]["x"] is where it IS in falling data
    if target_centre_thz is None:
        target_centre_thz = selected_peaks_for_ramps[0]["x"]

    # Build a common_x that covers the falling ramp frequency range
    # so per-ramp fits are not extrapolating outside the data
    x_min_fall = max(np.nanmin(r["freq"]) for r in falling_processed)
    x_max_fall = min(np.nanmax(r["freq"]) for r in falling_processed)
    common_x_falling = np.linspace(x_min_fall, x_max_fall, 1200)

    for ramp in falling_processed:
        avg_like = single_ramp_to_avg_like(ramp, cfg, common_x_falling)
        fit = fit_selected_peaks_global_with_fixed_labels(avg_like, selected_peaks_for_ramps, cfg)

        if fit is None or len(fit["fit_results"]) == 0:
            offsets_mhz.append(float("nan"))
            corrected_falling.append(None)
            continue

        # Find the anchor peak fit result
        anchor_result = None
        for r in fit["fit_results"]:
            if r["peak_label"] == anchor_peak_label:
                anchor_result = r
                break

        if anchor_result is None:
            offsets_mhz.append(float("nan"))
            corrected_falling.append(None)
            continue

        # Offset = where the anchor peak IS in this ramp - where we WANT it to be
        offset_thz = anchor_result["centre_thz"] - target_centre_thz
        offset_mhz = float(offset_thz * 1e6)
        offsets_mhz.append(offset_mhz)

        # Shift the frequency axis to bring anchor peak onto target_centre_thz
        corrected_ramp = dict(ramp)
        corrected_ramp["freq"] = ramp["freq"] - offset_thz
        corrected_falling.append(corrected_ramp)

    # Filter out None entries (failed ramps)
    n_total = len(corrected_falling)
    corrected_falling = [r for r in corrected_falling if r is not None]
    n_ok = len(corrected_falling)

    if cfg.debug:
        print(f"[falling correction] {n_ok}/{n_total} falling ramps successfully corrected")
        valid_offsets = [o for o in offsets_mhz if np.isfinite(o)]
        if valid_offsets:
            print(f"[falling correction] offset mean = {np.mean(valid_offsets):.1f} MHz, "
                  f"std = {np.std(valid_offsets):.1f} MHz, "
                  f"range = [{np.min(valid_offsets):.1f}, {np.max(valid_offsets):.1f}] MHz")

    return corrected_falling, offsets_mhz


def analyze(cfg: AnalysisConfig) -> Dict[str, Any]:
    import time as _time
    _t0 = _time.perf_counter()
    def _lap(label: str) -> None:
        print(f"[timing] {label}: {_time.perf_counter() - _t0:.3f}s")

    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cfg.expected_peak_positions_thz is None:
        cfg.expected_peak_positions_thz = default_expected_peaks()

    scope = load_scope_npz(cfg.scope_file)
    wm_ok = load_wavemeter_csv(cfg.wavemeter_file)
    _lap("load data")

    if cfg.debug:
        wm_t_arr = wm_ok["t_rel_perf"].to_numpy()
        polling_intervals = np.diff(wm_t_arr)
        print(f"[wavemeter] n_samples = {len(wm_t_arr)}")
        print(f"[wavemeter] mean polling interval = {np.mean(polling_intervals)*1000:.2f} ms")
        print(f"[wavemeter] median polling interval = {np.median(polling_intervals)*1000:.2f} ms")
        print(f"[wavemeter] max polling interval = {np.max(polling_intervals)*1000:.2f} ms")
        print(f"[wavemeter] total duration = {wm_t_arr[-1]:.3f} s")

    align = align_and_interpolate_frequency(scope, wm_ok, cfg)
    prepared = prepare_binned_scope(
        scope,
        align["freq_interp_raw"],
        cfg,
        freq_interp_raw_falling=align["freq_interp_raw_falling"] if cfg.use_falling_ramps else None,
    )
    _lap(f"align + prepare  [best_shift={align['best_shift']:.4f}s, score={align['best_score']:.4f}]")

    if cfg.debug:
        print(f"[scope] binned samples = {len(prepared['t'])}")
        print(f"[scope] binned sample rate = {1.0/(prepared['t'][1]-prepared['t'][0]):.1f} Hz")
        print(f"[scope] freq_clean_kernel = {cfg.freq_clean_kernel} samples = {cfg.freq_clean_kernel*(prepared['t'][1]-prepared['t'][0])*1000:.2f} ms")

    scan_smooth, ramp_slices = detect_rising_ramps(prepared["t"], prepared["scan"], cfg)

    if cfg.debug:
        print(f"[alignment] best_shift = {align['best_shift']:.6f} s  (rising)")
        if cfg.use_falling_ramps:
            print(f"[alignment] best_shift_falling = {align['best_shift_falling']:.6f} s  (falling)")
        print(f"[alignment] best_score = {align['best_score']:.6f}")
        print(f"[ramps] {len(ramp_slices)} rising ramps detected")
        if len(ramp_slices) > 0:
            durations = [(r.stop - r.start) * (prepared['t'][1] - prepared['t'][0]) for r in ramp_slices]
            print(f"[ramps] mean ramp duration = {np.mean(durations):.4f} s")

    processed = []
    for r in ramp_slices:
        processed.append(process_rising_ramp(
            prepared["t"][r],
            prepared["scan"][r],
            prepared["freq_interp"][r],
            prepared["probe"][r],
            prepared["ref"][r],
            edge_fraction=cfg.edge_fraction,
            eps=cfg.eps,
            max_shift=cfg.max_shift,
        ))
    _lap(f"process {len(processed)} rising ramps")

    avg = average_ramps_to_common_axis(processed, cfg)
    _lap("average ramps")

    # First-pass detection on the raw averaged traces
    candidates, strong_peaks, peak_diag = detect_strong_peaks(avg, cfg)
    best_spacing_match, isotope_assignment = match_isotopes(strong_peaks, cfg)

    # Use first-pass peak positions to remove any slow curved baseline from the averaged traces
    peak_positions_thz = [p["x"] for p in strong_peaks]
    avg = correct_average_baseline(avg, peak_positions_thz, cfg)
    _lap("baseline correction")

    # Re-run detection and assignment on the baseline-corrected averaged traces
    candidates, strong_peaks, peak_diag = detect_strong_peaks(avg, cfg)
    best_spacing_match, isotope_assignment = match_isotopes(strong_peaks, cfg)

    # Local search for weak predicted isotopes, starting with Dy163
    x_mhz = thz_to_mhz(avg["common_x"], cfg.f_ref_thz)
    y_detect = peak_diag["od_detect"]

    local_peaks = []

    predicted = predicted_positions_from_assignment(
        isotope_assignment=isotope_assignment,
        cfg=cfg,
        x_min_mhz=float(np.min(x_mhz)),
        x_max_mhz=float(np.max(x_mhz)),
    )

    # Quick Voigt fit of Dy164 so we can subtract its tail before searching for Dy163.
    # This is the cleanest baseline for the Dy163 shoulder search — we know exactly
    # what Dy164 contributes at the Dy163 position from its own fit.
    dy164_mhz = None
    y_dy164_subtracted = y_detect.copy()

    if isotope_assignment:
        for _, info in isotope_assignment.items():
            if info.get("expected_label") == "Dy164":
                dy164_mhz = float(thz_to_mhz(info["detected_x_thz"], cfg.f_ref_thz))
                break

    if dy164_mhz is not None:
        try:
            # Fit Dy164 alone in a narrow window around its peak
            fit_hw = 80.0  # MHz half-width for the Dy164 fit window
            mask164 = (x_mhz > dy164_mhz - fit_hw) & (x_mhz < dy164_mhz + fit_hw)
            if np.sum(mask164) > 20:
                x164 = x_mhz[mask164]
                y164 = y_detect[mask164]
                # Initial guesses
                A0    = float(np.max(y164) - np.median(y164))
                sig0  = 12e-3   # ~12 MHz in MHz units
                gam0  = 2.0    # ~2 MHz probe laser linewidth in MHz units
                p0    = [np.median(y164), 0.0, dy164_mhz, A0, sig0, gam0]
                x164_thz = cfg.f_ref_thz + x164 / 1e6
                x0_thz   = cfg.f_ref_thz + dy164_mhz / 1e6
                from scipy.optimize import curve_fit as _cf
                def _voigt_mhz(xm, c0, c1, x0, A, sigma, gamma):
                    return c0 + c1*(xm-x0) + A * voigt_profile(xm-x0, abs(sigma), abs(gamma))
                lb = [-np.inf, -np.inf, -np.inf, 0.0, 1e-3, 1.0]
                ub = [ np.inf,  np.inf,  np.inf, np.inf, 50.0, 6.0]
                popt, _ = _cf(_voigt_mhz, x164, y164, p0=p0, bounds=(lb, ub), maxfev=2000)
                # Evaluate Dy164 model across full x range and subtract
                dy164_model = _voigt_mhz(x_mhz, *popt)
                baseline_only = popt[0] + popt[1] * (x_mhz - popt[2])
                y_dy164_subtracted = y_detect - (dy164_model - baseline_only)
        except Exception:
            pass  # If fit fails, fall back to raw signal

    for iso, pred_mhz in predicted:
        # For Dy163 use the Dy164-subtracted signal so its shoulder is clearly visible
        if iso == "Dy163":
            peak = find_local_peak_near_prediction(
                x_mhz=x_mhz,
                y=y_dy164_subtracted,
                pred_mhz=pred_mhz,
                window_mhz=cfg.local_search_window_mhz,
                min_snr=max(1.5, cfg.local_search_min_snr * 0.5),
                edge_fraction=cfg.local_search_edge_fraction,
            )
        else:
            peak = find_local_peak_near_prediction(
                x_mhz=x_mhz,
                y=y_detect,
                pred_mhz=pred_mhz,
                window_mhz=cfg.local_search_window_mhz,
                min_snr=cfg.local_search_min_snr,
                edge_fraction=cfg.local_search_edge_fraction,
            )

        if peak is not None:
            peak["x"] = cfg.f_ref_thz + peak["x_mhz"] / 1e6

            # Do not add a local peak if it sits on top of an already detected strong peak
            too_close = any(abs(peak["x"] - p["x"]) * 1e6 < 25.0 for p in strong_peaks)
            if too_close:
                continue

            peak["label"] = iso
            peak["peak_label"] = f"P{len(strong_peaks) + len(local_peaks) + 1}"
            peak["area_snr"] = np.nan
            peak["ratio_bump_snr"] = np.nan
            local_peaks.append(peak)

    single_fit = fit_single_peak(avg, strong_peaks, cfg)

    # Dy163 and Dy161 are detected and noted (for plots/summary) but excluded from
    # the global Voigt fit. Including them destabilises the Dy164 fit in noisy datasets
    # and they are not needed for oven characterisation (flux, velocity spread).
    EXCLUDE_FROM_FIT = {"Dy163", "Dy161"}
    fit_local_peaks = [p for p in local_peaks if p.get("label") not in EXCLUDE_FROM_FIT]
    fit_input_peaks = sorted(strong_peaks + fit_local_peaks, key=lambda p: p["x"])

    fit_cfg = AnalysisConfig(**asdict(cfg))
    if cfg.multi_fit_peak_indices is None:
        fit_cfg.multi_fit_peak_indices = list(range(len(fit_input_peaks)))
    else:
        local_indices = list(range(len(strong_peaks), len(fit_input_peaks)))
        combined_indices = list(cfg.multi_fit_peak_indices) + local_indices

        # Deduplicate while preserving order and discard out-of-range indices
        deduped_indices = []
        seen = set()
        for idx in combined_indices:
            if 0 <= idx < len(fit_input_peaks) and idx not in seen:
                deduped_indices.append(idx)
                seen.add(idx)

        fit_cfg.multi_fit_peak_indices = deduped_indices

    multi_fit = fit_selected_peaks_global(avg, fit_input_peaks, fit_cfg)
    _lap("global Voigt fit")

    per_ramp_analysis = None
    if cfg.save_per_ramp_fits and multi_fit is not None and multi_fit.get("fit_results"):
        # Use the averaged-fit peak parameters as the fixed-shape reference for per-ramp centre fits
        selected_peaks_for_ramps = []
        for r in multi_fit["fit_results"]:
            selected_peaks_for_ramps.append({
                "peak_label": r["peak_label"],
                "x": r["centre_thz"],
                "sigma_thz": r["sigma_thz"],
                "gamma_thz": r["gamma_thz"],
            })

        per_ramp_analysis = analyze_per_ramp_fits(
            processed=processed,
            selected_peaks=selected_peaks_for_ramps,
            cfg=fit_cfg,
            common_x=avg["common_x"],
        )
    _lap("per-ramp fits")

    # ----------------------------------------------------------------
    # Falling ramp processing (validation mode, use_falling_ramps=True)
    # ----------------------------------------------------------------
    falling_processed = []
    avg_falling = None
    multi_fit_falling = None
    ramp_comparison_plot = None
    avg_combined = None
    multi_fit_combined = None
    falling_offsets_mhz = []

    if cfg.use_falling_ramps:
        try:
            falling_slices = detect_falling_ramps(prepared["t"], prepared["scan"], cfg)
        except RuntimeError as exc:
            if cfg.debug:
                print(f"[falling ramps] detection failed: {exc}")
            falling_slices = []

        if cfg.debug:
            print(f"[falling ramps] {len(falling_slices)} falling ramps detected")

        for r in falling_slices:
            try:
                falling_processed.append(process_rising_ramp(
                    prepared["t"][r],
                    prepared["scan"][r],
                    prepared["freq_interp_falling"][r],
                    prepared["probe"][r],
                    prepared["ref"][r],
                    edge_fraction=cfg.edge_fraction,
                    eps=cfg.eps,
                    max_shift=cfg.max_shift,
                ))
            except RuntimeError:
                continue

        if len(falling_processed) >= 2:

            # --- Raw falling average (for comparison plot) ---
            avg_falling = average_ramps_to_common_axis(falling_processed, cfg)
            avg_falling = correct_average_baseline(avg_falling, peak_positions_thz, cfg)

            _, strong_peaks_falling, _ = detect_strong_peaks(avg_falling, cfg)
            fit_input_falling = sorted(strong_peaks_falling, key=lambda p: p["x"])

            fall_fit_cfg = AnalysisConfig(**asdict(fit_cfg))
            if fall_fit_cfg.multi_fit_peak_indices is None:
                fall_fit_cfg.multi_fit_peak_indices = list(range(len(fit_input_falling)))
            else:
                valid_indices = [i for i in fall_fit_cfg.multi_fit_peak_indices
                                 if i < len(fit_input_falling)]
                fall_fit_cfg.multi_fit_peak_indices = valid_indices

            try:
                multi_fit_falling = fit_selected_peaks_global(avg_falling, fit_input_falling, fall_fit_cfg)
            except Exception as exc:
                print(f"[falling ramps] global fit failed: {exc}")
                multi_fit_falling = None

            # --- Frequency-corrected falling ramps + combined average ---
            if per_ramp_analysis is not None and multi_fit is not None and multi_fit.get("fit_results"):

                # Anchor peak: use the first (strongest) peak from rising fit
                rising_anchor = selected_peaks_for_ramps[0]
                rising_anchor_label = rising_anchor["peak_label"]
                rising_anchor_centre_thz = rising_anchor["x"]  # target position (rising)

                # Build per-ramp fit peaks for falling ramps:
                # - positions from multi_fit_falling (correct location in falling data)
                # - widths from rising average fit (better constrained)
                if multi_fit_falling is not None and multi_fit_falling.get("fit_results"):
                    rising_widths = {
                        r["isotope_assignment"]["expected_label"]: {
                            "sigma_thz": r["sigma_thz"],
                            "gamma_thz": r["gamma_thz"],
                            "peak_label": r["peak_label"],
                        }
                        for r in multi_fit["fit_results"]
                        if r.get("isotope_assignment") and r["isotope_assignment"].get("expected_label")
                    }
                    selected_peaks_for_falling = []
                    for r in multi_fit_falling["fit_results"]:
                        iso = r.get("isotope_assignment", {}).get("expected_label")
                        widths = rising_widths.get(iso)
                        if widths is not None:
                            selected_peaks_for_falling.append({
                                "peak_label": widths["peak_label"],
                                "x": r["centre_thz"],          # falling position for fit guess
                                "sigma_thz": widths["sigma_thz"],
                                "gamma_thz": widths["gamma_thz"],
                            })
                    # Sort by peak_label to match selected_peaks_for_ramps order
                    selected_peaks_for_falling = sorted(
                        selected_peaks_for_falling, key=lambda p: p["peak_label"]
                    )
                    if len(selected_peaks_for_falling) == 0:
                        selected_peaks_for_falling = selected_peaks_for_ramps
                else:
                    selected_peaks_for_falling = selected_peaks_for_ramps

                corrected_falling, falling_offsets_mhz = correct_falling_ramp_freq_axes(
                    falling_processed=falling_processed,
                    selected_peaks_for_ramps=selected_peaks_for_falling,
                    cfg=fit_cfg,
                    common_x=avg["common_x"],
                    target_centre_thz=rising_anchor_centre_thz,
                )

                if len(corrected_falling) >= 2:
                    # Pool rising + corrected falling ramps
                    combined_processed = processed + corrected_falling
                    avg_combined = average_ramps_to_common_axis(combined_processed, cfg)
                    avg_combined = correct_average_baseline(avg_combined, peak_positions_thz, cfg)

                    _, strong_peaks_combined, _ = detect_strong_peaks(avg_combined, cfg)
                    fit_input_combined = sorted(strong_peaks_combined, key=lambda p: p["x"])

                    combined_fit_cfg = AnalysisConfig(**asdict(fit_cfg))
                    if combined_fit_cfg.multi_fit_peak_indices is None:
                        combined_fit_cfg.multi_fit_peak_indices = list(range(len(fit_input_combined)))
                    else:
                        valid_indices = [i for i in combined_fit_cfg.multi_fit_peak_indices
                                         if i < len(fit_input_combined)]
                        combined_fit_cfg.multi_fit_peak_indices = valid_indices

                    try:
                        multi_fit_combined = fit_selected_peaks_global(avg_combined, fit_input_combined, combined_fit_cfg)
                        if cfg.debug:
                            print(f"[combined] {len(combined_processed)} total ramps "
                                  f"({len(processed)} rising + {len(corrected_falling)} corrected falling)")
                            if multi_fit_combined and multi_fit_combined.get("fit_results"):
                                for r in multi_fit_combined["fit_results"]:
                                    a = r["isotope_assignment"]
                                    print(f"[combined] {r['peak_label']} -> {a['expected_label']} | "
                                          f"centre = {r['centre_thz']:.9f} THz | "
                                          f"FWHM = {r['fwhm_mhz']:.3f} MHz")
                    except Exception as exc:
                        if cfg.debug:
                            print(f"[combined] global fit failed: {exc}")
                        multi_fit_combined = None
                else:
                    if cfg.debug:
                        print(f"[combined] not enough corrected falling ramps ({len(corrected_falling)}), skipping combined average")

        else:
            if cfg.debug:
                print(f"[falling ramps] only {len(falling_processed)} valid ramps found, skipping averaging")

    plot_paths = []
    if cfg.save_plots:
        plot_paths = make_summary_plots(
            avg,
            strong_peaks,
            isotope_assignment,
            local_peaks,
            single_fit,
            multi_fit,
            peak_diag,
            cfg,
            outdir,
        )

        isotope_plot = make_isotope_assignment_plot(
            avg=avg,
            strong_peaks=strong_peaks,
            isotope_assignment=isotope_assignment,
            local_peaks=local_peaks,
            multi_fit=multi_fit,
            diagnostics=peak_diag,
            cfg=cfg,
            outdir=outdir,
        )
        plot_paths.append(isotope_plot)

        if cfg.use_falling_ramps and avg_falling is not None:
            ramp_comparison_plot = make_ramp_comparison_plot(
                avg_rising=avg,
                avg_falling=avg_falling,
                multi_fit_rising=multi_fit,
                multi_fit_falling=multi_fit_falling,
                cfg=cfg,
                outdir=outdir,
                output_filename="ramp_comparison.png",
            )
            plot_paths.append(ramp_comparison_plot)

        if cfg.use_falling_ramps and avg_combined is not None:
            combined_comparison_plot = make_ramp_comparison_plot(
                avg_rising=avg,
                avg_falling=avg_combined,
                multi_fit_rising=multi_fit,
                multi_fit_falling=multi_fit_combined,
                cfg=cfg,
                outdir=outdir,
                output_filename="combined_comparison.png",
            )
            plot_paths.append(combined_comparison_plot)
    _lap("plots")

    strong_csv, fit_csv = save_csvs(strong_peaks, multi_fit, isotope_assignment, cfg, outdir)
    per_ramp_csv, per_ramp_summary_csv = save_per_ramp_csvs(per_ramp_analysis, cfg, outdir)

    # Pass per-ramp jitter into cfg so build_physics_results can correct sigma
    if per_ramp_analysis is not None:
        summary_df = per_ramp_analysis.get("per_ramp_summary_df")
        if summary_df is not None and not summary_df.empty and "centre_std_mhz" in summary_df.columns:
            # Use jitter from the primary isotope (Dy164) only.
            # Averaging across all peaks is unreliable because weak peaks like Dy161
            # have very large centre_std_mhz that would over-correct the jitter subtraction.
            primary_iso = "Dy164"
            iso_col = next((c for c in ["assigned_isotope", "isotope"] if c in summary_df.columns), None)
            if iso_col is not None:
                primary_row = summary_df[summary_df[iso_col].str.strip() == primary_iso]
            else:
                primary_row = pd.DataFrame()

            if not primary_row.empty:
                mean_jitter_mhz = float(primary_row["centre_std_mhz"].iloc[0])
            else:
                # Fall back to median across strong peaks only (exclude outliers > 20 MHz)
                vals = summary_df["centre_std_mhz"].dropna()
                vals = vals[vals < 20.0]
                mean_jitter_mhz = float(vals.median()) if len(vals) > 0 else 0.0
            cfg._per_ramp_jitter_sigma_thz = mean_jitter_mhz * 1e-6
        else:
            cfg._per_ramp_jitter_sigma_thz = None
    else:
        cfg._per_ramp_jitter_sigma_thz = None

    physics_df = build_physics_results(avg, multi_fit, cfg)
    physics_csv = save_physics_csv(physics_df, cfg, outdir)

    summary_txt = save_summary_text(
        processed,
        strong_peaks,
        local_peaks,
        isotope_assignment,
        single_fit,
        multi_fit,
        per_ramp_analysis,
        physics_df,
        cfg,
        outdir,
    )
    _lap("save outputs")
    _lap("TOTAL")

    return {
        "processed": processed,
        "avg": avg,
        "candidates": candidates,
        "strong_peaks": strong_peaks,
        "local_peaks": local_peaks,
        "best_spacing_match": best_spacing_match,
        "isotope_assignment": isotope_assignment,
        "single_fit": single_fit,
        "multi_fit": multi_fit,
        "per_ramp_analysis": per_ramp_analysis,
        "falling_processed": falling_processed,
        "avg_falling": avg_falling,
        "multi_fit_falling": multi_fit_falling,
        "falling_offsets_mhz": falling_offsets_mhz,
        "avg_combined": avg_combined,
        "multi_fit_combined": multi_fit_combined,
        "ramp_comparison_plot": ramp_comparison_plot,
        "plot_paths": plot_paths,
        "summary_txt": summary_txt,
        "strong_csv": strong_csv,
        "fit_csv": fit_csv,
        "per_ramp_csv": per_ramp_csv,
        "per_ramp_summary_csv": per_ramp_summary_csv,
        "physics_df": physics_df,
        "physics_csv": physics_csv,
    }


def config_from_json(path: str | Path) -> AnalysisConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return AnalysisConfig(**data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lean absorption scan analysis")
    parser.add_argument("--scope", required=True, help="Path to combined scope NPZ")
    parser.add_argument("--wavemeter", required=True, help="Path to wavemeter CSV")
    parser.add_argument("--outdir", required=True, help="Directory for outputs")
    parser.add_argument("--config-json", default=None, help="Optional JSON config file")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.config_json:
        cfg = config_from_json(args.config_json)
        cfg.scope_file = args.scope
        cfg.wavemeter_file = args.wavemeter
        cfg.output_dir = args.outdir
        if args.debug:
            cfg.debug = True
    else:
        cfg = AnalysisConfig(scope_file=args.scope, wavemeter_file=args.wavemeter, output_dir=args.outdir, debug=args.debug)

    result = analyze(cfg)

    print(f"Valid rising ramps: {len(result['processed'])}")
    print(f"Strong peaks found: {len(result['strong_peaks'])}")
    if result["isotope_assignment"]:
        for det_label, info in result["isotope_assignment"].items():
            print(f"{det_label} -> {info['expected_label']} | {info['detected_x_thz']:.9f} THz | offset = {info['offset_mhz']:.3f} MHz")
    if result["multi_fit"] and result["multi_fit"]["fit_results"]:
        print(f"Fit type: {result['multi_fit'].get('fit_type', 'unknown')}")
        for r in result["multi_fit"]["fit_results"]:
            a = r["isotope_assignment"]
            print(f"{r['peak_label']} fit -> {a['expected_label']} | centre = {r['centre_thz']:.9f} THz | FWHM = {r['fwhm_mhz']:.3f} MHz")
        print(f"Residual RMS: {result['multi_fit']['resid_rms']:.6f}")
    print(f"Summary saved: {result['summary_txt']}")
    if result["strong_csv"] is not None:
        print(f"Strong peaks CSV: {result['strong_csv']}")
    print(f"Fit results CSV: {result['fit_csv']}")
    if result["physics_csv"] is not None:
        print(f"Physics results CSV: {result['physics_csv']}")


if __name__ == "__main__":
    main()