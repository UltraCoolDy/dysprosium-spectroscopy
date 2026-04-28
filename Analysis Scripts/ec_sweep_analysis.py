"""
ec_sweep_analysis.py
====================
Sweep 1 analysis: EC temperature scan with fixed HL temperature.

Loads results files for each included dataset, applies quality flags
(warnings only), averages across repeats per temperature, and produces:
  - ec_sweep_summary.csv     : one row per EC temp with mean +/- std
  - ec_sweep_od_plot.png     : OD (flux proxy) vs EC temperature
  - ec_sweep_fwhm_plot.png   : Gaussian & Voigt FWHM vs EC temperature with sqrt(T) overlay
  - ec_sweep_overview.png    : combined 2x2 panel

Usage:
  1. Populate EC_SWEEP_DATASETS below with your dataset timestamps.
  2. Set ACQ_FOLDER and OUTPUT_DIR paths.
  3. Run: python ec_sweep_analysis.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ============================================================
# USER SETTINGS — edit these
# ============================================================

ACQ_FOLDER = Path(r"C:\Users\dysprosium\labscript-suite\userlib\labscriptlib\quantum_gas_microscope\Dysprosium\Spectroscopy\data_acq")

OUTPUT_DIR = Path(r"C:\Users\dysprosium\labscript-suite\userlib\labscriptlib\quantum_gas_microscope\Dysprosium\Spectroscopy\ec_sweep_analysis")

# Include list: EC temp (°C) -> list of timestamp prefixes (YYYYMMDD_HHMMSS)
# Leave empty dict {} on first run — script will print all discovered datasets
# grouped by EC temp to help you populate this.
EC_SWEEP_DATASETS: Dict[int, List[str]] = {
    1000: ["20260428_134529", "20260428_135246", "20260428_142406"],
    1025: ["20260428_151704", "20260428_151940", "20260428_152721", "20260428_153013"],
    1050: ["20260427_155452", "20260428_115600", "20260428_115809", "20260428_120116", "20260428_120553"],
    1075: ["20260427_165727", "20260427_165929", "20260427_170123", "20260427_170523"],
    1100: ["20260428_095722", "20260428_095933", "20260428_100758", "20260428_101110"],
}

# Discovery mode date filter — only used when EC_SWEEP_DATASETS is empty.
# List dates as "YYYYMMDD" strings to limit the discovery report to specific days.
# Leave as [] to scan all dates.
DISCOVERY_DATES: List[str] = [
    "20260427",
    "20260428",
]

# Fixed HL temperature for this sweep (used for labelling only)
HL_TEMP_FIXED = 1100

# Temperature aliases — map EC temps that are nominally the same to a canonical value.
# e.g. 1001 -> 1000 means any folder with EC1001 is treated as EC1000.
TEMP_ALIASES: Dict[int, int] = {
    1001: 1000,
    1024: 1025,
    1051: 1050,
    1074: 1075,
}

# Quality flag thresholds — datasets outside these get a WARNING but are
# still included unless you remove them from EC_SWEEP_DATASETS above
JITTER_MAX_MHZ     = 12.0
RESID_RMS_MAX      = 0.0015
FWHM_MAX_MHZ       = 60.0
N_RAMPS_MIN        = 15

# Primary isotope to analyse
PRIMARY_ISOTOPE = "Dy164"

# ============================================================
# FOLDER WALKING
# ============================================================

def find_all_datasets(acq_folder: Path, date_filter: List[str] = []) -> List[Dict]:
    """
    Walk data_acq/YYYY-MM/DD/dataset_folder/ and return metadata for
    every dataset that has a fit_results.csv present.
    """
    datasets = []
    month_re = re.compile(r"^\d{4}-\d{2}$")
    day_re   = re.compile(r"^\d{2}$")
    stem_re  = re.compile(r"^(\d{8}_\d{6})_EC(\d+)_HL(\d+)")

    for month_dir in sorted(acq_folder.iterdir()):
        if not month_dir.is_dir() or not month_re.match(month_dir.name):
            continue
        for day_dir in sorted(month_dir.iterdir()):
            if not day_dir.is_dir() or not day_re.match(day_dir.name):
                continue
            for ds_dir in sorted(day_dir.iterdir()):
                if not ds_dir.is_dir():
                    continue
                # Apply date filter if specified (discovery mode only)
                if date_filter:
                    ds_date = ds_dir.name[:8]
                    if ds_date not in date_filter:
                        continue
                m = stem_re.match(ds_dir.name)
                if not m:
                    continue
                timestamp = m.group(1)
                ec_temp   = int(m.group(2))
                hl_temp   = int(m.group(3))

                fit_csv          = ds_dir / f"{ds_dir.name}_fit_results.csv"
                physics_csv      = ds_dir / f"{ds_dir.name}_physics_results.csv"
                summary_txt      = ds_dir / f"{ds_dir.name}_summary.txt"
                ramp_summary_csv = ds_dir / f"{ds_dir.name}_per_ramp_fit_summary.csv"

                if not fit_csv.exists():
                    continue  # not yet analysed

                datasets.append({
                    "timestamp":        timestamp,
                    "stem":             ds_dir.name,
                    "folder":           ds_dir,
                    "ec_temp":          ec_temp,
                    "hl_temp":          hl_temp,
                    "fit_csv":          fit_csv,
                    "physics_csv":      physics_csv      if physics_csv.exists()      else None,
                    "summary_txt":      summary_txt      if summary_txt.exists()      else None,
                    "ramp_summary_csv": ramp_summary_csv if ramp_summary_csv.exists() else None,
                })

    return datasets


def discovery_report(datasets: List[Dict]) -> None:
    """Print all discovered datasets grouped by EC temp — helps populate include list."""
    from collections import defaultdict
    by_ec: Dict[int, List[Dict]] = defaultdict(list)
    for ds in datasets:
        by_ec[ds["ec_temp"]].append(ds)

    print("\n" + "=" * 70)
    print("DISCOVERY MODE — EC_SWEEP_DATASETS is empty")
    if DISCOVERY_DATES:
        print(f"Date filter active: {', '.join(DISCOVERY_DATES)}")
    print("All analysed datasets found, grouped by EC temperature:")
    print("=" * 70)
    for ec in sorted(by_ec):
        print(f"\n  EC = {ec}°C  (HL temps seen: {sorted({d['hl_temp'] for d in by_ec[ec]})})")
        for ds in sorted(by_ec[ec], key=lambda d: d["timestamp"]):
            print(f"    \"{ds['timestamp']}\",  # {ds['stem']}")
    print("\nCopy the timestamps you want into EC_SWEEP_DATASETS at the top of the script.")
    print("=" * 70 + "\n")

# ============================================================
# LOADING FIT RESULTS
# ============================================================

def load_fit_results(ds: Dict) -> Optional[Dict]:
    """
    Load all results files for a dataset and return a flat dict of key quantities.

    Sources:
      fit_results.csv          -> Voigt components: sigma_thz, gamma_thz, amplitude, fwhm_mhz
      physics_results.csv      -> peak_od, number_density_m3, beam_diameter_mm
      per_ramp_fit_summary.csv -> n_ramps, centre_std_mhz (jitter)
      summary.txt              -> Residual RMS
    """
    try:
        fit_df = pd.read_csv(ds["fit_csv"])
    except Exception as e:
        print(f"  WARNING: could not read {ds['fit_csv'].name}: {e}")
        return None

    # Apply temperature alias
    ec_temp_canonical = TEMP_ALIASES.get(ds["ec_temp"], ds["ec_temp"])

    result = {
        "stem":      ds["stem"],
        "timestamp": ds["timestamp"],
        "ec_temp":   ec_temp_canonical,
        "hl_temp":   ds["hl_temp"],
    }

    # ------------------------------------------------------------------
    # fit_results.csv
    # Columns: assigned_isotope, amplitude, fwhm_mhz, sigma_thz, gamma_thz
    # Gaussian FWHM  = 2*sqrt(2*ln2) * sigma_thz * 1e6  (THz -> MHz)
    # Lorentzian FWHM = 2 * gamma_thz * 1e6             (THz -> MHz)
    # ------------------------------------------------------------------
    SIGMA_TO_FWHM_G = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ~2.355
    THZ_TO_MHZ = 1e6

    iso_col = next((c for c in ["assigned_isotope", "isotope"] if c in fit_df.columns), None)
    if iso_col is None:
        print(f"  WARNING: no isotope column found in {ds['fit_csv'].name}")
        return None

    row = fit_df[fit_df[iso_col].str.strip() == PRIMARY_ISOTOPE]
    if row.empty:
        print(f"  WARNING: {PRIMARY_ISOTOPE} not found in {ds['fit_csv'].name}")
        result["od"]     = np.nan
        result["fwhm_g"] = np.nan
        result["fwhm_l"] = np.nan
        result["fwhm_v"] = np.nan
    else:
        r = row.iloc[0]
        sigma_thz = float(r.get("sigma_thz", np.nan))
        gamma_thz = float(r.get("gamma_thz", np.nan))
        result["od"]     = float(r.get("amplitude", np.nan))
        result["fwhm_g"] = SIGMA_TO_FWHM_G * sigma_thz * THZ_TO_MHZ if not np.isnan(sigma_thz) else np.nan
        result["fwhm_l"] = 2.0 * gamma_thz * THZ_TO_MHZ             if not np.isnan(gamma_thz) else np.nan
        result["fwhm_v"] = float(r.get("fwhm_mhz", np.nan))

    # ------------------------------------------------------------------
    # physics_results.csv
    # Columns: assigned_isotope, peak_od, number_density_m3, beam_diameter_mm,
    #          voigt_fwhm_mhz, velocity_sigma_ms, beam_speed_ms
    # ------------------------------------------------------------------
    result["density"]   = np.nan
    result["beam_diam"] = np.nan
    result["v_spread"]  = np.nan

    if ds.get("physics_csv"):
        try:
            phys_df = pd.read_csv(ds["physics_csv"])
            phys_iso_col = next((c for c in ["assigned_isotope", "isotope"] if c in phys_df.columns), None)
            if phys_iso_col:
                prow = phys_df[phys_df[phys_iso_col].str.strip() == PRIMARY_ISOTOPE]
                if not prow.empty:
                    pr = prow.iloc[0]
                    # Override OD with the more physical peak_od value
                    result["od"]        = float(pr.get("peak_od",           result["od"]))
                    result["density"]   = float(pr.get("number_density_m3", np.nan))
                    result["beam_diam"] = float(pr.get("beam_diameter_mm",  np.nan))
                    result["v_spread"]  = float(pr.get("velocity_sigma_ms", np.nan))
                    result["fwhm_v"]    = float(pr.get("voigt_fwhm_mhz",   result["fwhm_v"]))
        except Exception as e:
            print(f"  WARNING: could not read physics_results: {e}")

    # ------------------------------------------------------------------
    # per_ramp_fit_summary.csv
    # Columns: assigned_isotope, n_ramps, centre_std_mhz
    # Use centre_std_mhz of PRIMARY_ISOTOPE as shot-to-shot jitter
    # ------------------------------------------------------------------
    result["n_ramps"]    = 0
    result["jitter_mhz"] = np.nan

    if ds.get("ramp_summary_csv"):
        try:
            ramp_df = pd.read_csv(ds["ramp_summary_csv"])
            ramp_iso_col = next((c for c in ["assigned_isotope", "isotope"] if c in ramp_df.columns), None)
            if ramp_iso_col:
                prow = ramp_df[ramp_df[ramp_iso_col].str.strip() == PRIMARY_ISOTOPE]
                if not prow.empty:
                    pr = prow.iloc[0]
                    result["n_ramps"]    = int(pr.get("n_ramps", 0))
                    result["jitter_mhz"] = float(pr.get("centre_std_mhz", np.nan))
        except Exception as e:
            print(f"  WARNING: could not read ramp_summary: {e}")

    # ------------------------------------------------------------------
    # summary.txt — residual RMS
    # ------------------------------------------------------------------
    result["resid_rms"] = np.nan
    if ds.get("summary_txt"):
        try:
            text = ds["summary_txt"].read_text()
            m = re.search(r"Residual RMS\s*=\s*([\d.eE+\-]+)", text)
            if m:
                result["resid_rms"] = float(m.group(1))
        except Exception:
            pass

    return result

# ============================================================
# QUALITY FLAGS
# ============================================================

def quality_check(result: Dict) -> Tuple[bool, List[str]]:
    """Return (pass, [list of warning strings])."""
    warnings = []
    j = result.get("jitter_mhz", np.nan)
    r = result.get("resid_rms",  np.nan)
    n = result.get("n_ramps",    0)
    f = result.get("fwhm_g",     np.nan)

    if not np.isnan(j) and j > JITTER_MAX_MHZ:
        warnings.append(f"jitter={j:.1f}MHz > {JITTER_MAX_MHZ}MHz")
    if not np.isnan(r) and r > RESID_RMS_MAX:
        warnings.append(f"resid_rms={r:.5f} > {RESID_RMS_MAX}")
    if n > 0 and n < N_RAMPS_MIN:
        warnings.append(f"n_ramps={n} < {N_RAMPS_MIN}")
    if not np.isnan(f) and f > FWHM_MAX_MHZ:
        warnings.append(f"FWHM={f:.1f}MHz > {FWHM_MAX_MHZ}MHz")

    return len(warnings) == 0, warnings

# ============================================================
# AVERAGING PER TEMPERATURE POINT
# ============================================================

def average_by_temp(results: List[Dict]) -> pd.DataFrame:
    """Average all key quantities across datasets at the same EC temp."""
    quantities = ["od", "fwhm_g", "fwhm_l", "fwhm_v",
                  "density", "beam_diam", "v_spread",
                  "jitter_mhz", "n_ramps", "resid_rms"]

    df = pd.DataFrame(results)
    rows = []

    for ec_temp in sorted(df["ec_temp"].unique()):
        grp = df[df["ec_temp"] == ec_temp]
        row = {"ec_temp": ec_temp, "n_datasets": len(grp)}
        for q in quantities:
            if q not in grp.columns:
                row[f"{q}_mean"] = np.nan
                row[f"{q}_std"]  = np.nan
                continue
            vals = grp[q].dropna().values
            row[f"{q}_mean"] = np.nanmean(vals) if len(vals) > 0 else np.nan
            row[f"{q}_std"]  = np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows).sort_values("ec_temp").reset_index(drop=True)

# ============================================================
# SQRT(T) REFERENCE CURVE
# ============================================================

def sqrt_t_reference(temps_c: np.ndarray, fwhm_ref: float, t_ref_c: float) -> np.ndarray:
    """FWHM ∝ sqrt(T), anchored to (t_ref_c, fwhm_ref). Temps in Celsius."""
    t_k     = temps_c + 273.15
    t_ref_k = t_ref_c + 273.15
    return fwhm_ref * np.sqrt(t_k / t_ref_k)

# ============================================================
# PLOTTING

def set_ec_xticks(ax, temps: np.ndarray, step: int = 25) -> None:
    """Set x-axis ticks at fixed intervals aligned to data points."""
    lo = int(np.floor(temps.min() / step) * step)
    hi = int(np.ceil(temps.max()  / step) * step)
    ax.set_xticks(range(lo, hi + step, step))
    ax.xaxis.set_minor_locator(MultipleLocator(step))

# ============================================================

STYLE = {
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "font.family":       "sans-serif",
}


def plot_od(summary: pd.DataFrame, out_path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        temps = summary["ec_temp"].values
        od    = summary["od_mean"].values
        od_e  = summary["od_std"].values

        ax.errorbar(temps, od, yerr=od_e, fmt="o-", capsize=4,
                    color="#1f77b4", linewidth=1.5, label=PRIMARY_ISOTOPE)
        ax.set_xlabel(f"EC Temperature (°C)  [HL fixed = {HL_TEMP_FIXED}°C]")
        ax.set_ylabel("Peak Optical Depth")
        ax.set_title(f"Flux Proxy (OD) vs EC Temperature — {PRIMARY_ISOTOPE}")
        set_ec_xticks(ax, temps)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle=":")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    print(f"OD plot saved: {out_path}")


def plot_fwhm(summary: pd.DataFrame, out_path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        temps    = summary["ec_temp"].values
        fwhm_g   = summary["fwhm_g_mean"].values
        fwhm_g_e = summary["fwhm_g_std"].values
        fwhm_v   = summary["fwhm_v_mean"].values
        fwhm_v_e = summary["fwhm_v_std"].values

        # Gaussian FWHM + sqrt(T) reference
        ax = axes[0]
        ax.errorbar(temps, fwhm_g, yerr=fwhm_g_e, fmt="o-", capsize=4,
                    color="#1f77b4", linewidth=1.5, label=f"{PRIMARY_ISOTOPE} data")
        valid = ~np.isnan(fwhm_g)
        if valid.sum() >= 2:
            mid_idx  = valid.nonzero()[0][len(valid.nonzero()[0]) // 2]
            t_smooth = np.linspace(temps[valid].min(), temps[valid].max(), 200)
            sqt      = sqrt_t_reference(t_smooth, fwhm_g[mid_idx], temps[mid_idx])
            ax.plot(t_smooth, sqt, "k--", linewidth=1.2, alpha=0.6,
                    label=r"$\sqrt{T}$ scaling")
        ax.set_xlabel(f"EC Temperature (°C)  [HL = {HL_TEMP_FIXED}°C]")
        ax.set_ylabel("Gaussian FWHM (MHz)")
        ax.set_title(f"Gaussian Linewidth vs EC Temp ({PRIMARY_ISOTOPE})")
        set_ec_xticks(ax, temps)
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle=":")

        # Voigt FWHM + sqrt(T) reference
        ax = axes[1]
        ax.errorbar(temps, fwhm_v, yerr=fwhm_v_e, fmt="o-", capsize=4,
                    color="#2ca02c", linewidth=1.5, label=f"{PRIMARY_ISOTOPE} data")
        valid_v = ~np.isnan(fwhm_v)
        if valid_v.sum() >= 2:
            mid_idx  = valid_v.nonzero()[0][len(valid_v.nonzero()[0]) // 2]
            t_smooth = np.linspace(temps[valid_v].min(), temps[valid_v].max(), 200)
            sqt_v    = sqrt_t_reference(t_smooth, fwhm_v[mid_idx], temps[mid_idx])
            ax.plot(t_smooth, sqt_v, "k--", linewidth=1.2, alpha=0.6,
                    label=r"$\sqrt{T}$ scaling")
        ax.set_xlabel(f"EC Temperature (°C)  [HL = {HL_TEMP_FIXED}°C]")
        ax.set_ylabel("Voigt FWHM (MHz)")
        ax.set_title(f"Total Voigt Linewidth vs EC Temp ({PRIMARY_ISOTOPE})")
        set_ec_xticks(ax, temps)
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle=":")

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    print(f"FWHM plot saved: {out_path}")


def plot_overview(summary: pd.DataFrame, out_path: Path) -> None:
    """2x2 overview panel: OD, density, Gaussian FWHM, Voigt FWHM."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        temps = summary["ec_temp"].values

        def _eb(ax, y_col, e_col, color, ylabel, title):
            y = summary[y_col].values
            e = summary[e_col].values
            ax.errorbar(temps, y, yerr=e, fmt="o-", capsize=4,
                        color=color, linewidth=1.5, label=PRIMARY_ISOTOPE)
            ax.set_xlabel("EC Temperature (°C)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            set_ec_xticks(ax, temps)
            ax.grid(True, alpha=0.3, linestyle=":")
            ax.legend(fontsize=8)

        _eb(axes[0, 0], "od_mean",      "od_std",
            "#1f77b4", "Peak Optical Depth",
            f"Flux Proxy (OD) — {PRIMARY_ISOTOPE}")

        _eb(axes[0, 1], "v_spread_mean", "v_spread_std",
            "#9467bd", "Transverse Velocity Spread (m/s)",
            f"Transverse Velocity Spread — {PRIMARY_ISOTOPE}")

        # Gaussian FWHM with sqrt(T) overlay
        ax = axes[1, 0]
        fwhm_g   = summary["fwhm_g_mean"].values
        fwhm_g_e = summary["fwhm_g_std"].values
        ax.errorbar(temps, fwhm_g, yerr=fwhm_g_e, fmt="o-", capsize=4,
                    color="#1f77b4", linewidth=1.5, label=f"{PRIMARY_ISOTOPE} data")
        valid = ~np.isnan(fwhm_g)
        if valid.sum() >= 2:
            mid_idx  = valid.nonzero()[0][len(valid.nonzero()[0]) // 2]
            t_smooth = np.linspace(temps[valid].min(), temps[valid].max(), 200)
            sqt      = sqrt_t_reference(t_smooth, fwhm_g[mid_idx], temps[mid_idx])
            ax.plot(t_smooth, sqt, "k--", linewidth=1.2, alpha=0.6,
                    label=r"$\sqrt{T}$ scaling")
        ax.set_xlabel("EC Temperature (°C)")
        ax.set_ylabel("Gaussian FWHM (MHz)")
        ax.set_title(f"Gaussian Linewidth — {PRIMARY_ISOTOPE}")
        set_ec_xticks(ax, temps)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        fwhm_v_ov   = summary["fwhm_v_mean"].values
        fwhm_v_ov_e = summary["fwhm_v_std"].values
        ax.errorbar(temps, fwhm_v_ov, yerr=fwhm_v_ov_e, fmt="o-", capsize=4,
                    color="#2ca02c", linewidth=1.5, label=f"{PRIMARY_ISOTOPE} data")
        valid_v = ~np.isnan(fwhm_v_ov)
        if valid_v.sum() >= 2:
            mid_idx  = valid_v.nonzero()[0][len(valid_v.nonzero()[0]) // 2]
            t_smooth = np.linspace(temps[valid_v].min(), temps[valid_v].max(), 200)
            sqt_v    = sqrt_t_reference(t_smooth, fwhm_v_ov[mid_idx], temps[mid_idx])
            ax.plot(t_smooth, sqt_v, "k--", linewidth=1.2, alpha=0.6,
                    label=r"$\sqrt{T}$ scaling")
        ax.set_xlabel("EC Temperature (°C)")
        ax.set_ylabel("Voigt FWHM (MHz)")
        ax.set_title(f"Total Voigt Linewidth — {PRIMARY_ISOTOPE}")
        set_ec_xticks(ax, temps)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend(fontsize=8)

        fig.suptitle(f"EC Temperature Sweep — HL Fixed = {HL_TEMP_FIXED}°C",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    print(f"Overview plot saved: {out_path}")

# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nScanning: {ACQ_FOLDER}")
    all_datasets = find_all_datasets(ACQ_FOLDER, date_filter=DISCOVERY_DATES)
    print(f"Total analysed datasets found: {len(all_datasets)}")

    # --- Discovery mode ---
    if not EC_SWEEP_DATASETS:
        discovery_report(all_datasets)
        return

    # --- Filter to include list ---
    include_flat = {ts for tss in EC_SWEEP_DATASETS.values() for ts in tss}
    selected = [ds for ds in all_datasets if ds["timestamp"] in include_flat]

    # Warn about any requested timestamps not found on disk
    found_ts = {ds["timestamp"] for ds in selected}
    for ts in include_flat:
        if ts not in found_ts:
            print(f"  WARNING: requested timestamp {ts} not found under {ACQ_FOLDER}")

    print(f"\nSelected {len(selected)} datasets across "
          f"{len(EC_SWEEP_DATASETS)} temperature points\n")

    # --- Load and quality-check ---
    results = []
    print(f"{'Dataset':<45} {'EC':>5} {'Jitter':>8} {'FWHM_G':>8} {'OD':>8}  Status")
    print("-" * 95)

    for ds in sorted(selected, key=lambda d: (d["ec_temp"], d["timestamp"])):
        r = load_fit_results(ds)
        if r is None:
            continue

        passed, warns = quality_check(r)
        status = "OK" if passed else f"WARN: {'; '.join(warns)}"

        j = r.get("jitter_mhz", float("nan"))
        f = r.get("fwhm_g",     float("nan"))
        o = r.get("od",         float("nan"))
        print(f"  {ds['stem']:<43} {r['ec_temp']:>5}°C "
              f"{j:>7.1f}MHz {f:>7.1f}MHz {o:>10.5f}  {status}")
        results.append(r)

    if not results:
        print("\nNo results loaded — check paths and dataset names.")
        return

    # --- Average by temperature ---
    summary = average_by_temp(results)

    print(f"\n{'EC Temp':>8}  {'N':>3}  {'OD_mean':>10}  {'OD_std':>9}  "
          f"{'FWHM_G_mean':>12}  {'FWHM_G_std':>11}")
    print("-" * 70)
    for _, row in summary.iterrows():
        print(f"  {int(row['ec_temp']):>6}°C  {int(row['n_datasets']):>3}  "
              f"{row['od_mean']:>10.5f}  {row['od_std']:>9.5f}  "
              f"{row['fwhm_g_mean']:>12.2f}  {row['fwhm_g_std']:>11.2f}")

    # --- Save CSV ---
    csv_path = OUTPUT_DIR / "ec_sweep_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved: {csv_path}")

    # --- Plots ---
    plot_od(summary,       OUTPUT_DIR / "ec_sweep_od_plot.png")
    plot_fwhm(summary,     OUTPUT_DIR / "ec_sweep_fwhm_plot.png")
    plot_overview(summary, OUTPUT_DIR / "ec_sweep_overview.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
