"""
combined_sweep_analysis.py
==========================
Combines EC and HL temperature sweep results to identify the optimal
oven operating point for the Dy atomic beam.

Loads ec_sweep_summary.csv and hl_sweep_summary.csv, produces:
  - combined_sweep_overview.png  : 2x2 panel (OD, v_spread, FWHM_G, FOM)
  - combined_sweep_analysis.txt  : printed operating point recommendations

Usage:
  python combined_sweep_analysis.py

Edit the paths in USER SETTINGS below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ============================================================
# USER SETTINGS
# ============================================================

EC_SUMMARY_CSV = Path(r"C:\Users\dysprosium\labscript-suite\userlib\labscriptlib\quantum_gas_microscope\Dysprosium\Spectroscopy\ec_sweep_analysis\ec_sweep_summary.csv")
HL_SUMMARY_CSV = Path(r"C:\Users\dysprosium\labscript-suite\userlib\labscriptlib\quantum_gas_microscope\Dysprosium\Spectroscopy\hl_sweep_analysis\hl_sweep_summary.csv")

OUTPUT_DIR = Path(r"C:\Users\dysprosium\labscript-suite\userlib\labscriptlib\quantum_gas_microscope\Dysprosium\Spectroscopy\combined_sweep_analysis")

# Fixed temperatures used in each sweep (for axis labels)
HL_FIXED_IN_EC_SWEEP = 1100   # HL temp held fixed during EC sweep
EC_FIXED_IN_HL_SWEEP = 1100   # EC temp held fixed during HL sweep

# Candidate operating point to highlight on plots
CANDIDATE_EC = 1100
CANDIDATE_HL = 1175

PRIMARY_ISOTOPE = "Dy164"

# ============================================================
# STYLE
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

EC_COLOR  = "#1f77b4"   # blue
HL_COLOR  = "#d62728"   # red
FOM_COLOR = "#7f7f7f"   # grey

# ============================================================
# FIGURE OF MERIT
# ============================================================

def figure_of_merit(od: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    FOM = OD / v_spread^2

    OD is proportional to beam flux through the probe.
    v_spread^2 is proportional to the transverse velocity variance,
    which sets the difficulty of laser cooling / MOT capture.
    Higher FOM = more flux per unit of transverse heating.
    """
    return od / v**2


# ============================================================
# ANALYSIS TEXT
# ============================================================

def print_analysis(ec: pd.DataFrame, hl: pd.DataFrame) -> None:
    """Print operating point analysis to stdout."""

    sep = "=" * 64

    print(f"\n{sep}")
    print("COMBINED EC / HL SWEEP — OPERATING POINT ANALYSIS")
    print(f"{sep}\n")

    # --- EC sweep summary ---
    print("EC Temperature Sweep  (HL fixed = {}°C)".format(HL_FIXED_IN_EC_SWEEP))
    print("-" * 44)
    ec_ref_od = ec["od_mean"].iloc[0]
    for _, row in ec.iterrows():
        gain = (row["od_mean"] / ec_ref_od - 1) * 100
        print(f"  EC={int(row['ec_temp']):4d}°C  OD={row['od_mean']:.5f}  "
              f"v={row['v_spread_mean']:5.2f} m/s  "
              f"flux gain vs EC{int(ec['ec_temp'].iloc[0])}: {gain:+.0f}%")

    # --- HL sweep summary ---
    print()
    print("HL Temperature Sweep  (EC fixed = {}°C)".format(EC_FIXED_IN_HL_SWEEP))
    print("-" * 44)
    # Reference: shared EC=1100/HL=1100 point
    ref_row = hl[hl["hl_temp"] == HL_FIXED_IN_EC_SWEEP]
    ref_od = ref_row["od_mean"].iloc[0] if not ref_row.empty else hl["od_mean"].iloc[0]
    ref_v  = ref_row["v_spread_mean"].iloc[0] if not ref_row.empty else hl["v_spread_mean"].iloc[0]
    ref_fom = figure_of_merit(np.array([ref_od]), np.array([ref_v]))[0]

    for _, row in hl.iterrows():
        flux_gain = (row["od_mean"] / ref_od - 1) * 100
        v_penalty = (row["v_spread_mean"] / ref_v - 1) * 100
        fom = figure_of_merit(np.array([row["od_mean"]]), np.array([row["v_spread_mean"]]))[0]
        fom_gain = (fom / ref_fom - 1) * 100
        marker = " <-- candidate" if int(row["hl_temp"]) == CANDIDATE_HL else ""
        print(f"  HL={int(row['hl_temp']):4d}°C  OD={row['od_mean']:.5f}  "
              f"v={row['v_spread_mean']:5.2f} m/s  "
              f"flux:{flux_gain:+.0f}%  v_penalty:{v_penalty:+.1f}%  "
              f"FOM:{fom_gain:+.0f}%{marker}")

    # --- Recommendation ---
    hl_fom = figure_of_merit(hl["od_mean"].values, hl["v_spread_mean"].values)
    best_idx = np.argmax(hl_fom)
    best_hl  = int(hl["hl_temp"].iloc[best_idx])

    print()
    print(f"{sep}")
    print("RECOMMENDATION")
    print(f"{sep}")
    print(f"  EC temperature : {CANDIDATE_EC}°C  (maximum flux in EC sweep)")
    print(f"  HL temperature : {CANDIDATE_HL}°C  (candidate operating point)")
    print()
    print(f"  Highest FOM point in HL sweep: HL = {best_hl}°C")
    if best_hl != CANDIDATE_HL:
        best_row = hl[hl["hl_temp"] == best_hl].iloc[0]
        cand_row = hl[hl["hl_temp"] == CANDIDATE_HL].iloc[0] if CANDIDATE_HL in hl["hl_temp"].values else None
        print(f"  HL={best_hl}: OD={best_row['od_mean']:.5f}  v={best_row['v_spread_mean']:.2f} m/s")
        if cand_row is not None:
            print(f"  HL={CANDIDATE_HL}: OD={cand_row['od_mean']:.5f}  v={cand_row['v_spread_mean']:.2f} m/s")
    print()

    # --- Shared calibration point check ---
    ec_shared = ec[ec["ec_temp"] == EC_FIXED_IN_HL_SWEEP]
    hl_shared = hl[hl["hl_temp"] == HL_FIXED_IN_EC_SWEEP]
    if not ec_shared.empty and not hl_shared.empty:
        od_ec = ec_shared["od_mean"].iloc[0]
        od_hl = hl_shared["od_mean"].iloc[0]
        discrepancy = abs(od_ec - od_hl) / max(od_ec, od_hl) * 100
        print(f"  Calibration check  EC={EC_FIXED_IN_HL_SWEEP}/HL={HL_FIXED_IN_EC_SWEEP} shared point:")
        print(f"    EC sweep OD = {od_ec:.5f}")
        print(f"    HL sweep OD = {od_hl:.5f}")
        print(f"    Discrepancy = {discrepancy:.1f}%  ({'OK' if discrepancy < 5 else 'WARNING: >5%'})")
    print(f"{sep}\n")


# ============================================================
# PLOTTING
# ============================================================

def set_xticks(ax, temps: np.ndarray, step: int = 25) -> None:
    lo = int(np.floor(temps.min() / step) * step)
    hi = int(np.ceil(temps.max()  / step) * step)
    ax.set_xticks(range(lo, hi + step, step))
    ax.xaxis.set_minor_locator(MultipleLocator(step))


def _mark_candidate(ax, x_val: float, color: str = "#ff7f0e") -> None:
    """Draw a vertical dashed line marking the candidate operating point."""
    ax.axvline(x_val, color=color, linewidth=1.2, linestyle="--", alpha=0.7, zorder=1)


def plot_combined(ec: pd.DataFrame, hl: pd.DataFrame, out_path: Path) -> None:
    """
    2x2 panel:
      [0,0] OD vs temperature (both sweeps)
      [0,1] Transverse velocity spread (both sweeps)
      [1,0] Gaussian FWHM (both sweeps)
      [1,1] Figure of merit OD/v^2 (HL sweep only, EC sweep has no useful variation)
    """
    ec_temps = ec["ec_temp"].values
    hl_temps = hl["hl_temp"].values

    ec_od   = ec["od_mean"].values;    ec_od_e  = ec["od_std"].values
    hl_od   = hl["od_mean"].values;    hl_od_e  = hl["od_std"].values

    ec_v    = ec["v_spread_mean"].values;  ec_v_e  = ec["v_spread_std"].values
    hl_v    = hl["v_spread_mean"].values;  hl_v_e  = hl["v_spread_std"].values

    ec_fg   = ec["fwhm_g_mean"].values;  ec_fg_e = ec["fwhm_g_std"].values
    hl_fg   = hl["fwhm_g_mean"].values;  hl_fg_e = hl["fwhm_g_std"].values

    hl_fom  = figure_of_merit(hl_od, hl_v)
    # FOM error via propagation: dFOM/FOM = sqrt((dOD/OD)^2 + (2*dv/v)^2)
    hl_fom_e = hl_fom * np.sqrt((hl_od_e / hl_od)**2 + (2 * hl_v_e / hl_v)**2)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # --- [0,0] OD ---
        ax = axes[0, 0]
        ax.errorbar(ec_temps, ec_od, yerr=ec_od_e, fmt="o-", capsize=4,
                    color=EC_COLOR, linewidth=1.5, label=f"EC sweep  (HL={HL_FIXED_IN_EC_SWEEP}°C fixed)")
        ax.errorbar(hl_temps, hl_od, yerr=hl_od_e, fmt="s-", capsize=4,
                    color=HL_COLOR, linewidth=1.5, label=f"HL sweep  (EC={EC_FIXED_IN_HL_SWEEP}°C fixed)")
        _mark_candidate(ax, CANDIDATE_EC, EC_COLOR)
        _mark_candidate(ax, CANDIDATE_HL, HL_COLOR)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Peak Optical Depth")
        ax.set_title(f"Flux Proxy (OD) — {PRIMARY_ISOTOPE}")
        set_xticks(ax, np.concatenate([ec_temps, hl_temps]))
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle=":")

        # --- [0,1] Transverse velocity ---
        ax = axes[0, 1]
        ax.errorbar(ec_temps, ec_v, yerr=ec_v_e, fmt="o-", capsize=4,
                    color=EC_COLOR, linewidth=1.5, label=f"EC sweep  (HL={HL_FIXED_IN_EC_SWEEP}°C fixed)")
        ax.errorbar(hl_temps, hl_v, yerr=hl_v_e, fmt="s-", capsize=4,
                    color=HL_COLOR, linewidth=1.5, label=f"HL sweep  (EC={EC_FIXED_IN_HL_SWEEP}°C fixed)")
        _mark_candidate(ax, CANDIDATE_EC, EC_COLOR)
        _mark_candidate(ax, CANDIDATE_HL, HL_COLOR)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Transverse Velocity Spread (m/s)")
        ax.set_title(f"Transverse Velocity — {PRIMARY_ISOTOPE}")
        set_xticks(ax, np.concatenate([ec_temps, hl_temps]))
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle=":")

        # --- [1,0] Gaussian FWHM ---
        ax = axes[1, 0]
        ax.errorbar(ec_temps, ec_fg, yerr=ec_fg_e, fmt="o-", capsize=4,
                    color=EC_COLOR, linewidth=1.5, label=f"EC sweep  (HL={HL_FIXED_IN_EC_SWEEP}°C fixed)")
        ax.errorbar(hl_temps, hl_fg, yerr=hl_fg_e, fmt="s-", capsize=4,
                    color=HL_COLOR, linewidth=1.5, label=f"HL sweep  (EC={EC_FIXED_IN_HL_SWEEP}°C fixed)")
        _mark_candidate(ax, CANDIDATE_EC, EC_COLOR)
        _mark_candidate(ax, CANDIDATE_HL, HL_COLOR)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Gaussian FWHM (MHz)")
        ax.set_title(f"Doppler Linewidth — {PRIMARY_ISOTOPE}")
        set_xticks(ax, np.concatenate([ec_temps, hl_temps]))
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle=":")

        # --- [1,1] FOM (HL sweep) ---
        ax = axes[1, 1]
        ax.errorbar(hl_temps, hl_fom * 1e6, yerr=hl_fom_e * 1e6, fmt="s-", capsize=4,
                    color=FOM_COLOR, linewidth=1.5, label=f"HL sweep  (EC={EC_FIXED_IN_HL_SWEEP}°C fixed)")
        _mark_candidate(ax, CANDIDATE_HL, HL_COLOR)
        # Mark best FOM point
        best_idx = np.argmax(hl_fom)
        ax.scatter([hl_temps[best_idx]], [hl_fom[best_idx] * 1e6],
                   s=80, zorder=5, color="#ff7f0e", marker="*",
                   label=f"Best FOM: HL={int(hl_temps[best_idx])}°C")
        ax.set_xlabel("HL Temperature (°C)")
        ax.set_ylabel(r"OD / $v^2$  ($\times 10^{-6}$ m$^{-2}$ s$^2$)")
        ax.set_title(f"Figure of Merit (OD/$v^2$) — HL Sweep")
        set_xticks(ax, hl_temps)
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle=":")

        # Candidate annotation
        fig.text(0.5, -0.01,
                 f"Dashed lines: candidate operating point  "
                 f"EC={CANDIDATE_EC}°C (blue) / HL={CANDIDATE_HL}°C (red)",
                 ha="center", fontsize=9, color="#555555")

        fig.suptitle(
            f"Combined EC + HL Sweep — {PRIMARY_ISOTOPE}   "
            f"[candidate: EC={CANDIDATE_EC}°C, HL={CANDIDATE_HL}°C]",
            fontsize=13, fontweight="bold", y=1.01
        )
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    print(f"Combined plot saved: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading EC sweep: {EC_SUMMARY_CSV}")
    ec = pd.read_csv(EC_SUMMARY_CSV).sort_values("ec_temp").reset_index(drop=True)

    print(f"Loading HL sweep: {HL_SUMMARY_CSV}")
    hl = pd.read_csv(HL_SUMMARY_CSV).sort_values("hl_temp").reset_index(drop=True)

    print_analysis(ec, hl)

    plot_combined(ec, hl, OUTPUT_DIR / "combined_sweep_overview.png")

    print("Done.")


if __name__ == "__main__":
    main()
