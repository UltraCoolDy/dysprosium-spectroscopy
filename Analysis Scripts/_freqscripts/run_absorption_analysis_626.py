from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from absorption_analysis import AnalysisConfig, analyze, config_from_json, default_expected_peaks

# ============================================================
# USER SETTINGS
# ============================================================
# Point this to your acquisition folder.
ACQ_FOLDER = Path(r"C:\Users\dysprosium\labscript-suite\userlib\labscriptlib\quantum_gas_microscope\Dysprosium\Spectroscopy\combined_acq_full")

# Optional config file. Set to None to use built-in defaults.
CONFIG_JSON: Optional[Path] = Path(__file__).resolve().parent / "absorption_config_626.json"

# Runner mode:
#   "menu"      -> show all datasets and let you choose one / many / all new
#   "all_new"   -> automatically run every dataset that has not yet been analysed
#   "one_latest"-> automatically run only the newest unanalyzed dataset
MODE = "menu"

# If True, outputs are written into the SAME folder as the input .npz and .csv files.
# Files are prefixed with the dataset stem, e.g. 20260326_101120_summary.txt
OUTPUT_TO_INPUT_FOLDER = True

# Optional debug override
DEBUG = False


# ============================================================
# DATASET DISCOVERY
# ============================================================
def load_cfg_template() -> Dict:
    if CONFIG_JSON is not None:
        cfg_path = Path(CONFIG_JSON)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    return {
        "ttl_threshold_v": 1.5,
        "bin_size": 10,
        "freq_clean_kernel": 21,
        "scan_smooth_window": 101,
        "coarse_block": 20,
        "min_ramp_points": 150,
        "rise_threshold_fraction": 0.30,
        "start_backtrack_blocks": 1,
        "end_trim_blocks": 1,
        "drop_edge_ramps": True,
        "edge_fraction": 0.12,
        "eps": 1e-12,
        "max_shift": 8,
        "display_savgol_window": 3,
        "display_savgol_poly": 1,
        "peak_detect_window": 31,
        "peak_detect_poly": 2,
        "peak_threshold_sigma": 0.5,
        "peak_min_region_points": 12,
        "peak_ratio_window": 12,
        "peak_min_area_fraction": 0.3,
        "peak_min_area_snr": 0.4,
        "peak_min_ratio_bump_snr": 0.3,
        "f_ref_thz": 355.802555,
        "expected_peak_positions_thz": default_expected_peaks(),
        "single_fit_peak_index": 0,
        "fit_half_width_thz": 0.00015,
        "fit_exclude_half_width_thz": 6e-5,
        "multi_fit_peak_indices": [0, 1],
        "multi_fit_margin_thz": 0.00008,
        "multi_fit_local_half_width_thz": 6e-5,
        "global_baseline_exclude_half_width_thz": 4e-5,
        "save_per_ramp_fits": True,
        "per_ramp_min_successful_peaks": 2,
        "save_plots": True,
        "show_plots": False,
        "debug": False,
    }

def get_dataset_prefix(name: str) -> str:
    parts = name.split("_")

    # New labelled format:
    # YYYYMMDD_HHMMSS_EC1020_HL1091_...
    # Example:
    # 20260324_100954_EC1020_HL1091_full_scope_all
    if len(parts) >= 4 and parts[2].startswith("EC") and parts[3].startswith("HL"):
        return f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"

    # Old format:
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"

    return name

def extract_temperatures_from_name(name: str):
    """
    Extract EC and HL temperatures from filename or stem.

    Expected format:
    YYYYMMDD_HHMMSS_EC1020_HL1091_...

    Example:
    20260324_100954_EC1020_HL1091_full_scope_all.npz
    """
    parts = name.split("_")

    if len(parts) >= 4 and parts[2].startswith("EC") and parts[3].startswith("HL"):
        try:
            ec_temp = float(parts[2][2:])
            hl_temp = float(parts[3][2:])
            return ec_temp, hl_temp
        except ValueError:
            pass

    return None, None

def organize_loose_datasets(folder: Path) -> None:
    folder = folder.expanduser().resolve()

    scope_files = sorted(folder.glob("*_full_scope_all.npz"))
    wavemeter_files = sorted(folder.glob("*_full_wavemeter.csv"))

    scope_by_prefix = {}
    for p in scope_files:
        prefix = get_dataset_prefix(p.stem)
        scope_by_prefix[prefix] = p

    wavemeter_by_prefix = {}
    for p in wavemeter_files:
        prefix = get_dataset_prefix(p.stem)
        wavemeter_by_prefix[prefix] = p

    for prefix in sorted(set(scope_by_prefix) & set(wavemeter_by_prefix)):
        scope_file = scope_by_prefix[prefix]
        wavemeter_file = wavemeter_by_prefix[prefix]

        dataset_dir = folder / prefix
        dataset_dir.mkdir(exist_ok=True)

        target_scope = dataset_dir / scope_file.name
        target_wavemeter = dataset_dir / wavemeter_file.name

        if scope_file.resolve() != target_scope.resolve():
            scope_file.rename(target_scope)

        if wavemeter_file.resolve() != target_wavemeter.resolve():
            wavemeter_file.rename(target_wavemeter)

def find_datasets(folder: Path) -> List[Dict]:
    folder = folder.expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Acquisition folder does not exist: {folder}")

    # First, move any loose input pairs into their own folders
    organize_loose_datasets(folder)

    datasets: List[Dict] = []

    # Look for dataset folders named like:
    # YYYYMMDD_HHMMSS
    # or
    # YYYYMMDD_HHMMSS_EC1020_HL1091
    for dataset_dir in sorted([p for p in folder.iterdir() if p.is_dir()]):
        prefix = dataset_dir.name

        scope_files = sorted(dataset_dir.glob("*_full_scope_all.npz"))
        wavemeter_files = sorted(dataset_dir.glob("*_full_wavemeter.csv"))

        if len(scope_files) != 1 or len(wavemeter_files) != 1:
            continue

        npz = scope_files[0]
        csv = wavemeter_files[0]

        summary = dataset_dir / f"{prefix}_summary.txt"
        strong_csv = dataset_dir / f"{prefix}_strong_peaks.csv"
        fit_csv = dataset_dir / f"{prefix}_fit_results.csv"
        done_flag = dataset_dir / f"{prefix}_DONE.txt"

        analyzed = done_flag.exists()

        ec_temp, hl_temp = extract_temperatures_from_name(npz.stem)

        datasets.append(
            {
                "stem": prefix,
                "scope": npz,
                "wavemeter": csv,
                "folder": dataset_dir,
                "analyzed": analyzed,
                "ec_temp": ec_temp,
                "hl_temp": hl_temp,
            }
        )

    datasets.sort(key=lambda d: d["stem"])
    return datasets


# ============================================================
# CONFIG + RUN
# ============================================================
def build_cfg(dataset: Dict) -> AnalysisConfig:
    t = load_cfg_template()
    cfg = AnalysisConfig(
        scope_file=str(dataset["scope"]),
        wavemeter_file=str(dataset["wavemeter"]),
        output_dir=str(dataset["folder"] if OUTPUT_TO_INPUT_FOLDER else dataset["folder"] / f"{dataset['stem']}_analysis"),
        ttl_threshold_v=t.get("ttl_threshold_v", 1.5),
        bin_size=t.get("bin_size", 10),
        freq_clean_kernel=t.get("freq_clean_kernel", 21),
        scan_smooth_window=t.get("scan_smooth_window", 101),
        coarse_block=t.get("coarse_block", 20),
        min_ramp_points=t.get("min_ramp_points", 150),
        rise_threshold_fraction=t.get("rise_threshold_fraction", 0.30),
        start_backtrack_blocks=t.get("start_backtrack_blocks", 1),
        end_trim_blocks=t.get("end_trim_blocks", 1),
        drop_edge_ramps=t.get("drop_edge_ramps", True),
        edge_fraction=t.get("edge_fraction", 0.12),
        eps=t.get("eps", 1e-12),
        max_shift=t.get("max_shift", 8),
        display_savgol_window=t.get("display_savgol_window", 3),
        display_savgol_poly=t.get("display_savgol_poly", 1),
        peak_detect_window=t.get("peak_detect_window", 31),
        peak_detect_poly=t.get("peak_detect_poly", 2),
        peak_threshold_sigma=t.get("peak_threshold_sigma", 0.5),
        peak_min_region_points=t.get("peak_min_region_points", 12),
        peak_ratio_window=t.get("peak_ratio_window", 12),
        peak_min_area_fraction=t.get("peak_min_area_fraction", 0.3),
        peak_min_area_snr=t.get("peak_min_area_snr", 0.4),
        peak_min_ratio_bump_snr=t.get("peak_min_ratio_bump_snr", 0.3),
        f_ref_thz=t.get("f_ref_thz", 355.802555),
        expected_peak_positions_thz=t.get("expected_peak_positions_thz", default_expected_peaks()),
        single_fit_peak_index=t.get("single_fit_peak_index", 0),
        fit_half_width_thz=t.get("fit_half_width_thz", 0.00015),
        fit_exclude_half_width_thz=t.get("fit_exclude_half_width_thz", 6e-5),
        multi_fit_peak_indices=t.get("multi_fit_peak_indices", [0, 1]),
        multi_fit_margin_thz=t.get("multi_fit_margin_thz", 0.00008),
        multi_fit_local_half_width_thz=t.get("multi_fit_local_half_width_thz", 6e-5),
        global_baseline_exclude_half_width_thz=t.get("global_baseline_exclude_half_width_thz", 4e-5),
        save_per_ramp_fits=t.get("save_per_ramp_fits", True),
        per_ramp_min_successful_peaks=t.get("per_ramp_min_successful_peaks", 2),
        save_plots=t.get("save_plots", True),
        show_plots=t.get("show_plots", False),
        debug=DEBUG or t.get("debug", False),
        output_prefix=dataset["stem"] if OUTPUT_TO_INPUT_FOLDER else "",
        ec_temp=dataset.get("ec_temp"),
        hl_temp=dataset.get("hl_temp"),
    )
    return cfg


def print_result_summary(dataset: Dict, result: Dict) -> None:
    print(f"\n{dataset['stem']}")
    print("-" * len(dataset["stem"]))
    if dataset.get("ec_temp") is not None:
        print(f"EC temperature: {dataset['ec_temp']:.0f} C")
    if dataset.get("hl_temp") is not None:
        print(f"HL temperature: {dataset['hl_temp']:.0f} C")
    print(f"Valid rising ramps: {len(result['processed'])}")
    print(f"Strong peaks found: {len(result['strong_peaks'])}")
    if result["isotope_assignment"]:
        for det_label, info in result["isotope_assignment"].items():
            print(f"{det_label} -> {info['expected_label']} | {info['detected_x_thz']:.9f} THz | offset = {info['offset_mhz']:.3f} MHz")
    print(f"Summary: {result['summary_txt']}")


# ============================================================
# MENU
# ============================================================
def menu_select(datasets: List[Dict]) -> List[Dict]:
    print(f"\nFound {len(datasets)} dataset(s) in {ACQ_FOLDER.resolve()}\n")
    for i, d in enumerate(datasets, start=1):
        status = "done" if d["analyzed"] else "new"
        ec_str = "?" if d.get("ec_temp") is None else f"{d['ec_temp']:.0f}"
        hl_str = "?" if d.get("hl_temp") is None else f"{d['hl_temp']:.0f}"
        print(f"{i:2d}. {d['stem']} [EC={ec_str} C, HL={hl_str} C] [{status}]")

    print("\nOptions:")
    print("  n   = run all new datasets")
    print("  a   = run all datasets")
    print("  l   = run latest new dataset")
    print("  3   = run dataset 3")
    print("  2,5 = run datasets 2 and 5")

    choice = input("Select dataset(s): ").strip().lower()
    if choice == "n":
        return [d for d in datasets if not d["analyzed"]]
    if choice == "a":
        return datasets
    if choice == "l":
        new_ds = [d for d in datasets if not d["analyzed"]]
        return new_ds[-1:] if new_ds else []

    idxs: List[int] = []
    for chunk in choice.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        idx = int(chunk)
        if idx < 1 or idx > len(datasets):
            raise IndexError(f"Selection {idx} is out of range")
        idxs.append(idx - 1)
    return [datasets[i] for i in idxs]


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    datasets = find_datasets(ACQ_FOLDER)
    if not datasets:
        print(f"No matched .npz/.csv dataset pairs found in {ACQ_FOLDER.resolve()}")
        return

    if MODE == "all_new":
        selected = [d for d in datasets if not d["analyzed"]]
    elif MODE == "one_latest":
        new_ds = [d for d in datasets if not d["analyzed"]]
        selected = new_ds[-1:] if new_ds else []
    else:
        selected = menu_select(datasets)

    if not selected:
        print("No datasets selected.")
        return

    for dataset in selected:
        if dataset["analyzed"]:
            reply = input(f"{dataset['stem']} is already analysed. Re-run and overwrite outputs? [y/N]: ").strip().lower()
            if reply != "y":
                print(f"Skipping {dataset['stem']}")
                continue

        cfg = build_cfg(dataset)
        
        done_flag = dataset["folder"] / f"{dataset['stem']}_DONE.txt"
        if done_flag.exists():
            done_flag.unlink()
        
        result = analyze(cfg)
        done_flag = dataset["folder"] / f"{dataset['stem']}_DONE.txt"
        done_flag.write_text("Analysis complete\n", encoding="utf-8")
        print_result_summary(dataset, result)


if __name__ == "__main__":
    main()
