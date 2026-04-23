"""
migrate_datasets.py
===================
One-off migration script to reorganise existing flat dataset folders
into the new date-hierarchical structure with a raw/ subfolder.

BEFORE:
    data_acq/
        20260421_132154_EC1050_HL1100/
            20260421_132154_EC1050_HL1100_full_scope_all.npz
            20260421_132154_EC1050_HL1100_full_wavemeter.csv
            20260421_132154_EC1050_HL1100_summary.txt
            ...

AFTER:
    data_acq/
        2026-04/
            21/
                20260421_132154_EC1050_HL1100/
                    raw/
                        20260421_132154_EC1050_HL1100_full_scope_all.npz
                        20260421_132154_EC1050_HL1100_full_wavemeter.csv
                    20260421_132154_EC1050_HL1100_summary.txt
                    ...

Usage:
    python migrate_datasets.py            <- dry run (no changes)
    python migrate_datasets.py --execute  <- actually move files
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

# ── Edit this to point at your data_acq folder ──────────────
DATA_ACQ = Path(r"C:\Users\dysprosium\labscript-suite\userlib\labscriptlib\quantum_gas_microscope\Dysprosium\Spectroscopy\data_acq")

# Files that belong in raw/ (raw scope + wavemeter data)
RAW_PATTERNS = ["*_full_scope_all.npz", "*_full_wavemeter.csv"]

# Date pattern: YYYYMMDD at start of folder name
DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_")


def find_flat_datasets(root: Path) -> list[Path]:
    """Find dataset folders sitting directly in root (old flat structure)."""
    flat = []
    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        # Skip if it already looks like a YYYY-MM date folder
        if re.match(r"^\d{4}-\d{2}$", item.name):
            continue
        # Must start with a date
        if DATE_RE.match(item.name):
            flat.append(item)
    return flat


def plan_migration(root: Path) -> list[tuple[Path, Path]]:
    """
    Return list of (src_dataset_dir, dst_dataset_dir) pairs.
    dst is the new location under YYYY-MM/DD/.
    """
    moves = []
    for src in find_flat_datasets(root):
        m = DATE_RE.match(src.name)
        if not m:
            continue
        year, month, day = m.group(1), m.group(2), m.group(3)
        dst = root / f"{year}-{month}" / day / src.name
        moves.append((src, dst))
    return moves


def migrate(root: Path, execute: bool) -> None:
    moves = plan_migration(root)

    if not moves:
        print("No flat dataset folders found — nothing to migrate.")
        return

    print(f"{'DRY RUN — no files will be moved' if not execute else 'EXECUTING MIGRATION'}")
    print(f"Found {len(moves)} dataset(s) to migrate\n")

    for src, dst in moves:
        print(f"  {src.name}")
        print(f"    → {dst.relative_to(root)}")

        # Work out which files go to raw/ and which stay in root
        raw_files = []
        for pattern in RAW_PATTERNS:
            raw_files.extend(src.glob(pattern))

        other_files = [
            f for f in src.iterdir()
            if f.is_file() and f not in raw_files
        ]

        if raw_files:
            print(f"    raw/  : {', '.join(f.name for f in sorted(raw_files))}")
        if other_files:
            print(f"    root  : {', '.join(f.name for f in sorted(other_files))}")

        if execute:
            # Create destination folders
            raw_dst = dst / "raw"
            raw_dst.mkdir(parents=True, exist_ok=True)

            # Move raw files into raw/
            for f in raw_files:
                shutil.move(str(f), str(raw_dst / f.name))

            # Move remaining files to dataset root
            for f in other_files:
                shutil.move(str(f), str(dst / f.name))

            # Remove now-empty source folder
            try:
                src.rmdir()
            except OSError:
                print(f"    WARNING: could not remove {src} — may not be empty")

        print()

    if not execute:
        print("─" * 60)
        print("Dry run complete. Run with --execute to apply changes.")
    else:
        print("─" * 60)
        print("Migration complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate flat datasets to date-hierarchical structure")
    parser.add_argument("--execute", action="store_true",
                        help="Actually move files (default is dry run)")
    args = parser.parse_args()
    migrate(DATA_ACQ, execute=args.execute)
