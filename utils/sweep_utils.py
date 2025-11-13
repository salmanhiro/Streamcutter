import os
import shutil
from pathlib import Path
from collections import Counter
import numpy as np
import astropy.units as u
from astropy.table import Table, QTable

def _to_deg_array(x):
    """Return a 1D float64 numpy array of degrees from Quantity, Table column, or ndarray."""
    if isinstance(x, u.Quantity):
        return np.asarray(x.to(u.deg).value, dtype=np.float64)
    # astropy Column or plain array-like
    arr = np.asarray(x)
    return np.asarray(arr, dtype=np.float64)

def get_needed_sweep_files(
    ra,
    dec,
    sweep_lookup_csv,
    dest_dir="./sweep_needed",
    min_count=50,
    skip_if_exists=True,
    dry_run=False,
):
    """
    Determine and (optionally) copy the minimal set of LS DR9 'sweep' bricks needed
    to cover a simulated stream's RA/Dec footprint.

    Parameters
    ----------
    ra, dec : array-like or astropy.units.Quantity
        RA and Dec of simulated stream points. Units can be Quantity (deg) or raw floats (deg).
    sweep_lookup_csv : str or Path
        CSV with columns mapping integer (floored) RA, Dec -> file_location.
        Expected columns: 'RA', 'Dec', 'file_location'
        (matches your ../sweep/sweep_photometry_lookup_coord.csv)
    dest_dir : str or Path, optional
        Local directory where the selected bricks will be copied. Created if missing.
    min_count : int, optional
        Keep only bricks that cover at least this many stream points (default 50).
    skip_if_exists : bool, optional
        If True, do not overwrite files that are already present at dest_dir.
    dry_run : bool, optional
        If True, only compute and return which files would be copied; no I/O.

    Returns
    -------
    result : dict
        {
          'selected_files': set of source file paths (meeting min_count),
          'counts': dict {source_path: count},
          'copied_files': [dest_paths],         # empty if dry_run=True
          'skipped_existing': [dest_paths],     # empty if dry_run=True
          'missing_source_files': [source_paths]
        }
    """
    # --- Normalize inputs ---
    ra_deg  = _to_deg_array(ra)
    dec_deg = _to_deg_array(dec)
    if ra_deg.shape != dec_deg.shape:
        raise ValueError("ra and dec must have the same shape")

    # Floor to integer degree tiles (matches how your lookup was constructed)
    ra_floor  = np.floor(ra_deg).astype(int)
    dec_floor = np.floor(dec_deg).astype(int)

    # --- Read lookup table ---
    sweep_lookup = Table.read(str(sweep_lookup_csv), format="csv")
    if not set(["RA", "Dec", "file_location"]).issubset(sweep_lookup.colnames):
        raise ValueError("Lookup CSV must contain columns: RA, Dec, file_location")

    # Build fast dict: (RA_int, Dec_int) -> file_location
    lookup_dict = {
        (int(row["RA"]), int(row["Dec"])): str(row["file_location"])
        for row in sweep_lookup
    }

    # --- Map each (ra_i, dec_i) to a brick path (or None if no match) ---
    matched_files = [lookup_dict.get((ri, di), None) for ri, di in zip(ra_floor, dec_floor)]

    # Count hits per brick
    counts_all = Counter(f for f in matched_files if f is not None)
    # Keep only bricks with >= min_count
    selected_files = {f for f, c in counts_all.items() if c >= int(min_count)}

    # Prepare destination
    dest_dir = Path(dest_dir)
    copied_files, skipped_existing, missing_source = [], [], []
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Copy each selected brick
        for src in sorted(selected_files):
            src_path = Path(src)
            if not src_path.is_file():
                missing_source.append(str(src_path))
                # continue without failing
                continue

            dest_path = dest_dir / src_path.name
            if dest_path.exists() and skip_if_exists:
                skipped_existing.append(str(dest_path))
                continue

            shutil.copy2(str(src_path), str(dest_path))
            copied_files.append(str(dest_path))

    return {
        "selected_files": set(selected_files),
        "counts": dict(counts_all),
        "copied_files": copied_files,
        "skipped_existing": skipped_existing,
        "missing_source_files": missing_source,
    }

# ---------------------------
# Example usage with your sim table:
# ---------------------------
# res = get_needed_sweep_files(
#     sim_stream_tab["RA"],          # astropy Quantity (deg)
#     sim_stream_tab["DEC"],         # astropy Quantity (deg)
#     sweep_lookup_csv="../sweep/sweep_photometry_lookup_coord.csv",
#     dest_dir="./data/sweep_needed",
#     min_count=50,
#     skip_if_exists=True,
#     dry_run=False,                 # set True first to preview
# )
# print("Selected bricks:", res["selected_files"])
# print("Missing sources:", res["missing_source_files"])
# print("Copied:", res["copied_files"])
# print("Skipped existing:", res["skipped_existing"])
