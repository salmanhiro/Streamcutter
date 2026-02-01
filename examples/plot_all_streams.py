#!/usr/bin/env python3
"""
Plot all simulated GC streams (with Pal 5FK15 variant) over DESI and LS footprints.
- GC label positions come from GCParams table (not from stream points)
- adjustText is used to relax labels
- Black circled marker indicates the GC position, on the topmost layer
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.units as u
from adjustText import adjust_text

import sys
sys.path.append('..')

from utils.GC_utils import GCParams

# import your GCParams (adjust the import path to your project if needed)

# Import your existing footprint functions
from utils.plot_utils import plot_desi_region, plot_ls_region

# ---------------------------------------------------------------------
# Use DESI plot style
# ---------------------------------------------------------------------
plt.style.use('desi.mplstyle')

# Directory containing the simulated stream FITS files
stream_dir = "../simulated_streams"

# ---------------------------------------------------------------------
# Load FITS files (include Pal_5FK15, exclude default Pal_5 and VB21)
# ---------------------------------------------------------------------
stream_files = sorted([
    f for f in os.listdir(stream_dir)
    if f.endswith(".fits") 
])

print(f"[v] Found {len(stream_files)} simulated streams")

# ---------------------------------------------------------------------
# GC catalog
# ---------------------------------------------------------------------
gc = GCParams()
gc.table_path = os.path.abspath("../data/mw_gc_parameters_orbital_structural_time.ecsv")

def fname_to_gcname(fname: str) -> str:
    """
    Map stream filename to the base GC key used in the GCParams table.
    Example:
      simulated_stream_Pal_5FK15.fits -> Pal_5
      simulated_stream_NGC_5466_xyz.fits -> NGC_5466
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    base = base.replace("simulated_stream_", "")
    # Keep leading pattern like 'Pal_5' or 'NGC_5466' and drop trailing variants (FK15, etc.)
    m = re.match(r'^([A-Za-z]+_\d+)', base)
    if m:
        return m.group(1)
    # Fallback: if no number form, return the cleaned base (you can adapt if your table uses spaces)
    return base

def get_gc_radec(gc_name: str):
    """
    Query GCParams for RA/Dec (in degrees) of the named GC.
    Handles Quantity or plain floats and multiple possible column casings.
    """
    row = gc.get_row(gc_name)  # user-provided API
    # Try common column names
    for ra_key in ("ra", "RA", "ra_deg", "RA_DEG"):
        if ra_key in row.colnames:
            ra_val = row[ra_key]
            break
    else:
        raise KeyError(f"RA column not found for GC '{gc_name}'")

    for dec_key in ("dec", "DEC", "dec_deg", "DEC_DEG"):
        if dec_key in row.colnames:
            dec_val = row[dec_key]
            break
    else:
        raise KeyError(f"Dec column not found for GC '{gc_name}'")

    # Convert to float degrees if they carry units
    try:
        ra_deg  = (ra_val * 1.0).to_value(u.deg) if hasattr(ra_val, "to_value") else float(ra_val)
        dec_deg = (dec_val * 1.0).to_value(u.deg) if hasattr(dec_val, "to_value") else float(dec_val)
    except Exception:
        # Some ECSV tables store Quantity directly
        ra_deg  = ra_val.to(u.deg).value if hasattr(ra_val, "to") else float(ra_val)
        dec_deg = dec_val.to(u.deg).value if hasattr(dec_val, "to") else float(dec_val)

    return float(ra_deg % 360.0), float(dec_deg)

# ---------------------------------------------------------------------
# Create the plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel(r"RA [deg]")
ax.set_ylabel(r"Dec [deg]")

# Plot DESI and LS footprint outlines
plot_desi_region(ax=ax, length_threshold=5, color="black", tiles_path="../data/tiles-main.ecsv")
plot_ls_region(ax=ax, length_threshold=5, color="gray", tiles_path="../data/tiles-main.ecsv")


# ---------------------------------------------------------------------
# Assign distinct colors per GC (for stream points & label color)
# ---------------------------------------------------------------------
cmap = plt.cm.tab20
colors = [cmap(i % 20) for i in range(len(stream_files))]

texts = []   # collect for adjust_text

# ---------------------------------------------------------------------
# Plot each simulated stream
# ---------------------------------------------------------------------
for i, fname in enumerate(stream_files):
    path = os.path.join(stream_dir, fname)
    stream_tab = Table.read(path)

    ra_stream = np.asarray(stream_tab["RA"], dtype=float)
    dec_stream = np.asarray(stream_tab["DEC"], dtype=float)

    # Scatter of the whole stream
    ax.scatter(
        ra_stream, dec_stream, s=1, alpha=1, lw=0,
        color=colors[i], zorder=2
    )

    # Look up GC name and position from GCParams
    gc_name_key = fname_to_gcname(fname)          # e.g., "Pal_5"
    ra_gc, dec_gc = get_gc_radec(gc_name_key)     # authoritative RA/Dec for label & marker

    # Aesthetic display name (optional): tighten label text (remove underscores, shorten NGC)
    label_text = gc_name_key.replace("_", "").replace("NGC", "N")

    # --- Topmost GC marker: black circle with small white dot ---
    ax.scatter([ra_gc], [dec_gc], s=50, c="white", edgecolors="k",
               linewidths=0.9, zorder=30)

    # Collect text for adjust_text (colored by stream color), on top
    txt = ax.text(
        ra_gc, dec_gc, label_text,
        fontsize=7, color='k',
        ha="center", va="center", alpha=0.95, zorder=31
    )
    texts.append(txt)

# ---------------------------------------------------------------------
# Relax labels to avoid overlaps
# ---------------------------------------------------------------------
adjust_text(
    texts,
    expand=(1.2, 2.0),
    only_move={'points':'y', 'texts':'xy', 'objects':'xy'},
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.5, alpha=0.8),
    ax=ax
)

# ---------------------------------------------------------------------
# Finalize plot
# ---------------------------------------------------------------------
ax.invert_xaxis()  # RA increases to the left (sky convention)
ax.grid(alpha=0.3)

# ---------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------
os.makedirs("results", exist_ok=True)
out_path = "results/all_gc_streams_desi_ls.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"[v] Saved atlas figure to: {out_path}")
