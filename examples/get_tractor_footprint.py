from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import argparse
from astropy.table import Table, vstack
from astropy.io import fits
import matplotlib.pyplot as plt
from utils import plot_utils
from tqdm import tqdm

MAX_WORKERS = 10

# keep only columns you actually use (include mask columns!)
COLS_TO_KEEP = [
    "ra", "dec",
    "flux_g", "flux_r", "flux_z",
    "pmra", "pmdec",
    "brickid", "type", "ref_id",
    "mw_transmission_g", "mw_transmission_r", "mw_transmission_z",
    "parallax",
    "allmask_g", "allmask_r",
]


from tqdm import tqdm
import fitsio, os

def _filter_psf(arr):
    if "type" not in arr.dtype.names:
        return arr
    t = arr["type"]
    m = (t == b"PSF") if t.dtype.kind == "S" else (t == "PSF")
    return arr[m]

def _downcast_all_fields(arr):
    """Keep ALL columns; just shrink dtypes safely."""
    names = arr.dtype.names
    cols, newdt = [], []
    for nm in names:
        a = np.asarray(arr[nm])
        k = a.dtype.kind
        # float64 -> float32
        if k == "f" and a.dtype.itemsize > 4:
            a = a.astype("f4", copy=False)
        # ints: try to fit in 32/16/8 where possible
        elif k in "iu":
            mx = int(a.max(initial=0)); mn = int(a.min(initial=0))
            if k == "i":
                if -2**31 <= mn and mx < 2**31: a = a.astype("i4", copy=False)
            else:
                if 0 <= mn and mx < 2**32: a = a.astype("u4", copy=False)
            # mask-ish columns → u1/u2 if they fit
            low = nm.lower()
            if any(tag in low for tag in ("mask","allmask","blob","bits","flag")):
                mx = int(a.max(initial=0))
                if mx <= 0xFF:   a = a.astype("u1", copy=False)
                elif mx <= 0xFFFF: a = a.astype("u2", copy=False)
        # ensure fixed width for 'type'
        elif k == "S" and nm.lower() == "type":
            a = a.astype("S3", copy=False)
        cols.append(a); newdt.append((nm, a.dtype))
    out = np.empty(cols[0].shape, dtype=np.dtype(newdt))
    for nm, a in zip(names, cols): out[nm] = a
    return out


from concurrent.futures import ThreadPoolExecutor, as_completed


def build_from_list(paths, out_fits):
    """
    Read a list of Tractor FITS files, keep only COLS_TO_KEEP, PSF-only,
    downcast dtypes, and write a single TRACTOR table with fitsio.

    Astropy fallback is disabled here on purpose.
    """
    import fitsio
    import numpy as np

    def read_recarray(p):
        """Read TRACTOR extension from one file, only COLS_TO_KEEP."""
        try:
            with fitsio.FITS(p, "r") as f:
                idx = None
                for i, h in enumerate(f):
                    if (
                        h.get_exttype() == "BINARY_TBL"
                        and h.get_extname().upper() == "TRACTOR"
                    ):
                        idx = i
                        break
                if idx is None:
                    # fallback: pick largest binary table
                    sizes = [
                        (i, h.get_nrows())
                        for i, h in enumerate(f)
                        if h.get_exttype() == "BINARY_TBL"
                    ]
                    if not sizes:
                        return None
                    idx = max(sizes, key=lambda x: x[1])[0]

                if f[idx].get_nrows() == 0:
                    return None

                # read only the columns we actually need
                return f[idx].read(columns=COLS_TO_KEEP)
        except Exception as e:
            print(f"[skip] {p}: {e}")
            return None

    recs = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(read_recarray, p): p for p in paths}
        for i, fut in enumerate(as_completed(futs), 1):
            arr = fut.result()
            if arr is not None and arr.size:
                recs.append(arr)
            if i % 100 == 0 or i == len(futs):
                total = sum(a.size for a in recs) if recs else 0
                print(
                    f"[fitsio] {out_fits}: {i}/{len(futs)} read; "
                    f"rows so far: {total}"
                )

    if not recs:
        raise RuntimeError(f"[err] {out_fits}: fitsio read returned no rows.")

    # stack all recarrays
    stacked = np.concatenate(recs, axis=0)

    # PSF-only
    stacked = _filter_psf(stacked)
    if stacked.size == 0:
        raise RuntimeError(f"[err] {out_fits}: no PSF rows after reading")

    # shrink dtypes, keeping all selected fields
    stacked = _downcast_all_fields(stacked)

    # write out
    with fitsio.FITS(out_fits, "rw", clobber=True) as f:
        f.write(stacked, extname="TRACTOR")

    print(
        f"[done] wrote {out_fits}  rows={stacked.shape[0]}  "
        f"cols={len(stacked.dtype.names)}  (fitsio, cols-only)"
    )
    return stacked.shape[0]

def _read_lookup(csv_path):
    ras, decs, paths = [], [], []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        # accept either header set: (path, ra_deg, dec_deg) or (ra_deg, dec_deg, path)
        for row in r:
            ra  = float(row["ra_deg"])
            dec = float(row["dec_deg"])
            p   = row["path"]
            ras.append(ra); decs.append(dec); paths.append(p)
    return np.array(ras), np.array(decs), np.array(paths)

def _angsep_deg(ra1, dec1, ra2, dec2):
    """Great-circle separation (deg), vectorized."""
    ra1 = np.radians(ra1); dec1 = np.radians(dec1)
    ra2 = np.radians(ra2); dec2 = np.radians(dec2)
    s = np.sin((dec2 - dec1)/2.0)**2 + np.cos(dec1)*np.cos(dec2)*np.sin((ra2 - ra1)/2.0)**2
    return np.degrees(2.0*np.arcsin(np.sqrt(s)))

def _dra_wrap(ra_a, ra_b):
    """|ΔRA| in deg with wrap (0..360). Works with arrays."""
    return np.abs((ra_a - ra_b + 180.0) % 360.0 - 180.0)

def select_needed_files(stream_ra, stream_dec, brick_ra, brick_dec, brick_paths,
                        radius_deg=1.0, min_sim_stars=10):
    """
    For each stream point, select bricks within radius_deg of that (ra,dec).
    Keep only bricks that have >= min_sim_stars simulated points within radius.
    Returns (paths, ras, decs) as aligned lists (unique by path).
    """
    selected = {}        # path -> (ra, dec)
    hit_counts = {}      # path -> number of sim stars within radius

    # cheap prefilter width (pad to avoid missing at high |dec|)
    pad_ra  = radius_deg * 1.2
    pad_dec = radius_deg

    for ra0, dec0 in zip(stream_ra, stream_dec):
        # prefilter by RA (wrap safe) and Dec window
        m = (_dra_wrap(brick_ra, ra0) <= pad_ra) & (np.abs(brick_dec - dec0) <= pad_dec)
        if not np.any(m):
            continue

        # exact great-circle cut on the subset (brick centers within radius of this sim star)
        sep = _angsep_deg(ra0, dec0, brick_ra[m], brick_dec[m])
        sel = sep <= radius_deg
        if not np.any(sel):
            continue

        # map back to original indices; increment per-brick hit counts
        subset_idx = np.nonzero(m)[0]
        hit_idx = subset_idx[sel]
        for j in hit_idx:
            p = brick_paths[j]
            if p not in selected:              # preserve first-seen RA/DEC
                selected[p] = (brick_ra[j], brick_dec[j])
            hit_counts[p] = hit_counts.get(p, 0) + 1

    # apply the min_sim_stars threshold
    keep_paths = [p for p, n in hit_counts.items() if n >= min_sim_stars]

    # return aligned lists
    paths = sorted(keep_paths)                 # or keep insertion order: list(keep_paths)
    ras   = [selected[p][0] for p in paths]
    decs  = [selected[p][1] for p in paths]
    return paths, ras, decs


def main():
    # create args for simulated stream file
    parser = argparse.ArgumentParser(description="Match simulated streams to LS Tractor files.")
    parser.add_argument("--lookup", type=str, default="/cluster/home/salmanhiro/DESI/tractor_ra_dec_dr10.csv",
                        help="CSV file with columns: path, ra_deg, dec_deg for Tractor files.")
    parser.add_argument("--sim-file", type=str, nargs='+', required=True,
                        help="One or more simulated stream files (CSV/ECSV/FITS) with columns RA [deg], DEC [deg].")
    parser.add_argument("--target-radius", type=float, default=1.0,
                        help="Radius [deg] around stream points to select target Tractor files.")
    parser.add_argument("--env-radius", type=float, default=1.5,
                        help="Radius [deg] around stream points to select background Tractor files.")
    parser.add_argument("--min-stars", type=int, default=1,
                        help="Minimum number of simulated stars within radius to keep a Tractor file.")
    parser.add_argument("--outdir", type=str, default="tractor",
                        help="Output directory for results.")
    parser.add_argument("--concat", action="store_true",
                        help="If set, concatenate selected Tractor files into one FITS per stream.")
    
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Load lookup table
    brick_ra, brick_dec, brick_paths = _read_lookup(args.lookup)

    # read simulated stream file
    simulated_stream = Table.read(args.sim_file[0])

    stream_ra = np.asarray(simulated_stream["RA"], dtype=float)
    stream_dec = np.asarray(simulated_stream["DEC"], dtype=float)
    print(f"simulated stream ra and dec len: {len(stream_ra)}, {len(stream_dec)}")
    tgt_file_list, tgt_ra_sel, tgt_dec_sel = select_needed_files(stream_ra, stream_dec, brick_ra, brick_dec, 
                                                                 brick_paths, radius_deg=args.target_radius, 
                                                                 min_sim_stars=args.min_stars)
    print(f"[ok] Needed target Tractor files: {len(tgt_file_list)}")

    env_all, env_all_ra_sel, env_all_dec_sel = select_needed_files(stream_ra, stream_dec, brick_ra, brick_dec,
                              brick_paths, radius_deg=args.env_radius, min_sim_stars=args.min_stars)
    env_only = sorted(set(env_all) - set(tgt_file_list))
    print(f"[ok] Needed background Tractor files: {len(env_only)}")

    stream_name = Path(args.sim_file[0]).stem

    # visualize the ra and dec for target and background files
    plt.figure(figsize=(8,6))
    plt.scatter(env_all_ra_sel, env_all_dec_sel, s=10, color='orange', label='Background Files', alpha=0.5)
    plt.scatter(tgt_ra_sel, tgt_dec_sel, s=10, color='green', label='Target Files', alpha=0.7)
    # overlay stellar stream points
    plt.scatter(stream_ra, stream_dec, s=1, color='blue', label='Simulated Stream Points', alpha=0.3)

    # draw legacysurvey footprint to canvas
    ax = plt.gca()
    plot_utils.plot_ls_region(ax=ax, color='k')

    plt.xlabel('RA (deg)')
    plt.ylabel('DEC (deg)')
    plt.gca().invert_xaxis()
    plt.title(f"{stream_name} - Tractor File Footprint")
    plt.legend()
    plt.grid(True)
    # set limit from env_all_ra_sel and env_all_dec_sel
    plt.xlim(max(env_all_ra_sel)+2, min(env_all_ra_sel)-2)
    plt.ylim(min(env_all_dec_sel)-2, max(env_all_dec_sel)+2)
    plt.savefig(outdir / f"{stream_name}_tractor_footprint.png")

    # Save file to list
    # Get stream name from sim file name
    tgt_file_list_path = outdir / f"{stream_name}_tgt_tractor_files.txt"
    with open(tgt_file_list_path, "w") as f:
        for p in tgt_file_list:
            f.write(f"{p}\n")
    print(f"[ok] Wrote target Tractor file list to {tgt_file_list_path}")
    bg_file_list_path = outdir / f"{stream_name}_bg_tractor_files.txt"
    with open(bg_file_list_path, "w") as f:
        for p in env_only:
            f.write(f"{p}\n")
    print(f"[ok] Wrote background Tractor file list to {bg_file_list_path}")

    # calculate total area of tgt and bg files
    area_per_brick = 0.25  # deg^2
    tgt_area = len(tgt_file_list) * area_per_brick
    bg_area = len(env_only) * area_per_brick
    # print metadata to file
    metadata_path = outdir / f"{stream_name}_tractor_metadata.txt"

    # calculate estimate total file size 
    total_size_tgt = sum(Path(f).stat().st_size for f in tgt_file_list)
    total_size_bg = sum(Path(f).stat().st_size for f in env_only)
    print(f"[info] Estimated total size of target Tractor files: {total_size_tgt / 1e9:.2f} GB")
    print(f"[info] Estimated total size of background Tractor files: {total_size_bg / 1e9:.2f} GB")


    with open(metadata_path, "w") as f:
        f.write(f"Stream Name: {stream_name}\n")
        f.write(f"Number of Target Tractor Files: {len(tgt_file_list)}\n")
        f.write(f"Total Target Area (deg^2): {tgt_area}\n")
        f.write(f"Number of Background Tractor Files: {len(env_only)}\n")
        f.write(f"Total Background Area (deg^2): {bg_area}\n")
        f.write(f"Ratio of Background to Target Area: {bg_area / tgt_area}\n")
        f.write(f"Estimated Total Size of target Tractor Files (GB): {total_size_tgt / 1e9:.2f}\n")
        f.write(f"Estimated Total Size of background Tractor Files (GB): {total_size_bg / 1e9:.2f}\n")
    print(f"[ok] Wrote metadata to {metadata_path}")

    # Read file on list and concat them into a single fits table
    # For target
    if args.concat and tgt_file_list:
        tgt_out_path = outdir / f"{stream_name}_tgt_tractor_combined.fits"
        build_from_list(tgt_file_list, tgt_out_path)
        print(f"[ok] Wrote combined target Tractor catalog to {tgt_out_path}")
        # check file size
        tgt_size = Path(tgt_out_path).stat().st_size
        print(f"[info] Combined target Tractor catalog size: {tgt_size / 1e9:.2f} GB")
    # For background
    if args.concat and env_only:
        bg_out_path = outdir / f"{stream_name}_bg_tractor_combined.fits"
        build_from_list(env_only, bg_out_path)
        print(f"[ok] Wrote combined background Tractor catalog to {bg_out_path}")
        # check file size
        bg_size = Path(bg_out_path).stat().st_size
        print(f"[info] Combined background Tractor catalog size: {bg_size / 1e9:.2f} GB")

if __name__ == "__main__":
    main()