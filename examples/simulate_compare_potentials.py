#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import astropy.units as u
from astropy.table import QTable
from astropy.io import fits
import matplotlib.pyplot as plt
import agama
from utils.plot_utils import plot_desi_region
from matplotlib.lines import Line2D
from astropy.table import Table


from utils import mock_stream_utils, coordinate_utils

ALL_POTENTIALS = [
    "BarMWPortail17.ini",
    "MWPotential2014.ini",
    # "BT08.ini",
    "Cautun20.ini",
    "SCM_MW.ini",
    "LMCVasiliev24",
    # "DB98.ini",
    # "MWPotential2022.ini"
]

def simulate_stream_for_potential(gc_row, pot_ini_path, n_particles, n_orbits, seed=0):
    import os
    import numpy as np
    import astropy.units as u
    import agama
    from astropy.table import QTable
    from utils import mock_stream_utils, coordinate_utils

    agama.setUnits(length=1, velocity=1, mass=1)

    # --- GC phase-space from table ---
    ra_q   = gc_row["RA"].value
    dec_q  = gc_row["DEC"].value
    dist_q = gc_row["Rsun"].value
    pmra   = gc_row["mualpha"].value
    pmdec  = gc_row["mu_delta"].value
    mass_sat = gc_row["Mass"].value
    rv       = gc_row["<RV>"].value
    rhm      = gc_row["rh,m"].value
    orbit_t  = gc_row["orbit_period_max"].value

    # --- Host potential selection (robust) ---
    # pot_ini_path can be a full path, filename, or the sentinel "LMCVasiliev24"
    s = "" if pot_ini_path is None else str(pot_ini_path)
    pot_key = os.path.splitext(os.path.basename(s))[0]  # e.g. "MWPotential2014" or "LMCVasiliev24"

    if pot_key == "LMCVasiliev24":
        # dynamic MW+LMC potential
        from potentials import LMCVasiliev24 as LMCmod
        pot_host = LMCmod.build_potential()
    else:
        # regular static potential from INI file (path or relative filename)
        pot_host = agama.GalaPotential(pot_ini_path)

    # satellite (GC progenitor) plummer model
    pot_sat  = agama.GalaPotential(type='Plummer', mass=mass_sat, scaleRadius=rhm)

    # progenitor IC in Galactocentric coordinates
    prog_w0  = coordinate_utils.get_galactocentric_coords(ra_q, dec_q, dist_q, rv, pmra, pmdec)[0]

    # total integration time (negative = integrate backward)
    time_total = (-n_orbits * orbit_t)/978 if orbit_t < 1000 else -3

    rng = np.random.default_rng(seed)
    _, _, xv_stream, _ = mock_stream_utils.create_stream(
        mock_stream_utils.create_initial_condition_fardal15,
        rng,
        time_total, n_particles,
        pot_host, prog_w0, mass_sat,
        pot_sat=pot_sat
    )

    # convert to observables
    ra, dec, vlos, pmra_o, pmdec_o, dist = coordinate_utils.get_observed_coords(xv_stream)

    return QTable({
        "RA":    ra * u.deg,
        "DEC":   dec * u.deg,
        "PMRA":  pmra_o * u.mas/u.yr,
        "PMDEC": pmdec_o * u.mas/u.yr,
        "VLOS":  vlos * u.km/u.s,
        "DIST":  dist * u.kpc,
        "X":     xv_stream[:, 0] * u.kpc,
        "Y":     xv_stream[:, 1] * u.kpc,
        "Z":     xv_stream[:, 2] * u.kpc,
        "Vx":    xv_stream[:, 3] * u.km/u.s,
        "Vy":    xv_stream[:, 4] * u.km/u.s,
        "Vz":    xv_stream[:, 5] * u.km/u.s,
    })

def overlay_plot(sim_tables, labels, gc_name, save_path, gc_row=None, show_desi=True):
    plt.figure(figsize=(9.5, 7.5))
    ax = plt.gca()

    # plot each simulated stream
    scatter_handles = []
    for tab, lab in zip(sim_tables, labels):
        ra  = tab["RA"].to_value(u.deg)
        dec = tab["DEC"].to_value(u.deg)
        sc = ax.scatter(ra, dec, s=6, alpha=0.5, lw=0, label=lab, zorder=2)  # a tad larger points
        scatter_handles.append(sc)

    plot_desi_region(ax=ax, color="lightgrey")

    # GC marker (from table)
    gc_ra  = float(getattr(gc_row["RA"],  "value", gc_row["RA"]))
    gc_dec = float(getattr(gc_row["DEC"], "value", gc_row["DEC"]))
    ax.scatter(gc_ra, gc_dec, s=160, marker='*', facecolor='gold', edgecolor='k',
                linewidths=0.9, zorder=5)
    ax.annotate(gc_name, (gc_ra, gc_dec), xytext=(8, 8), textcoords='offset points',
                fontsize=11, color='k', weight='bold', zorder=6)

    

    pal_5_members = "notebooks/catalog/Pal5_stream_data_pm_rv_feh.fits"
    # Load TARGET_RA and TARGET_DEC columns from the FITS file
    data = Table.read(pal_5_members)
    # filter with only FEH_CORRECTED 
    data_feh = data[data['FEH_CORRECTED'].mask == False]
    ra_data = data_feh['ra']
    dec_data = data_feh['dec']
    ax.scatter(ra_data, dec_data, s=20, color='red', marker='o', edgecolor='k',
                label='Pal 5 members', zorder=4)
    
    pal5_handle = Line2D(
            [0], [0],
            marker='o', linestyle='None',
            markersize=8,
            markerfacecolor='red',
            markeredgecolor='k',
            label='Pal 5 Stream DESI RV + FE/H'
        )

        
    # axes cosmetics
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(f"{gc_name}: Simulated stream — potential comparison", fontsize=15, pad=10)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25)
            

    # limits (include GC point if present)
    all_ra  = np.concatenate([t["RA"].to_value(u.deg)  for t in sim_tables])
    all_dec = np.concatenate([t["DEC"].to_value(u.deg) for t in sim_tables])
    if gc_row is not None:
        all_ra  = np.concatenate([all_ra,  [gc_ra]])
        all_dec = np.concatenate([all_dec, [gc_dec]])
    pad_ra  = 0.03 * (np.nanmax(all_ra)  - np.nanmin(all_ra)  + 1e-6)
    pad_dec = 0.03 * (np.nanmax(all_dec) - np.nanmin(all_dec) + 1e-6)
    ax.set_xlim(237,222)
    ax.set_ylim(-6,6)

    # BIG legend swatches
    # build custom legend entries so we can control marker size
    legend_handles = []
    for sc, lab in zip(scatter_handles, labels):
        color = sc.get_facecolor()[0]  # RGBA tuple
        legend_handles.append(
            Line2D([0], [0], marker='o', linestyle='None', markersize=10,  # bigger dots
                   markerfacecolor=color, markeredgecolor='none', label=lab)
        )
    # add GC star entry
    legend_handles.append(
        Line2D([0], [0], marker='*', linestyle='None', markersize=14,
               markerfacecolor='gold', markeredgecolor='k', label='GC position')
    )
    legend_handles.append(pal5_handle)

    ax.legend(handles=legend_handles, loc="upper right", fontsize=10, frameon=False, ncol=1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[v] saved overlay plot: {save_path}")

def simulate_for_gc(gc_row, gc_name, pot_files, outdir, n_particles, n_orbits, seed, save_fits):
    pot_paths = []
    pot_labels = []
    for pf in pot_files:
        pf = pf.strip()
        # Accept both "LMCVasiliev24" and "LMCVasiliev24.py"
        if pf in ("LMCVasiliev24", "LMCVasiliev24.py"):
            pot_paths.append("LMCVasiliev24")       # sentinel for the special module
            pot_labels.append("LMCVasiliev24")
            continue

        # Otherwise, expect a standard INI under potentials/
        ppath = os.path.join("potentials", pf)
        if not os.path.isfile(ppath):
            raise FileNotFoundError(f"Potential file not found: {ppath}")
        pot_paths.append(ppath)
        pot_labels.append(os.path.splitext(pf)[0])

    sim_tables = []
    for ppath, plab in zip(pot_paths, pot_labels):
        print(f"  - Simulating for potential: {plab}")
        tab = simulate_stream_for_potential(
            gc_row=gc_row,
            pot_ini_path=ppath,   # may be 'LMCVasiliev24' (sentinel) or a real ini path
            n_particles=n_particles,
            n_orbits=n_orbits,
            seed=seed
        )
        sim_tables.append(tab)
        if save_fits:
            fits_dir = os.path.join(outdir, "fits")
            os.makedirs(fits_dir, exist_ok=True)
            out_fits = os.path.join(fits_dir, f"simulated_stream_{gc_name}_{plab}.fits")
            tab.write(out_fits, overwrite=True)
            print(f"    [v] wrote FITS: {out_fits}")

    overlay_path = os.path.join(outdir, f"{gc_name}_simulated_stream_overlay.png")
    overlay_plot(sim_tables, pot_labels, gc_name, overlay_path, gc_row=gc_row)

def main():
    ap = argparse.ArgumentParser(description="Simulate GC streams under multiple MW potentials and overlay the results.")
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--gc", help="Globular cluster name (e.g., Pal_5)")
    mx.add_argument("--all", action="store_true", help="Run for all clusters in the GC parameter table")

    ap.add_argument("--n-particles", type=int, default=10000, help="Number of stream particles (default: 10000)")
    ap.add_argument("--n-orbits", type=int, default=3, help="Number of orbits to integrate backward (default: 3)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    ap.add_argument(
        "--potentials",
        type=str,
        default=",".join(ALL_POTENTIALS),
        help="Comma-separated list of INI files under potentials/ (default: all)"
    )
    ap.add_argument("--save-fits", action="store_true", help="Also save one FITS per potential")
    ap.add_argument("--outdir", default="pot_compare", help="Output directory for PNGs/FITS (default: pot_compare)")
    args = ap.parse_args()

    t0 = time.time()

    gc_param_path = "data/mw_gc_parameters_orbital_structural_time.ecsv"
    gc_all = QTable.read(gc_param_path)

    pot_files = [p.strip() for p in args.potentials.split(",") if p.strip()]

    # ensure pot_compare/ exists up front
    os.makedirs(args.outdir, exist_ok=True)

    if args.all:
        clusters = [str(x) for x in gc_all["Cluster"]]
        seen = set()
        clusters_unique = []
        for c in clusters:
            if c not in seen:
                seen.add(c)
                clusters_unique.append(c)

        print(f"Running in --all mode for {len(clusters_unique)} clusters.")
        for idx, cname in enumerate(clusters_unique, 1):
            print(f"\n[{idx}/{len(clusters_unique)}] Cluster: {cname}")
            sel = gc_all[gc_all["Cluster"] == cname]
            if len(sel) == 0:
                print(f"  ! Skipping {cname}: not found in table")
                continue
            gc_row = sel[0]
            # pass the flat outdir (pot_compare/) — no per-GC subfolders for PNGs
            simulate_for_gc(
                gc_row=gc_row,
                gc_name=cname,
                pot_files=pot_files,
                outdir=args.outdir,
                n_particles=args.n_particles,
                n_orbits=args.n_orbits,
                seed=args.seed,
                save_fits=args.save_fits
            )
    else:
        cname = args.gc
        sel = gc_all[gc_all["Cluster"] == cname]
        if len(sel) == 0:
            raise ValueError(f"Cluster '{cname}' not found in {gc_param_path}")
        gc_row = sel[0]
        simulate_for_gc(
            gc_row=gc_row,
            gc_name=cname,
            pot_files=pot_files,
            outdir=args.outdir,
            n_particles=args.n_particles,
            n_orbits=args.n_orbits,
            seed=args.seed,
            save_fits=args.save_fits
        )

    print(f"\nDone. Total time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
