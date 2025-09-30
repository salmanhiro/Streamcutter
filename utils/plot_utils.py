import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib.patches import Circle
from concave_hull import concave_hull_indexes
import astropy.units as u
import os

def plot_desi_region(
    tiles_path='data/tiles-main.ecsv',
    length_threshold=5,
    color='black',
    ax=None
):
    tiles = Table.read(tiles_path, format='ascii.ecsv')
    program_tiles = tiles[tiles['PROGRAM'] != "BACKUP"]
    tiles_desi = program_tiles[program_tiles['IN_DESI'] == True]

    coords_1 = np.column_stack((
        tiles_desi[(tiles_desi["RA"] < 100) & (tiles_desi["DEC"] < 40)]['RA'],
        tiles_desi[(tiles_desi["RA"] < 100) & (tiles_desi["DEC"] < 40)]['DEC']
    ))
    coords_2 = np.column_stack((
        tiles_desi[(tiles_desi["RA"] < 300) & (tiles_desi["RA"] > 80)]['RA'],
        tiles_desi[(tiles_desi["RA"] < 300) & (tiles_desi["RA"] > 80)]['DEC']
    ))
    coords_3 = np.column_stack((
        tiles_desi[(tiles_desi["RA"] > 300) & (tiles_desi["DEC"] < 40)]['RA'],
        tiles_desi[(tiles_desi["RA"] > 300) & (tiles_desi["DEC"] < 40)]['DEC']
    ))

    def _plot_hull(coords):
        if len(coords) < 3:
            return
        x, y = coords[:, 0], coords[:, 1]
        idxes = concave_hull_indexes(coords, length_threshold=length_threshold)
        for f, t in zip(idxes, np.roll(idxes, -1)):
            ax.plot([x[f], x[t]], [y[f], y[t]], color=color, alpha=0.7)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    _plot_hull(coords_1)
    _plot_hull(coords_2)
    _plot_hull(coords_3)

    return ax


def plot_ls_region(
    tiles_path='data/tiles-main.ecsv',
    length_threshold=5,
    color='red',
    ax=None
):
    tiles = Table.read(tiles_path, format='ascii.ecsv')
    program_tiles = tiles[tiles['PROGRAM'] != "BACKUP"]

    coords_1 = np.column_stack((
        program_tiles[(program_tiles["RA"] < 97) & (program_tiles["DEC"] < 35)]['RA'],
        program_tiles[(program_tiles["RA"] < 97) & (program_tiles["DEC"] < 35)]['DEC']
    ))
    coords_2 = np.column_stack((
        program_tiles[(program_tiles["RA"] < 300) & (program_tiles["RA"] > 90)]['RA'],
        program_tiles[(program_tiles["RA"] < 300) & (program_tiles["RA"] > 90)]['DEC']
    ))
    coords_3 = np.column_stack((
        program_tiles[(program_tiles["RA"] > 300) & (program_tiles["DEC"] < 40)]['RA'],
        program_tiles[(program_tiles["RA"] > 300) & (program_tiles["DEC"] < 40)]['DEC']
    ))

    def _plot_hull(coords):
        if len(coords) < 3:
            return
        x, y = coords[:, 0], coords[:, 1]
        idxes = concave_hull_indexes(coords, length_threshold=length_threshold)
        for f, t in zip(idxes, np.roll(idxes, -1)):
            ax.plot([x[f], x[t]], [y[f], y[t]], color=color, alpha=0.7)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    _plot_hull(coords_1)
    _plot_hull(coords_2)
    _plot_hull(coords_3)

    return ax

def plot_gc_stream_and_tidal_radius(gc_name, gc_ra, gc_dec, gc_rt, gc_rsun, stream_data, ax=None):
    rt_ang = (np.arctan((gc_rt.to(u.kpc) / gc_rsun).value) * u.rad)
    radius_deg = rt_ang.to(u.deg).value

    print(f"Tidal radius ≈ {rt_ang.to(u.arcmin):.2f}")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 15))
        created_fig = True

    plot_desi_region(ax=ax, length_threshold=5, color="black")

    ax.scatter(stream_data["RAdeg"], stream_data["DEdeg"],
               alpha=0.95, s=10, label="GC Streamfinder (Ibata+23)")

    gc_circle = Circle((gc_ra, gc_dec), radius_deg,
                       edgecolor="crimson", facecolor="none",
                       lw=1.8, zorder=6, label="GC tidal radius (r_t)")
    ax.add_patch(gc_circle)

    ax.scatter([gc_ra], [gc_dec], s=25, color="crimson", zorder=7)
    ax.annotate(f"{gc_name} (r_t ≈ {rt_ang.to(u.arcmin).value:.1f}′)",
                (gc_ra, gc_dec), xytext=(5, 5), textcoords="offset points",
                fontsize=9, color="crimson")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(f"{gc_name} stream track over DESI region")
    ax.set_xlim(240, 220)
    ax.set_ylim(-10, 10)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower center", ncol=2, fontsize=9)

    # Save figure if created here
    if created_fig:
        out_dir = "figures"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{gc_name}_stream_desi.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[v] Saved stream plot to: {out_path}")
        plt.close(fig)

    return ax

def plot_matched_streamfinder_vs_sim(
    sf_data,
    sim_stream_tab,
    idx_sim_box,
    mask_boxcut,
    gc_ra,
    gc_dec,
    gc_rt,
    gc_rsun,
    gc_name="GC"
):
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_desi_region(ax=ax, length_threshold=5, color="black")
    rt_ang = (np.arctan((gc_rt.to(u.kpc) / gc_rsun).value) * u.rad)
    radius_deg = rt_ang.to(u.deg).value

    # Observed and simulated matched coordinates
    ra_obs = sf_data["RAdeg"][mask_boxcut].astype(float) * u.deg
    dec_obs = sf_data["DEdeg"][mask_boxcut].astype(float) * u.deg

    ra_sim = sim_stream_tab["RA"][idx_sim_box].astype(float)
    dec_sim = sim_stream_tab["DEC"][idx_sim_box].astype(float)

    # Plot STREAMFINDER matched
    ax.scatter(
        ra_obs, dec_obs,
        s=22, facecolors='none', edgecolors='tab:orange', linewidths=1.2,
        label=f"STREAMFINDER matched (n={len(idx_sim_box)})", zorder=3
    )

    # Plot Simulation matched
    ax.scatter(
        ra_sim, dec_sim,
        s=22, facecolors='none', edgecolors='tab:blue', linewidths=1.2,
        label=f"Simulation matched (n={len(idx_sim_box)})", zorder=3
    )

    # Connection lines between pairs
    for ra1, dec1, ra2, dec2 in zip(ra_obs, dec_obs, ra_sim, dec_sim):
        ax.plot([ra1.value, ra2.value], [dec1.value, dec2.value],
                color='gray', alpha=0.4, linewidth=0.8, zorder=2)

    # Tidal radius circle
    gc_circle = Circle(
        (gc_ra, gc_dec), radius_deg,
        edgecolor="crimson", facecolor="none",
        lw=1.8, zorder=6, label="GC tidal radius (r_t)"
    )
    ax.add_patch(gc_circle)

    # Labels and limits
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title("Matched STREAMFINDER vs Simulation Positions")
    ax.legend(loc="best", fontsize=9)

    padding = 1.0
    ax.set_xlim(ra_sim.max().value + padding, ra_sim.min().value - padding)
    ax.set_ylim(dec_sim.min().value - padding, dec_sim.max().value + padding)

    ax.grid(True, alpha=0.35)

    # Save
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{gc_name}_matched_stream_positions.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[v] Saved matched position plot to: {out_path}")

    plt.close(fig)
    return ax

def plot_density_with_sim_stream(
    RV_T,
    sim_stream_tab,
    main_sel,
    gc_df,
    gc_name="GC",
    out_dir="figures",
    bins=(100, 100),
    cmap='plasma',
    cmin=1
):
    # Extract target coordinates
    ra  = np.asarray(RV_T['TARGET_RA'][main_sel],  float)
    dec = np.asarray(RV_T['TARGET_DEC'][main_sel], float)

    # Sim stream extent
    ra_min, ra_max = min(sim_stream_tab["RA"].value), max(sim_stream_tab["RA"].value)
    dec_min, dec_max = min(sim_stream_tab["DEC"].value), max(sim_stream_tab["DEC"].value)

    fig, ax = plt.subplots(figsize=(8, 6))

    h = ax.hist2d(
        ra, dec,
        bins=bins,
        cmap=cmap,
        range=[(ra_min, ra_max), (dec_min, dec_max)],
        cmin=cmin
    )

    # Simulated stream overlay
    ax.scatter(sim_stream_tab["RA"], sim_stream_tab["DEC"],
               s=2, c='black', alpha=0.6, lw=0,
               zorder=3, label=f"Sim stream (n={len(sim_stream_tab)})")

    # GC marker
    gc_ra = gc_df["RA"][0]; gc_dec = gc_df["DEC"][0]
    gc_ra_deg  = gc_ra.to_value(u.deg)  if getattr(gc_ra,  "unit", None) else float(gc_ra)
    gc_dec_deg = gc_dec.to_value(u.deg) if getattr(gc_dec, "unit", None) else float(gc_dec)

    ax.scatter(gc_ra_deg, gc_dec_deg, s=80, marker='*', facecolor='crimson',
               edgecolor='k', linewidths=0.8, zorder=5)
    ax.annotate(gc_name, (gc_ra_deg, gc_dec_deg),
                xytext=(6, 6), textcoords='offset points',
                fontsize=9, color='black')

    # Axis and legend
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    cb = fig.colorbar(h[3], ax=ax, label="Counts per bin")
    ax.legend(loc='upper right', fontsize=8, frameon=False)

    ax.invert_xaxis()
    ax.set_xlim(ra_max, ra_min)
    ax.set_ylim(dec_min, dec_max)
    ax.set_title(f"Simulated {gc_name} stream in DESI MWS tiles background")
    plt.tight_layout()

    # Save plot
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{gc_name}_stream_density_overlay.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[v] Saved density plot to: {out_path}")

    plt.close(fig)
    return ax
