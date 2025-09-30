import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib.patches import Circle
from concave_hull import concave_hull_indexes


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
    import astropy.units as u
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import os

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
