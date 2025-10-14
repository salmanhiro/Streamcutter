import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib.patches import Circle
from concave_hull import concave_hull_indexes
import astropy.units as u
import os
from scipy.stats import norm
from matplotlib.colors import Normalize

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
    
    pad = 0.5  # degrees

    ra_min = min(stream_data["RAdeg"].min(), gc_ra) - pad
    ra_max = max(stream_data["RAdeg"].max(), gc_ra) + pad

    dec_min = min(stream_data["DEdeg"].min(), gc_dec) - pad
    dec_max = max(stream_data["DEdeg"].max(), gc_dec) + pad

    ax.set_xlim(ra_max, ra_min)  # invert x-axis
    ax.set_ylim(dec_min, dec_max)


    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower center", ncol=2, fontsize=9)

    # Save figure if created here
    if created_fig:
        out_dir = f"results/{gc_name}"
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
    out_dir = f"results/{gc_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{gc_name}_matched_stream_positions.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[v] Saved matched position plot to: {out_path}")

    plt.close(fig)
    return ax


def plot_overlay_streamfinder_vs_sim(
    sf_data,
    sim_stream_tab,
    idx_sim_box,
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
    ra_obs = sf_data["RAdeg"].astype(float) * u.deg
    dec_obs = sf_data["DEdeg"].astype(float) * u.deg

    ra_sim = sim_stream_tab["RA"].astype(float)
    dec_sim = sim_stream_tab["DEC"].astype(float)

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
    out_dir = f"results/{gc_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{gc_name}_sim_sf.png")
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
    out_dir = f"results/{gc_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"results/{gc_name}/{gc_name}_density_with_sim_stream.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[v] Saved density plot to: {out_path}")

    plt.close(fig)
    return ax



def _fit_and_plot_hist(ax, data, bins, color, xlabel, title):
    data = data[np.isfinite(data)]
    if len(data) == 0:
        ax.set_visible(False)
        return None, None

    mu, sigma = norm.fit(data)
    n, bins, _ = ax.hist(data, bins=bins, density=True, alpha=0.6, color=color, label='Histogram')

    x = np.linspace(bins[0], bins[-1], 500)
    p = norm.pdf(x, mu, sigma)
    ax.plot(x, p, 'k--', linewidth=2, label=f'Gaussian Fit\nμ = {mu:.2f}, σ = {sigma:.2f}')
    ax.axvline(mu, color='red', linestyle='--', linewidth=1.5)
    ax.text(mu, max(p)*0.9, f'Peak: {mu:.2f}', color='red', ha='center')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability Density')
    ax.set_title(title)
    ax.legend()
    return mu, sigma
def plot_candidate_histograms(candidates_fm, candidates_rv, gc_name, out_dir, gc_df=None):
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    #  Radial velocity histogram 
    print(f"Lowest VRAD_CORRECTED: {np.nanmin(candidates_rv['VRAD_CORRECTED']):.2f}")
    print(f"Highest VRAD_CORRECTED: {np.nanmax(candidates_rv['VRAD_CORRECTED']):.2f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    _fit_and_plot_hist(
        ax, candidates_rv['VRAD_CORRECTED'], bins=30, color='blue',
        xlabel='Corrected Radial Velocity (km/s)',
        title=f'Radial Velocity Distribution for {gc_name} Candidates'
    )
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{gc_name}_candidates_rv_histogram.png", dpi=300)
    plt.close()

    #  Proper motion histograms 
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    _fit_and_plot_hist(
        ax[0], candidates_fm['PMRA'], bins=30, color='green',
        xlabel='PMRA (mas/yr)',
        title=f'PMRA Distribution for {gc_name} Candidates'
    )
    _fit_and_plot_hist(
        ax[1], candidates_fm['PMDEC'], bins=30, color='orange',
        xlabel='PMDEC (mas/yr)',
        title=f'PMDEC Distribution for {gc_name} Candidates'
    )
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{gc_name}_candidates_pm_histogram.png", dpi=300)
    plt.close()

    #  Sky distribution plot 
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(candidates_fm['TARGET_RA'], candidates_fm['TARGET_DEC'],
               s=5, color='red', label='Selected Candidates')

    # Optional GC position overlay
    if gc_df is not None:
        gc_ra = gc_df['RA'][0]
        gc_dec = gc_df['DEC'][0]
        ax.scatter(gc_ra, gc_dec, s=80, color='black', marker='*', label='GC Center (from catalog)', zorder=5)

    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('DEC (deg)')
    ax.set_title(f'Sky Distribution of {gc_name} Candidates')
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{gc_name}_candidates_sky_distribution.png", dpi=300)
    plt.close()

    #  Metallicity histogram 
    fig, ax = plt.subplots(figsize=(8, 6))
    _fit_and_plot_hist(
        ax, candidates_rv['FEH_CORRECTED'], bins=30, color='purple',
        xlabel='Corrected [Fe/H]',
        title=f'Metallicity Distribution for {gc_name} Candidates'
    )
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{gc_name}_candidates_feh_histogram.png", dpi=300)
    plt.close()


def plot_subplots_sf_vs_sim_Rsun(
    sf_data,            # columns: RAdeg, DEdeg, Rsun
    sim_stream_tab,     # columns: RA, DEC, Rsun
    idx_sim_box,        # indices into sim_stream_tab to show
    gc_ra,              # deg (float or Quantity)
    gc_dec,             # deg (float or Quantity)
    gc_rt,              # length Quantity (e.g., kpc)
    gc_rsun,            # length Quantity (e.g., kpc)
    gc_name="GC",
    cmap="viridis",
    point_size=18,
    padding=1.0,
):
    # --- pick sim subset ---
    sim_sel = sim_stream_tab[idx_sim_box]

    # --- exactly as requested ---
    ra_sf,  dec_sf,  r_sf  =  sf_data["RAdeg"], sf_data["DEdeg"], sf_data["Rsun"]
    ra_sim, dec_sim, r_sim =  sim_sel["RA"],   sim_sel["DEC"],   sim_sel["DIST"]

    # Cast to plain floats for plotting/stats (keeps it robust if columns have units)
    ra_sf  = np.asarray(ra_sf,  dtype=float)
    dec_sf = np.asarray(dec_sf, dtype=float)
    r_sf   = np.asarray(r_sf,   dtype=float)

    ra_sim  = np.asarray(ra_sim,  dtype=float)
    dec_sim = np.asarray(dec_sim, dtype=float)
    r_sim   = np.asarray(r_sim,   dtype=float)

    # Shared color normalization for Rsun
    all_r = np.concatenate([r_sf[np.isfinite(r_sf)], r_sim[np.isfinite(r_sim)]])
    vmin = np.nanpercentile(all_r, 5) if all_r.size else 0.0
    vmax = np.nanpercentile(all_r, 95) if all_r.size else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = (float(np.nanmin(all_r)), float(np.nanmax(all_r))) if all_r.size else (0.0, 1.0)
        if vmin == vmax:
            vmin, vmax = vmin - 1e-3, vmin + 1e-3
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True, constrained_layout=True)

    # Tidal radius circle on both panels
    radius_deg = (np.arctan((gc_rt.to(u.kpc) / gc_rsun).value) * u.rad).to_value(u.deg)
    gc_ra_f = float(getattr(gc_ra, "value", gc_ra))
    gc_dec_f = float(getattr(gc_dec, "value", gc_dec))
    for ax in (ax1, ax2):
        ax.add_patch(Circle((gc_ra_f, gc_dec_f), radius_deg,
                            edgecolor="crimson", facecolor="none", lw=1.8, zorder=6,
                            label="GC tidal radius (r_t)"))

    # Scatter plots (colored by Rsun)
    ax1.scatter(ra_sf,  dec_sf,  c=r_sf,  s=point_size, cmap=cmap, norm=norm, linewidths=0, zorder=3)
    ax1.set_title(f"STREAMFINDER (n={len(sf_data)})")

    ax2.scatter(ra_sim, dec_sim, c=r_sim, s=point_size, cmap=cmap, norm=norm, linewidths=0, zorder=3)
    ax2.set_title(f"Simulation (n={len(sim_sel)})")

    # Labels, limits, reverse RA like sky
    for ax in (ax1, ax2):
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
        ax.grid(True, alpha=0.35)

    ra_min = np.nanmin(np.concatenate([ra_sf, ra_sim]))
    ra_max = np.nanmax(np.concatenate([ra_sf, ra_sim]))
    dec_min = np.nanmin(np.concatenate([dec_sf, dec_sim]))
    dec_max = np.nanmax(np.concatenate([dec_sf, dec_sim]))
    ax1.set_xlim(ra_max + padding, ra_min - padding)  # reverse RA
    ax1.set_ylim(dec_min - padding, dec_max + padding)
    ax1.legend(loc="best", fontsize=9)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    cbar.set_label("Rsun (heliocentric distance) [kpc]")

    # Save
    out_dir = f"results/{gc_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{gc_name}_sf_vs_sim_Rsun_subplots.png")
    fig.suptitle("STREAMFINDER vs Simulation — colored by Rsun")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[v] Saved subplot figure to: {out_path}")

    return fig, (ax1, ax2)

from matplotlib.colors import LogNorm

def plot_fm_density(
    FM_T_pos_sel,                 # already filtered table
    ra_col="TARGET_RA",
    dec_col="TARGET_DEC",
    gridsize=120,
    cmap="viridis",
    padding=1.0,
    method="hexbin",              # "hexbin" or "hist2d"
    save_path=None,
    title=None,
):
    ra  = np.asarray(FM_T_pos_sel[ra_col], dtype=float)
    dec = np.asarray(FM_T_pos_sel[dec_col], dtype=float)

    if ra.size == 0:
        print("[density] No rows to plot.")
        return None, None

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

    if method == "hexbin":
        m = ax.hexbin(ra, dec, gridsize=gridsize, bins="log", cmap=cmap)
        cbar = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    else:
        h = ax.hist2d(ra, dec, bins=gridsize, norm=LogNorm(), cmap=cmap)
        cbar = fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)

    cbar.set_label("Counts (log scale)")
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(title or f"DESI FM number density (n={ra.size})")

    # reverse RA like a sky plot
    ax.set_xlim(np.nanmax(ra) + padding, np.nanmin(ra) - padding)
    ax.set_ylim(np.nanmin(dec) - padding, np.nanmax(dec) + padding)
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[v] saved: {save_path}")

    return fig, ax
