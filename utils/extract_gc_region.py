import os
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def select_gc_region_feh(gc_df, FM_T_sel, RV_T_sel, radius_deg=None,
                         use_tidal_radius=True, plot=False,
                         feh_hist_bins=25, feh_cmap='coolwarm',
                         hexbin_gridsize=60, hexbin_mincnt=4):
    """
    Select stars around GC position and return valid FEH_CORRECTED values.
    Also:
      • Saves density-normalized [Fe/H] histogram with bars colored by metallicity.
      • Saves spatial hexbin map colored by median [Fe/H] in each bin (extended +1° FoV).
    """

    # ---- GC info (robust to Quantity/float) ----
    gc_name = str(gc_df['Cluster'][0])
    gc_rt   = gc_df['rt'][0]   # pc
    gc_rsun = gc_df['Rsun'][0] # kpc

    def _to_deg(x):
        try:
            return float(u.Quantity(x).to_value(u.deg))
        except Exception:
            return float(x)

    gc_ra_deg  = _to_deg(gc_df['RA'][0])
    gc_dec_deg = _to_deg(gc_df['DEC'][0])

    # ---- radius / separation threshold ----
    if radius_deg is not None:
        base_radius_deg = float(radius_deg)
    elif use_tidal_radius:
        theta = np.arctan((u.Quantity(gc_rt).to(u.kpc) / u.Quantity(gc_rsun)).value) * u.rad
        base_radius_deg = theta.to_value(u.deg)
    else:
        raise ValueError("Either provide radius_deg or set use_tidal_radius=True.")

    sep_max = base_radius_deg * u.deg
    ext_radius_deg = base_radius_deg + 1.0  # +1 degree FoV

    gc_coord    = SkyCoord(ra=gc_ra_deg * u.deg,  dec=gc_dec_deg * u.deg)
    star_coords = SkyCoord(ra=FM_T_sel['TARGET_RA'] * u.deg,
                           dec=FM_T_sel['TARGET_DEC'] * u.deg)

    sep = gc_coord.separation(star_coords)
    mask_within_base = sep < sep_max               # for FEH selection (cluster body)
    mask_within_ext  = sep < (ext_radius_deg * u.deg)  # for extended spatial maps

    stars_near_gc = RV_T_sel[mask_within_base]
    mask_valid = np.isfinite(stars_near_gc['FEH_CORRECTED'])
    feh_values = np.array(stars_near_gc['FEH_CORRECTED'][mask_valid])

    print(f"[v] Found {len(feh_values)} stars within {base_radius_deg:.3f}° of {gc_name} with valid [Fe/H].")
    hexbin_global_median = np.nan
    if plot:
        outdir = f"results/{gc_name}"
        os.makedirs(outdir, exist_ok=True)

        ra_all  = np.asarray(FM_T_sel['TARGET_RA'])[mask_within_ext]
        dec_all = np.asarray(FM_T_sel['TARGET_DEC'])[mask_within_ext]
        feh_all = np.asarray(RV_T_sel['FEH_CORRECTED'])[mask_within_ext]

        m = np.isfinite(ra_all) & np.isfinite(dec_all) & np.isfinite(feh_all)
        ra_all, dec_all, feh_all = ra_all[m], dec_all[m], feh_all[m]

        fig, ax = plt.subplots(figsize=(7, 6))

        # robust color limits (avoid outliers crushing the scale)
        if feh_all.size >= 50:
            vmin, vmax = np.nanpercentile(feh_all, [5, 95])
        elif feh_all.size > 0:
            vmin, vmax = np.nanmin(feh_all), np.nanmax(feh_all)
        else:
            vmin, vmax = -2.5, 0.5  # fallback GC-ish range

        sc = ax.scatter(
            ra_all, dec_all,
            c=feh_all,
            s=9,                 # point size; bump if sparse
            cmap='coolwarm',     # low Fe/H = blue, high Fe/H = red
            vmin=vmin, vmax=vmax,
            alpha=0.85,
            edgecolors='none'
        )
        fig.colorbar(sc, ax=ax, label='[Fe/H]')

        # GC center and circles
        ax.scatter([gc_ra_deg], [gc_dec_deg], marker='*', s=180, color='k', zorder=3, label=gc_name)
        ax.add_patch(Circle((gc_ra_deg, gc_dec_deg), base_radius_deg, fill=False, ec='k', lw=1.8,
                            label=f"GC radius {base_radius_deg:.2f}°"))
        ax.add_patch(Circle((gc_ra_deg, gc_dec_deg), ext_radius_deg, fill=False, ec='k', ls='--', lw=1.2,
                            label=f"+1° field ({ext_radius_deg:.2f}°)"))

        # plot window (astronomer style: RA increasing to the left)
        ax.set_xlim(gc_ra_deg + ext_radius_deg, gc_ra_deg - ext_radius_deg)
        ax.set_ylim(gc_dec_deg - ext_radius_deg, gc_dec_deg + ext_radius_deg)

        ax.set_title(f"[Fe/H] scatter around {gc_name} (+1° FoV)")
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("DEC (deg)")
        ax.legend(loc='upper right', framealpha=0.85)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(f"{outdir}/{gc_name}_spatial_scatter_feh.png", dpi=150)
        plt.close(fig)


        # (2) Spatial hexbin colored by median [Fe/H] (extended +1°)
        # To select central metallicities better
            

        ra_all  = np.asarray(FM_T_sel['TARGET_RA'])[mask_within_ext]
        dec_all = np.asarray(FM_T_sel['TARGET_DEC'])[mask_within_ext]
        feh_all = np.asarray(RV_T_sel['FEH_CORRECTED'])[mask_within_ext]
        m = np.isfinite(ra_all) & np.isfinite(dec_all) & np.isfinite(feh_all)
        ra_all, dec_all, feh_all = ra_all[m], dec_all[m], feh_all[m]

        plt.figure(figsize=(6.8, 6.3))

        # Robust color limits for [Fe/H]
        if feh_all.size >= 50:
            vmin2, vmax2 = np.nanpercentile(feh_all, [5, 95])
        elif feh_all.size > 0:
            vmin2, vmax2 = np.nanmin(feh_all), np.nanmax(feh_all)
        else:
            vmin2, vmax2 = -2.5, 0.5

        hb = plt.hexbin(
            ra_all, dec_all,
            C=feh_all,
            reduce_C_function=np.nanmedian,  # color = median [Fe/H] per hex
            gridsize=hexbin_gridsize,
            mincnt=hexbin_mincnt,           # require at least N stars per hex
            extent=[gc_ra_deg - ext_radius_deg, gc_ra_deg + ext_radius_deg,
                    gc_dec_deg - ext_radius_deg, gc_dec_deg + ext_radius_deg],
            cmap=feh_cmap, vmin=vmin2, vmax=vmax2
        )
        hex_vals = np.asarray(hb.get_array())
        # (hb may return a masked array; use nanmedian to be safe)
        if hex_vals.size:
            hexbin_global_median = float(np.nanmedian(hex_vals))
        else:
            hexbin_global_median = np.nan

        print(f"[v] Global median of hexbin median [Fe/H]: {hexbin_global_median:.3f}")

        cbar = plt.colorbar(hb)
        cbar.set_label('Median [Fe/H]')

        ax = plt.gca()
        # GC center and radii
        ax.scatter([gc_ra_deg], [gc_dec_deg], color='k', marker='*', s=180, zorder=3, label=gc_name)
        ax.add_patch(Circle((gc_ra_deg, gc_dec_deg), base_radius_deg,
                            fill=False, ec='k', lw=1.8, label=f"GC radius {base_radius_deg:.2f}°"))
        ax.add_patch(Circle((gc_ra_deg, gc_dec_deg), ext_radius_deg,
                            fill=False, ec='k', ls='--', lw=1.2, label=f"+1° field ({ext_radius_deg:.2f}°)"))

        plt.title(f"[Fe/H] map around {gc_name} (hexbin median, +1° FoV)")
        plt.xlabel("RA (deg)")
        plt.ylabel("DEC (deg)")
        plt.legend(loc='upper right', framealpha=0.85)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(f"{outdir}/{gc_name}_spatial_hexbin_median_feh.png", dpi=150)
        plt.close()

        # (Optional) Keep your count heatmap too, if you still want it:
        plt.figure(figsize=(6.6, 6.2))
        plt.hist2d(
            np.asarray(FM_T_sel['TARGET_RA']),
            np.asarray(FM_T_sel['TARGET_DEC']),
            bins=60,
            range=[[gc_ra_deg - ext_radius_deg, gc_ra_deg + ext_radius_deg],
                   [gc_dec_deg - ext_radius_deg, gc_dec_deg + ext_radius_deg]],
            cmap='Blues'
        )
        plt.colorbar(label='Star count')
        ax = plt.gca()
        ax.scatter([gc_ra_deg], [gc_dec_deg], color='crimson', marker='*', s=180, zorder=3, label=gc_name)
        ax.add_patch(Circle((gc_ra_deg, gc_dec_deg), base_radius_deg, fill=False, ec='crimson', lw=2))
        ax.add_patch(Circle((gc_ra_deg, gc_dec_deg), ext_radius_deg, fill=False, ec='black', ls='--', lw=1.5))
        plt.title(f"Star density around {gc_name} (counts, +1° FoV)")
        plt.xlabel("RA (deg)")
        plt.ylabel("DEC (deg)")
        plt.legend(loc='upper right', framealpha=0.8)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(f"{outdir}/{gc_name}_spatial_distribution_ext_plus1deg.png", dpi=150)
        plt.close()

    return feh_values, hexbin_global_median
