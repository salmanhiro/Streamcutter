import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

def select_gc_region_feh(gc_df, FM_T_sel, RV_T_sel, radius_deg=None, use_tidal_radius=True, plot=False):
    """
    Select stars around GC position and return valid FEH_CORRECTED values.

    Parameters
    ----------
    gc_df : astropy.Table
        Table with 1-row cluster info: must contain 'RA', 'DEC', 'rt' (pc), 'Rsun' (kpc), 'Cluster'
    FM_T_sel : astropy.Table
        Table with at least 'TARGET_RA', 'TARGET_DEC'
    RV_T_sel : astropy.Table
        Table with at least 'FEH_CORRECTED', same order as FM_T_sel
    radius_deg : float, optional
        Radius in degrees to define GC region (overrides tidal radius if provided)
    use_tidal_radius : bool, default=True
        Whether to use rt/Rsun as tidal radius (converted to degrees)
    plot : bool, default=False
        If True, plots histogram of [Fe/H] distribution

    Returns
    -------
    feh_values : np.ndarray
        Array of [Fe/H] values within the GC region (NaN removed)
    """

    # GC position
    gc_ra = gc_df["RA"][0]
    gc_dec = gc_df["DEC"][0]
    gc_name = gc_df['Cluster'][0]
    gc_rt = gc_df["rt"][0]
    gc_rsun = gc_df["Rsun"][0]

    # Define separation threshold
    if radius_deg is not None:
        sep_max = radius_deg * u.deg
    elif use_tidal_radius:
        rt_ang = (np.arctan((gc_rt.to(u.kpc) / gc_rsun).value) * u.rad)
        radius_deg = rt_ang.to(u.deg).value   # for plotting
        sep_max = radius_deg * u.deg
    else:
        raise ValueError("Either radius_deg must be provided or use_tidal_radius=True.")

    # SkyCoords
    gc_coord = SkyCoord(ra=gc_ra, dec=gc_dec)
    star_coords = SkyCoord(ra=FM_T_sel['TARGET_RA'] * u.deg,
                           dec=FM_T_sel['TARGET_DEC'] * u.deg)

    # Compute separation and apply mask
    sep = gc_coord.separation(star_coords)
    mask_within = sep < sep_max

    # Apply mask to RV_T_sel
    stars_near_gc = RV_T_sel[mask_within]

    # Remove invalid FEH
    mask_valid = np.isfinite(stars_near_gc['FEH_CORRECTED'])
    feh_values = stars_near_gc['FEH_CORRECTED'][mask_valid]

    print(f"[v] Found {len(feh_values)} stars within {sep_max:.3f} of {gc_name} with valid [Fe/H]")

    # Optional: Plot metallicity histogram
    if plot:
        plt.figure(figsize=(8, 5))
        plt.hist(feh_values, bins=25, color='purple', alpha=0.7)
        plt.axvline(np.median(feh_values), color='red', linestyle='--', label=f"Median: {np.median(feh_values):.2f}")
        plt.title(f"[Fe/H] Distribution near {gc_name}")
        plt.xlabel("[Fe/H] (corrected)")
        plt.ylabel("Star count")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{gc_name}/{gc_name}_gc_body_feh_distribution.png")

        # Plot 2d histogram of RA/DEC around GC
        plt.figure(figsize=(6, 6))
        plt.hist2d(FM_T_sel['TARGET_RA'][mask_within], FM_T_sel['TARGET_DEC'][mask_within],
                   bins=50, range=[[gc_ra.value - radius_deg, gc_ra.value + radius_deg],
                                   [gc_dec.value - radius_deg, gc_dec.value + radius_deg]],
                   cmap='Blues')
        plt.colorbar(label='Star count')
        plt.scatter(gc_ra, gc_dec, color='red', marker='*', s=200, label=gc_name)
        plt.title(f"Stars around {gc_name} within {sep_max:.3f}")
        plt.xlabel("RA (deg)")
        plt.ylabel("DEC (deg)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/{gc_name}/{gc_name}_spatial_distribution.png")

    # also plot region


    return feh_values
