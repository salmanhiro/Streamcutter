import numpy as np
import astropy.units as u
import astropy.coordinates as coord
import matplotlib.pyplot as plt
def box_and_cone_cut(streamfinder_gc, sim_stream_tab,
                     pm_sigma=2.0,  # mas/yr
                     sep_max=1.0 * u.deg):
    """
    Apply box (PMRA/PMDEC) and cone (angular separation) cut based on stream simulation.
    
    Parameters
    ----------
    pm_sigma : float
        Use fixed ±pm_sigma (in mas/yr) cut from median PMRA and PMDEC.
    """
    # 1. Proper motion cut (±pm_sigma around the median)
    pmra_med = np.nanmedian(sim_stream_tab["PMRA"])
    pmde_med = np.nanmedian(sim_stream_tab["PMDEC"])

    lower_pmra = pmra_med - pm_sigma
    upper_pmra = pmra_med + pm_sigma
    lower_pmde = pmde_med - pm_sigma
    upper_pmde = pmde_med + pm_sigma

    print(f"PMRA median: {pmra_med:.2f}, PMDEC median: {pmde_med:.2f}")
    print(f"PMRA cut: {lower_pmra:.2f} to {upper_pmra:.2f} mas/yr")
    print(f"PMDEC cut: {lower_pmde:.2f} to {upper_pmde:.2f} mas/yr")

    mask_pm = (
        (streamfinder_gc["pmRA"] * u.mas/u.yr >= lower_pmra) & (streamfinder_gc["pmRA"] * u.mas/u.yr <= upper_pmra) &
        (streamfinder_gc["pmDE"] * u.mas/u.yr >= lower_pmde) & (streamfinder_gc["pmDE"] * u.mas/u.yr <= upper_pmde)
    )
    print("Number after PM cut:", np.sum(mask_pm))

    # 2. Cone cut (angular separation to sim stream)
    sc_obs = coord.SkyCoord(streamfinder_gc["RAdeg"] * u.deg, streamfinder_gc["DEdeg"] * u.deg)
    sc_sim = coord.SkyCoord(sim_stream_tab["RA"], sim_stream_tab["DEC"])
    
    idx, d2d, _ = sc_obs.match_to_catalog_sky(sc_sim)
    sep_ok = (d2d <= sep_max)

    # 3. Combined mask
    keep_mask = mask_pm & sep_ok
    print("Number after PM + cone cut:", np.sum(keep_mask))

    return keep_mask, idx[keep_mask]

def cone_cut(streamfinder_gc, sim_stream_tab,
                     sep_max=1.0 * u.deg):
    """
    Apply cone (angular separation) cut based on stream simulation.
    
    Parameters
    ----------
    pm_sigma : float
        Use fixed ±pm_sigma (in mas/yr) cut from median PMRA and PMDEC.
    """
    # 1. Cone cut (angular separation to sim stream)
    sc_obs = coord.SkyCoord(streamfinder_gc["RAdeg"] * u.deg, streamfinder_gc["DEdeg"] * u.deg)
    sc_sim = coord.SkyCoord(sim_stream_tab["RA"], sim_stream_tab["DEC"])
    
    idx, d2d, _ = sc_obs.match_to_catalog_sky(sc_sim)
    keep_mask = (d2d <= sep_max)

    # 2. Combined mask
    print("Number after PM + cone cut:", np.sum(keep_mask))

    return keep_mask, idx[keep_mask]

def box_xyz_cut(streamfinder_gc, sim_stream_tab,
                       spatial_tol=1.0 * u.kpc):
    """
    Select STREAMFINDER stars that match simulated stream in both:
    - Proper motion (box cut in PMRA/PMDEC)
    - Galactocentric (X,Y,Z) spatial position within a given kpc tolerance.

    Parameters
    ----------
    streamfinder_gc : table
        STREAMFINDER stars with at least 'pmRA', 'pmDE', 'X', 'Y', 'Z' columns.
    sim_stream_tab : table
        Simulated stream table with same columns.
    spatial_tol : Quantity
        3D Cartesian distance tolerance (default: 1 kpc).

    Returns
    -------
    keep_mask : boolean array
        Selection mask on `streamfinder_gc`.
    idx_sim_match : integer array
        Index into `sim_stream_tab` of the closest matched simulation particle per kept star.
    """

    # 2. Physical (X, Y, Z) separation cut
    stream_pos = np.vstack([
        streamfinder_gc["X_gc"] * (u.kpc),
        streamfinder_gc["Y_gc"] * (u.kpc),
        streamfinder_gc["Z_gc"] * (u.kpc)
    ]).T  # shape (N_obs, 3)

    # plot xyz positions of streamfinder stars histogram
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].hist(streamfinder_gc["X_gc"], bins=50, alpha=0.7, label='StreamFinder X', color='blue')
    ax[1].hist(streamfinder_gc["Y_gc"], bins=50, alpha=0.7, label='StreamFinder Y', color='orange')
    ax[2].hist(streamfinder_gc["Z_gc"], bins=50, alpha=0.7, label='StreamFinder Z', color='green')
    ax[0].set_xlabel('X (kpc)')
    ax[1].set_xlabel('Y (kpc)')
    ax[2].set_xlabel('Z (kpc)')
    ax[0].set_ylabel('Number of Stars')
    ax[0].legend()
    plt.suptitle('StreamFinder Stars XYZ Distribution')
    plt.savefig('streamfinder_xyz_distribution.png')


    sim_pos = np.vstack([
        sim_stream_tab["X"],
        sim_stream_tab["Y"],
        sim_stream_tab["Z"],
    ]).T  # shape (N_sim, 3)

    # plot xyz positions of sim stars histogram
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].hist(sim_stream_tab["X"], bins=50, alpha=0.7, label='Sim X', color='blue')
    ax[1].hist(sim_stream_tab["Y"], bins=50, alpha=0.7, label='Sim Y', color='orange')
    ax[2].hist(sim_stream_tab["Z"], bins=50, alpha=0.7, label='Sim Z', color='green')
    ax[0].set_xlabel('X (kpc)')
    ax[1].set_xlabel('Y (kpc)')
    ax[2].set_xlabel('Z (kpc)')
    ax[0].set_ylabel('Number of Stars')
    ax[0].legend()
    plt.suptitle('Simulated Stream Stars XYZ Distribution')
    plt.savefig('simulated_stream_xyz_distribution.png')
    

    # Find nearest simulated stream point for each observation
    from scipy.spatial import cKDTree
    tree = cKDTree(sim_pos)
    dists, idx_sim_match = tree.query(stream_pos)

    print(f"Distance to minimum XYZ point (kpc): {dists.min():.3f}")

    mask_xyz = (dists * u.kpc <= spatial_tol)
    print("Number after XYZ distance cut:", np.sum(mask_xyz))

    # 3. Combined mask
    keep_mask = mask_xyz
    print("Number after XYZ cut:", np.sum(keep_mask))

    return keep_mask, idx_sim_match[keep_mask]


def box_and_cone_cut_DESI(FM_T, sim_stream_tab, gc_df, frac=0.5, sep_max=1.0*u.deg):
    """
    Select stars from FM_T that lie within a box in PM space and within a cone around
    simulated stream stars, but exclude those inside the GC tidal radius.

    Parameters:
    - FM_T: astropy Table of DESI data (must have PMRA, PMDEC, TARGET_RA, TARGET_DEC)
    - sim_stream_tab: simulated stream stars with RA, DEC, PMRA, PMDEC
    - gc_df: 1-row astropy Table with 'RA', 'DEC', 'Rsun' (kpc), 'rt' (pc)
    - frac: fractional width for PM box cut
    - sep_max: maximum angular separation for cone cut (default: 1 deg)

    Returns:
    - keep_mask: boolean mask for FM_T
    - matched_sim_idx: indices into sim_stream_tab (same length as mask sum)
    """
    pmra_med = np.nanmedian(sim_stream_tab["PMRA"])
    pmde_med = np.nanmedian(sim_stream_tab["PMDEC"])
    
    half_pmra = frac * abs(pmra_med)
    half_pmde = frac * abs(pmde_med)
    
    lower_pmra = (pmra_med - half_pmra) 
    upper_pmra = (pmra_med + half_pmra)
    lower_pmde = (pmde_med - half_pmde)
    upper_pmde = (pmde_med + half_pmde)

    mask_pm = (
        (FM_T["PMRA"] * u.mas/u.yr >= lower_pmra) & (FM_T["PMRA"] * u.mas/u.yr <= upper_pmra) &
        (FM_T["PMDEC"] * u.mas/u.yr >= lower_pmde) & (FM_T["PMDEC"] * u.mas/u.yr <= upper_pmde)
    )

    sc_obs = coord.SkyCoord(FM_T["TARGET_RA"] * u.deg, FM_T["TARGET_DEC"] * u.deg)
    sc_sim = coord.SkyCoord(sim_stream_tab["RA"], sim_stream_tab["DEC"])
    idx, d2d, _ = sc_obs.match_to_catalog_sky(sc_sim)
    sep_ok = (d2d <= sep_max)

    # print min and max ra and dec of matching stars
    print(f"Matched stars RA: {FM_T['TARGET_RA'][mask_pm & sep_ok].min()} to {FM_T['TARGET_RA'][mask_pm & sep_ok].max()}")
    print(f"Matched stars DEC: {FM_T['TARGET_DEC'][mask_pm & sep_ok].min()} to {FM_T['TARGET_DEC'][mask_pm & sep_ok].max()}")


    gc_ra   = gc_df["RA"][0]
    gc_dec  = gc_df["DEC"][0]
    rsun    = gc_df["Rsun"][0]  # e.g., 20 * u.kpc
    rt      = gc_df["rt"][0]    # e.g., 50 * u.pc

    rt_rad = (rt / rsun) * u.rad
    print(rt_rad)
    rt_arcmin = rt_rad.to(u.deg).to(u.arcmin)
    print(f"rt_arcmin : {rt_arcmin:.3f}")

    sc_gc = coord.SkyCoord(gc_ra, gc_dec)
    sep_to_gc = sc_obs.separation(sc_gc)
    outside_gc = sep_to_gc > rt_arcmin

    # Final selection
    keep_mask = mask_pm & sep_ok & outside_gc
    return keep_mask, idx[keep_mask]


def box_pm_cone_rv_cut_DESI(FM_T, RV_T, sim_stream_tab, gc_df,
                            pm_sigma=10 * u.mas/u.yr,
                            sep_max=1.0 * u.deg,
                            rv_sigma=10 * u.km/u.s):
    """
    Select stars from FM_T that lie within:
    - a proper motion box (±pm_sigma around sim median),
    - a cone around the simulated stream,
    - outside GC tidal radius,
    - a radial velocity window ±rv_sigma.

    Returns:
    - keep_mask: boolean mask on FM_T
    - matched_sim_idx: indices into sim_stream_tab
    """

    # --- PM box cut (±pm_sigma from median) ---
    pmra_med = np.nanmedian(sim_stream_tab["PMRA"]) 
    pmde_med = np.nanmedian(sim_stream_tab["PMDEC"])

    lower_pmra = pmra_med - pm_sigma
    upper_pmra = pmra_med + pm_sigma
    lower_pmde = pmde_med - pm_sigma
    upper_pmde = pmde_med + pm_sigma

    mask_pm = (
        (FM_T["PMRA"] * u.mas/u.yr >= lower_pmra) &
        (FM_T["PMRA"] * u.mas/u.yr <= upper_pmra) &
        (FM_T["PMDEC"] * u.mas/u.yr >= lower_pmde) &
        (FM_T["PMDEC"] * u.mas/u.yr <= upper_pmde)
    )

    # --- Cone cut (sky separation from sim stream) ---
    sc_obs = coord.SkyCoord(FM_T["TARGET_RA"] * u.deg, FM_T["TARGET_DEC"] * u.deg)
    sc_sim = coord.SkyCoord(sim_stream_tab["RA"], sim_stream_tab["DEC"])
    idx, d2d, _ = sc_obs.match_to_catalog_sky(sc_sim)
    sep_ok = (d2d <= sep_max)

    # --- GC tidal radius exclusion ---
    gc_ra = gc_df["RA"][0]
    gc_dec = gc_df["DEC"][0]
    rsun = gc_df["Rsun"][0]
    rt = gc_df["rt"][0]

    rt_rad = (rt / rsun) * u.rad
    rt_arcmin = rt_rad.to(u.arcmin)

    sc_gc = coord.SkyCoord(gc_ra, gc_dec)
    sep_to_gc = sc_obs.separation(sc_gc)
    outside_gc = sep_to_gc > rt_arcmin

    vlos_med = np.nanmedian(sim_stream_tab["VLOS"])
    vlos_lower = vlos_med - rv_sigma
    vlos_upper = vlos_med + rv_sigma

    # Defensive: fall back if "VRAD_CORRECTED" missing
    rv_column = "VRAD_CORRECTED" if "VRAD_CORRECTED" in RV_T.colnames else "VRAD"
    rv_values = RV_T[rv_column] * u.km/u.s
    mask_rv = (rv_values >= vlos_lower) & (rv_values <= vlos_upper)

    keep_mask = mask_pm & sep_ok & outside_gc & mask_rv

    print(f"[PM] RA cut: {lower_pmra:.2f} to {upper_pmra:.2f}")
    print(f"[PM] DE cut: {lower_pmde:.2f} to {upper_pmde:.2f}")
    print(f"[RV] VLOS cut: {vlos_lower:.1f} to {vlos_upper:.1f}")
    print(f"Selected: {np.sum(keep_mask)} / {len(FM_T)} stars")

    return keep_mask, idx[keep_mask]

def cone_cut_DESI(FM_T, RV_T, sim_stream_tab,
                        sep_max=1.0 * u.deg):
    """
    Cone-only selection:
      keep stars whose angular separation to the nearest simulation point
      is <= sep_max. PM/RV/GC cuts intentionally disabled.

    Returns
    -------
    keep_mask : boolean array (len = len(FM_T))
    matched_sim_idx : int array of sim indices for kept stars
    """
    # Sky coords (ensure degree units)
    sc_obs = coord.SkyCoord(FM_T["TARGET_RA"] * u.deg, FM_T["TARGET_DEC"] * u.deg)
    sc_sim = coord.SkyCoord(sim_stream_tab["RA"], sim_stream_tab["DEC"]) # already in units

    # Nearest sim point per star on the sky
    idx, d2d, _ = sc_obs.match_to_catalog_sky(sc_sim)

    # Cone cut
    keep_mask = (d2d <= sep_max)
    n_keep = int(np.sum(keep_mask))
    print(f"[CONE] sep_max = {sep_max:.2f}; kept {n_keep} / {len(FM_T)} stars")

    return keep_mask, idx[keep_mask]

def mask_gc_region_DESI(FM_T, gc_ra, gc_dec, rsun, rt):
    """
    Exclude targets inside the GC tidal radius circle on the sky.

    FM_T: astropy Table with columns TARGET_RA, TARGET_DEC (degrees)
    gc_ra, gc_dec: GC center (deg or Quantity with angle unit)
    rsun, rt: same length units (e.g., kpc)
    """
    # angular tidal radius: atan(rt / rsun) in radians -> arcmin
    rt_arcmin = ((rt / rsun) * u.rad).to(u.arcmin)

    # Sky coords
    sc_obs = coord.SkyCoord(FM_T["TARGET_RA"] * u.deg, FM_T["TARGET_DEC"] * u.deg)
    ra_gc  = gc_ra if hasattr(gc_ra, "unit") else gc_ra * u.deg
    dec_gc = gc_dec if hasattr(gc_dec, "unit") else gc_dec * u.deg
    sc_gc  = coord.SkyCoord(ra_gc, dec_gc)

    # angular separation and mask
    sep_to_gc = sc_obs.separation(sc_gc).to(u.arcmin)
    print(f"Separation to GC center (arcmin): min {sep_to_gc.min():.3f}, max {sep_to_gc.max():.3f}")
    gc_mask = sep_to_gc < rt_arcmin
    return gc_mask
