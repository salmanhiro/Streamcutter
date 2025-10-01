import numpy as np
import astropy.units as u
import astropy.coordinates as coord

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

    # --- Final selection ---
    keep_mask = mask_pm & sep_ok & outside_gc & mask_rv

    # --- Logging ---
    print(f"[PM] RA cut: {lower_pmra:.2f} to {upper_pmra:.2f}")
    print(f"[PM] DE cut: {lower_pmde:.2f} to {upper_pmde:.2f}")
    print(f"[RV] VLOS cut: {vlos_lower:.1f} to {vlos_upper:.1f}")
    print(f"Selected: {np.sum(keep_mask)} / {len(FM_T)} stars")

    return keep_mask, idx[keep_mask]
