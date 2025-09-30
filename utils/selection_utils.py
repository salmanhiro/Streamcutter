import numpy as np
import astropy.units as u
import astropy.coordinates as coord

def box_and_cone_cut(streamfinder_gc, sim_stream_tab, frac=0.5, sep_max=1.0*u.deg):
    # PM median
    pmra_med = np.nanmedian(sim_stream_tab["PMRA"])
    pmde_med = np.nanmedian(sim_stream_tab["PMDEC"])
    
    half_pmra = frac * abs(pmra_med.to_value(u.mas/u.yr))
    half_pmde = frac * abs(pmde_med.to_value(u.mas/u.yr))
    
    lower_pmra, upper_pmra = pmra_med.value - half_pmra, pmra_med.value + half_pmra
    lower_pmde, upper_pmde = pmde_med.value - half_pmde, pmde_med.value + half_pmde
    
    # Box cut
    mask_pm = (
        (streamfinder_gc["pmRA"] >= lower_pmra) & (streamfinder_gc["pmRA"] <= upper_pmra) &
        (streamfinder_gc["pmDE"] >= lower_pmde) & (streamfinder_gc["pmDE"] <= upper_pmde)
    )

    # Sky separation
    sc_obs = coord.SkyCoord(streamfinder_gc["RAdeg"] * u.deg, streamfinder_gc["DEdeg"] * u.deg)
    sc_sim = coord.SkyCoord(sim_stream_tab["RA"], sim_stream_tab["DEC"])
    
    idx, d2d, _ = sc_obs.match_to_catalog_sky(sc_sim)
    sep_ok = (d2d <= sep_max)
    
    # Final mask
    keep_mask = mask_pm & sep_ok

    return keep_mask, idx[keep_mask]  # returns mask on streamfinder_gc and matched sim idx

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
