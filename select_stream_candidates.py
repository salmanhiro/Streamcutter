import argparse
import pandas as pd
from astropy.table import QTable, Table
import numpy as np
import astropy.units as u
import agama
from utils import plot_utils, mock_stream_utils, coordinate_utils, selection_utils, feh_correct
from astropy.table import join
from astropy.io import fits

def main():
    parser = argparse.ArgumentParser(description="StreamCutter GC setup")
    parser.add_argument("--gc", type=str, required=True, help="Globular cluster name (e.g., 'Pal_5')")
    parser.add_argument("--potential", type=str, default="MWPotential2022", help="Potential name (default: MWPotential2022)")
    parser.add_argument("--n-particles", type=int, default=2000, help="Number of stream particles to simulate (default: 2000)")
    parser.add_argument("--pm-")
    
    args = parser.parse_args()

    gc_name = args.gc
    potential_name = args.potential
    n_particles = args.n_particles

    # Load GC parameters
    gc_param_path = "data/mw_gc_parameters_orbital_structural_time.ecsv"
    gc_df = QTable.read(gc_param_path)
    gc_df = gc_df[gc_df["Cluster"] == gc_name]
    if len(gc_df) == 0:
        raise ValueError(f"Cluster '{gc_name}' not found in GC parameter file: {gc_param_path}")

    # Load StreamFinder mapping
    sf_map_df = pd.read_csv("data/streamfinder_baumgardt_map.csv")
    sf_name_dict = {
        row["Cluster"]: row["streamfinder_name"]
        for _, row in sf_map_df.iterrows()
        if str(row["in_streamfinder"]).strip().upper() == "TRUE"
    }
    sf_name = sf_name_dict.get(gc_name, gc_name)

    print(f"GC: {gc_name}")
    print(f"StreamFinder name: {sf_name}")
    print(f"Potential: {potential_name}")

    # Agama units
    agama.setUnits(length=1, velocity=1, mass=1)
    timeUnitGyr = agama.getUnits()['time']

    # Extract GC info
    row = gc_df[0]
    ra_q, dec_q, dist_q = row["RA"].value, row["DEC"].value, row["Rsun"].value
    rt_q = row["rt"].value
    pmra, pmdec = row["mualpha"].value, row["mu_delta"].value
    mass_sat, rv = row["Mass"].value, row["<RV>"].value
    rhm = row["rh,m"].value

    x_gc, y_gc, z_gc = row["X_gc"].value, row["Y_gc"].value, row["Z_gc"].value
    vx_gc, vy_gc, vz_gc = row["Vx_gc"].value, row["Vy_gc"].value, row["Vz_gc"].value
    orbit_t = row["orbit_period_max"].value

    # Streamfinder data
    sf_data = Table.read("data/cleaned_streamfinder_ibata24.csv")
    sf_data = sf_data[sf_data["Name"] == sf_name]

    plot_utils.plot_gc_stream_and_tidal_radius(gc_name, ra_q, dec_q, rt_q * u.pc, dist_q * u.kpc, stream_data=sf_data)

    # Setup potential & progenitor
    pot_host = agama.GalaPotential(f"potentials/{potential_name}.ini")
    pot_sat = agama.GalaPotential(type='Plummer', mass=mass_sat, scaleRadius=rhm)
    prog_w0 = [x_gc, y_gc, z_gc, vx_gc, vy_gc, vz_gc]

    # Stream generation
    time_total = (-3 * orbit_t) / 978 if orbit_t < 1000 else -3
    time_sat, orbit_sat, xv_stream, ic_stream = mock_stream_utils.create_stream(
        mock_stream_utils.create_initial_condition_fardal15,
        np.random.default_rng(0),
        time_total, n_particles,
        pot_host, prog_w0, mass_sat,
        pot_sat=pot_sat
    )

    # Convert to observables
    ra, dec, vlos, pmra, pmdec, dist = coordinate_utils.get_observed_coords(xv_stream)

    # Ensure units
    ra = ra * u.deg if not hasattr(ra, "unit") else ra
    dec = dec * u.deg if not hasattr(dec, "unit") else dec
    vlos = vlos * (u.km/u.s) if not hasattr(vlos, "unit") else vlos
    pmra = pmra * (u.mas/u.yr) if not hasattr(pmra, "unit") else pmra
    pmdec = pmdec * (u.mas/u.yr) if not hasattr(pmdec, "unit") else pmdec
    dist = dist * u.kpc if not hasattr(dist, "unit") else dist

    sim_stream_tab = QTable({
        "RA": ra.to(u.deg),
        "DEC": dec.to(u.deg),
        "PMRA": pmra.to(u.mas/u.yr),
        "PMDEC": pmdec.to(u.mas/u.yr),
        "VLOS": vlos.to(u.km/u.s),
        "DIST": dist.to(u.kpc),
    })

    print(f"Simulated stream particles: {len(sim_stream_tab)}")

    mask_boxcut, idx_sim_box = selection_utils.box_and_cone_cut(sf_data, sim_stream_tab, frac=0.5, sep_max=1.0*u.deg)
    print(f"PM box+ positional cone: {mask_boxcut.sum()} / {len(mask_boxcut)} = {100*mask_boxcut.mean():.2f}%")

    plot_utils.plot_matched_streamfinder_vs_sim(
        sf_data=sf_data,
        sim_stream_tab=sim_stream_tab,
        idx_sim_box=idx_sim_box,
        mask_boxcut=mask_boxcut,
        gc_ra=ra_q,
        gc_dec=dec_q,
        gc_rt=rt_q * u.pc,
        gc_rsun=dist_q * u.kpc,
        gc_name=gc_name
    )

    print("Reading RV table and applying corrections...")
    # Koposov https://academic.oup.com/mnras/article/533/1/1012/7724389
    # Open the FITS file
    fits_path = 'data/mwsall-pix-iron.fits'
    RV_OFFSET_TAB =  Table().read('data/backup_correction.fits')

    # Note these are pretty big tables
    RV_T = Table().read('../data/mwsall-pix-iron.fits',
                            'RVTAB',
                            mask_invalid=False)

    FM_T = Table().read('../data/mwsall-pix-iron.fits',
                            'FIBERMAP',
                            mask_invalid=False)

    RV_T['FEH_CORRECTED'] = feh_correct.calibrate(RV_T['FEH'], RV_T['TEFF'], RV_T['LOGG'])

    # For backup program, need to correct RV since there is RV offset. Otherwise, use RV value
    RV_OFFSET_TAB =  Table().read('data/backup_correction.fits')

    RV_T = join(RV_T, RV_OFFSET_TAB,
                keys='TARGETID', join_type='left')

    # Compute corrected radial velocity for backup program
    RV_T['VRAD_CORRECTED'] = np.where(
        np.isnan(RV_T['VRAD_BIAS']),
        RV_T['VRAD'],                         # Use uncorrected VRAD if bias is NaN
        RV_T['VRAD'] - RV_T['VRAD_BIAS']      # Otherwise apply bias correction
    )

    num_masked = RV_T['VRAD_BIAS'].mask.sum()
    num_unmasked = len(RV_T) - num_masked

    print(f"Total rows: {len(RV_T):,}")
    print(f"Rows corrected (with VRAD_BIAS): {num_unmasked:,} ({num_unmasked / len(RV_T) * 100:.2f}%)")
    print(f"Rows uncorrected (missing VRAD_BIAS): {num_masked:,} ({num_masked / len(RV_T) * 100:.2f}%)")

    # Save the corrected RV table for future use
    corrected_fits_path = 'data/mwsall-pix-iron-rv-corrected.fits'
    RV_T.write(corrected_fits_path, overwrite=True)

    main_sel = RV_T['PRIMARY'] & (RV_T['RVS_WARN'] == 0) & (RV_T['RR_SPECTYPE'] == 'STAR')
    RV_T_sel = RV_T[main_sel]
    FM_T_sel = FM_T[main_sel]

    plot_utils.plot_density_with_sim_stream(
        RV_T=RV_T,
        sim_stream_tab=sim_stream_tab,
        main_sel=main_sel,
        gc_df=gc_df,
        gc_name="Pal_5"
    )

    mask_boxcut, idx_sim_box = selection_utils.box_and_cone_cut_DESI(FM_T_sel, sim_stream_tab, gc_df, frac=0.5, sep_max=1.0*u.deg)
    print(f"PM box+ positional cone from DESI FM: {mask_boxcut.sum()} / {len(mask_boxcut)} = {100*mask_boxcut.mean():.2f}%")

    candidates_filtered_pos_pm_rv = RV_T_sel[mask_boxcut]
    idx_sim_box_filtered_pos_pm_fm = FM_T_sel[mask_boxcut]

    # Save results
    out_dir = "results"
    candidates_filtered_pos_pm_rv.write(f"{out_dir}/{gc_name}_candidates_filtered_pos_pm_rv.fits")
    idx_sim_box_filtered_pos_pm_fm.write(f"{out_dir}/{gc_name}_candidates_filtered_pos_pm_fm.fits")

    print(f"[v] Saved filtered candidates to {out_dir}/{gc_name}_candidates_filtered_pos_pm_rv.fits")

    # Plot RV histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(candidates_filtered_pos_pm_rv['VRAD_CORRECTED'], bins=30, color='blue', alpha=0.7)
    ax.set_xlabel('Corrected Radial Velocity (km/s)')
    ax.set_ylabel('Number of Stars')
    ax.set_title(f'Radial Velocity Distribution for {gc_name} Candidates')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{gc_name}_candidates_rv_histogram.png", dpi=300)

if __name__ == "__main__":
    main()