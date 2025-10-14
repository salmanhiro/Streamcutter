import argparse
import pandas as pd
from astropy.table import QTable, Table
import numpy as np
import astropy.units as u
import agama
from utils import plot_utils, mock_stream_utils, coordinate_utils, selection_utils, feh_correct, extract_gc_region
import os 
import matplotlib.pyplot as plt
import time

def main():
    parser = argparse.ArgumentParser(description="StreamCutter GC setup")
    parser.add_argument("--gc", type=str, required=True, help="Globular cluster name (e.g., 'Pal_5')")
    parser.add_argument("--potential", type=str, default="MWPotential2014", help="Potential name (default: MWPotential2022)")
    parser.add_argument("--n-particles", type=int, default=10000, help="Number of stream particles to simulate (default: 2000)")
    parser.add_argument("--pm-sigma", type=float, default=1.0, help="Fraction for PM box cut (default: 2)")
    parser.add_argument("--sep-max", type=float, default=1.0, help="Max separation in degrees for positional cut (default: 1.0)")
    parser.add_argument("--n-orbits", type=int, default=3, help="Number of orbits to simulate (default: 3)")
    
    args = parser.parse_args()

    gc_name = args.gc
    potential_name = args.potential
    n_particles = args.n_particles
    sep_max = args.sep_max
    n_orbits = args.n_orbits

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

    orbit_t = row["orbit_period_max"].value

    # Streamfinder data
    sf_data = Table.read("data/cleaned_streamfinder_ibata24.csv")
    sf_data = sf_data[sf_data["Name"] == sf_name]
    # check if parallax is 0 then remove those rows
    sf_data = sf_data[sf_data["plx"] > 0]
    sf_data["Rsun"] = (1 / sf_data["plx"]) # mas to kpc
    # check rv is not 0
    sf_data = sf_data[sf_data["VHel"] != 0]


    sf_w0 = coordinate_utils.get_galactocentric_coords(
        sf_data["RAdeg"], sf_data["DEdeg"], sf_data["Rsun"], sf_data["VHel"], sf_data["pmRA"], sf_data["pmDE"]
    )
    sf_data["X_gc"] = sf_w0[:, 0]
    sf_data["Y_gc"] = sf_w0[:, 1]
    sf_data["Z_gc"] = sf_w0[:, 2]
    sf_data["Vx_gc"] = sf_w0[:, 3]
    sf_data["Vy_gc"] = sf_w0[:, 4]
    sf_data["Vz_gc"] = sf_w0[:, 5]

    plot_utils.plot_gc_stream_and_tidal_radius(gc_name, ra_q, dec_q, rt_q * u.pc, dist_q * u.kpc, stream_data=sf_data)

    # Setup potential & progenitor
    pot_host = agama.GalaPotential(f"potentials/{potential_name}.ini")
    pot_sat = agama.GalaPotential(type='Plummer', mass=mass_sat, scaleRadius=rhm)
    prog_w0 = coordinate_utils.get_galactocentric_coords(ra_q, dec_q, dist_q, rv, pmra, pmdec)[0]
    # Stream generation
    time_total = (-n_orbits * orbit_t)/978 if orbit_t < 1000 else -3
    print(f"Simulating stream for {time_total:.2f} Gyr with {n_particles} particles...")
    time_sat, orbit_sat, xv_stream, ic_stream = mock_stream_utils.create_stream(
        mock_stream_utils.create_initial_condition_fardal15,
        np.random.default_rng(0),
        time_total, n_particles,
        pot_host, prog_w0, mass_sat,
        pot_sat=pot_sat
    )

    # Convert to observables
    ra, dec, vlos, pmra, pmdec, dist = coordinate_utils.get_observed_coords(xv_stream)

    sim_stream_tab = QTable({
        "RA":    ra * u.deg,
        "DEC":   dec * u.deg,
        "PMRA":  pmra * u.mas/u.yr,
        "PMDEC": pmdec * u.mas/u.yr,
        "VLOS":  vlos * u.km/u.s,
        "DIST":  dist * u.kpc,
        "X":     (xv_stream[:, 0] * u.kpc),
        "Y":     (xv_stream[:, 1] * u.kpc),
        "Z":     (xv_stream[:, 2] * u.kpc),
        "Vx":    (xv_stream[:, 3] * u.km/u.s),
        "Vy":    (xv_stream[:, 4] * u.km/u.s),
        "Vz":    (xv_stream[:, 5] * u.km/u.s),
    })

    print(f"Simulated stream particles: {len(sim_stream_tab)}")
    mask_boxcut, idx_sim_box = selection_utils.cone_cut(sf_data, sim_stream_tab, sep_max=sep_max*u.deg)
    percentage_match = mask_boxcut.sum() / len(mask_boxcut) * 100
    print(f"Positional cone match with StreamFinder: {mask_boxcut.sum()} / {len(mask_boxcut)} = {percentage_match:.2f}%")

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

    plot_utils.plot_overlay_streamfinder_vs_sim(
        sf_data=sf_data,
        sim_stream_tab=sim_stream_tab,
        idx_sim_box=idx_sim_box,
        gc_ra=ra_q,
        gc_dec=dec_q,
        gc_rt=rt_q * u.pc,
        gc_rsun=dist_q * u.kpc,
        gc_name=gc_name
    )

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Time taken: {time.time() - start:.2f} seconds")