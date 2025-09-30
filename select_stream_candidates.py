import argparse
import pandas as pd
from astropy.table import QTable, Table
import numpy as np
import astropy.units as u
import agama
from utils import plot_utils, mock_stream_utils, coordinate_utils


def main():
    parser = argparse.ArgumentParser(description="StreamCutter GC setup")
    parser.add_argument("--gc", type=str, required=True, help="Globular cluster name (e.g., 'Pal_5')")
    parser.add_argument("--potential", type=str, default="MWPotential2022", help="Potential name (default: MWPotential2022)")
    parser.add_argument("--n-particles", type=int, default=2000, help="Number of stream particles to simulate (default: 2000)")
    
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


if __name__ == "__main__":
    main()