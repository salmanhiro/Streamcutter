import argparse, os, time
import astropy.units as u
from astropy.table import QTable

import sys
sys.path.append('..')

from utils.desi_utils import (
    
    DESIData, DESISelector, Plotter, Exporter, SweepFetcher
)

from utils.GC_utils import (
    GCParams, PotentialFactory,
)

from utils.mock_stream_utils import StreamSimulator

def main():
    p = argparse.ArgumentParser(description="StreamCutter GC setup (OO refactor)")
    p.add_argument("--potential", default="MWPotential2014")
    p.add_argument("--n-particles", type=int, default=1000)
    p.add_argument("--sep-max", type=float, default=1.0)
    p.add_argument("--n-orbits", type=int, default=3)
    p.add_argument("--simulate-only", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    #  Load GC row

    from astropy.table import Table

    gc_name_tab = Table.read("data/gc_names_list_indesi.csv")
    gc_names = list(gc_name_tab["Cluster"])

    for gc_name in gc_names:
        
        gc = GCParams().get_row(gc_name)

        # First, check GC distance from gc["Rsun"]
        print(f"[*] Checking GC: {gc_name} at distance {gc['Rsun'].value[0]:.2f} kpc")
        if gc["Rsun"].value[0] > 31.0:
            print(f"[!] Skipping {gc_name}: distance {gc['Rsun'].value[0]:.2f} kpc > 31 kpc")
            continue

        print(f"[*] Simulating stream for GC: {gc_name}")

        #  Simulate
        factory = PotentialFactory()
        sim = StreamSimulator(
            factory,
        )
        time_sat, orbit_sat, xv_stream, ic_stream, sim_stream_tab, sim_stream_meta = sim.simulate(
            gc_row=gc,
            potential_name=args.potential,
            n_particles=args.n_particles,
            n_orbits=args.n_orbits,
            rng_seed=0
        )

        # Save sim stream (always)
        out_sim_dir = "simulated_streams"
        os.makedirs(out_sim_dir, exist_ok=True)
        sim_path = f"{out_sim_dir}/simulated_stream_{gc_name}.fits"
        Exporter.write_sim_stream(sim_path, sim_stream_tab)
        print(f"[v] wrote simulated stream to: {sim_path}")

        if not args.no_plot:
            tag = f"{args.potential} / restricted"
            Plotter.sim_stream(sim_stream_tab, gc,
                            save_path=f"{out_sim_dir}/{gc_name}_simulated_stream.png",
                            potential_name=tag,
                            n_orbit=args.n_orbits)

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Time taken: {time.time()-t0:.2f}s")
