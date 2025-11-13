import argparse, os, time
import astropy.units as u
from astropy.table import QTable

from utils.desi_utils import (
    GCParams, PotentialFactory, StreamSimulator,
    DESIData, DESISelector, Plotter, Exporter, SweepFetcher
)

def main():
    p = argparse.ArgumentParser(description="StreamCutter GC setup (OO refactor)")
    p.add_argument("--potential", default="MWPotential2022")
    p.add_argument("--n-particles", type=int, default=10000)
    p.add_argument("--sep-max", type=float, default=1.0)
    p.add_argument("--n-orbits", type=int, default=3)
    p.add_argument("--simulate-only", action="store_true")
    p.add_argument("--no-plot", action="store_true")

    # sweep fetch (copy bricks, no loading)
    p.add_argument("--sweep-lookup-csv", default="./data/sweep_photometry_lookup_coord.csv")
    p.add_argument("--fetch-sweep", action="store_true", help="Copy needed sweep bricks locally")
    p.add_argument("--sweep-min-count", type=int, default=50)
    p.add_argument("--sweep-dest-dir", default="./data/sweep_catalogue",
                   help="Local dir where sweep bricks will be stored")

    # Full nbody additional options
    p.add_argument("--sim-mode", choices=["restricted", "full"], default="restricted",
                   help="restricted = original streakline (default); full = pyfalcon self-grav")
    p.add_argument("--vcirc10", type=float, default=200.0,
                   help="Host normalization: v_circ(10 kpc) [km/s] (full mode)")
    p.add_argument("--eps", type=float, default=0.1,
                   help="Softening length [kpc] for full N-body (pyfalcon)")
    p.add_argument("--tau", type=float, default=None,
                   help="Leapfrog step (kpc/(km/s)); if None, auto-choose (full mode)")
    p.add_argument("--no-df", action="store_true",
                   help="Disable Chandrasekhar DF on the CoM in full mode")

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
            sim_mode=args.sim_mode,         # "restricted" (default) or "full"
            vcirc10=args.vcirc10,          # used only in full mode
            eps=args.eps,                  # used only in full mode
            tau=args.tau,                  # used only in full mode
            use_df=not args.no_df          # used only in full mode
        )
        time_sat, orbit_sat, xv_stream, ic_stream, sim_stream_tab = sim.simulate(
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
            tag = f"{args.potential} / {args.sim_mode}"
            Plotter.sim_stream(sim_stream_tab, gc,
                            save_path=f"{out_sim_dir}/{gc_name}_simulated_stream.png",
                            potential_name=tag)

        # Fetch/copy sweep bricks that cover the simulated stream footprint
        if args.fetch_sweep:
            sf = SweepFetcher(args.sweep_lookup_csv)
            res = sf.fetch(sim_stream_tab["RA"], sim_stream_tab["DEC"],
                        dest_dir=args.sweep_dest_dir,
                        min_count=args.sweep_min_count,
                        skip_if_exists=True, dry_run=False)
            print(f"[sweep] selected={len(res['selected_files'])} "
                f"copied={len(res['copied_files'])} "
                f"skipped={len(res['skipped_existing'])} "
                f"missing={len(res['missing_source_files'])}")
            # Optional: write a small manifest to the dest dir
            if res["selected_files"]:
                os.makedirs(args.sweep_dest_dir, exist_ok=True)
                manifest = os.path.join(args.sweep_dest_dir, f"sweep_manifest_{gc_name}.txt")
                with open(manifest, "w") as f:
                    f.write("\n".join(sorted(res["selected_files"])))
                print(f"[sweep] manifest -> {manifest}")

        if args.simulate_only:
            return

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Time taken: {time.time()-t0:.2f}s")
