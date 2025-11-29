# streamcutter.py (fetch-only)
import argparse, os, time
import astropy.units as u
from astropy.table import QTable

from utils.desi_utils import (
    DESIData, DESISelector, Plotter, Exporter, SweepFetcher
)
from utils.GC_utils import GCParams, PotentialFactory  
from utils.mock_stream_utils import StreamSimulator

def main():
    p = argparse.ArgumentParser(description="StreamCutter GC setup (OO refactor)")
    p.add_argument("--gc", required=True, help="Globular cluster name (e.g., 'Pal_5')")
    p.add_argument("--potential", default="MWPotential2022")
    p.add_argument("--n-particles", type=int, default=10000)
    p.add_argument("--sep-max", type=float, default=1.0)
    p.add_argument("--n-orbits", required=False, type=int,
                   help="Number of orbits to simulate")
    p.add_argument("--total-time", type=float, required=False, default=None,
                   help="Total integration time [Gyr]; if None, use 2*orbital_period")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--sim-stream-outfile", default=None,
                   help="If provided, write simulated stream to this path")

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
    gc = GCParams().get_row(args.gc)

    #  Simulate
    factory = PotentialFactory()
    sim = StreamSimulator(
        factory,
        sim_mode=args.sim_mode,        # "restricted" (default) or "full"
        vcirc10=args.vcirc10,          # used only in full mode
        eps=args.eps,                  # used only in full mode
        tau=args.tau,                  # used only in full mode
        use_df=not args.no_df          # used only in full mode
    )
    time_sat, orbit_sat, xv_stream, ic_stream, sim_stream_tab, sim_stream_meta = sim.simulate(
        gc_row=gc,
        potential_name=args.potential,
        n_particles=args.n_particles,
        n_orbits=args.n_orbits,
        rng_seed=0,
        total_time=args.total_time
    )

    # Save sim stream (always)
    out_sim_dir = "simulated_streams"
    os.makedirs(out_sim_dir, exist_ok=True)
    
    sim_path = args.sim_stream_outfile if args.sim_stream_outfile else f"{out_sim_dir}/simulated_stream_{args.gc}.fits"
    # also save metadata
    sim_stream_tab.meta.update(sim_stream_meta)
    
    Exporter.write_sim_stream(sim_path, sim_stream_tab)
    print(f"[v] wrote simulated stream to: {sim_path}")

    # print metadata
    print("Simulation metadata:")
    for k, v in sim_stream_meta.items():
        print(f"  {k}: {v}")

    if not args.no_plot:
        tag = f"{args.potential} / {args.sim_mode}"
        plot_save_path = f"{out_sim_dir}/{args.gc}_simulated_stream.png" if not args.sim_stream_outfile else f"{args.sim_stream_outfile}.png"
        Plotter.sim_stream(sim_stream_tab, gc,
                           save_path=plot_save_path,
                           potential_name=tag,
                           n_orbit=args.n_orbits)
        # Plot orbit
        orbit_plot_path = f"{out_sim_dir}/{args.gc}_simulated_orbit.png" if not args.sim_stream_outfile else f"{args.sim_stream_outfile}_orbit.png"
        Plotter.sim_orbit(time_sat, orbit_sat, gc, save_path=orbit_plot_path, potential_name=tag, n_orbit=args.n_orbits)
        print(f"[v] wrote orbit plot to: {orbit_plot_path}")
        

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
            manifest = os.path.join(args.sweep_dest_dir, f"sweep_manifest_{args.gc}.txt")
            with open(manifest, "w") as f:
                f.write("\n".join(sorted(res["selected_files"])))
            print(f"[sweep] manifest -> {manifest}")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Time taken: {time.time()-t0:.2f}s")

