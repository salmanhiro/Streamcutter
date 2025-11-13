# streamcutter.py (fetch-only)
import argparse, os, time
import astropy.units as u
from astropy.table import QTable

from utils.desi_utils import (
    GCParams, PotentialFactory, StreamSimulator,
    DESIData, DESISelector, Plotter, Exporter, SweepFetcher
)

def main():
    p = argparse.ArgumentParser(description="StreamCutter GC setup (OO refactor)")
    p.add_argument("--gc", required=True, help="Globular cluster name (e.g., 'Pal_5')")
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
    gc = GCParams().get_row(args.gc)

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
    sim_path = f"{out_sim_dir}/simulated_stream_{args.gc}.fits"
    Exporter.write_sim_stream(sim_path, sim_stream_tab)
    print(f"[v] wrote simulated stream to: {sim_path}")

    if not args.no_plot:
        tag = f"{args.potential} / {args.sim_mode}"
        Plotter.sim_stream(sim_stream_tab, gc,
                           save_path=f"{out_sim_dir}/{args.gc}_simulated_stream.png",
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
            manifest = os.path.join(args.sweep_dest_dir, f"sweep_manifest_{args.gc}.txt")
            with open(manifest, "w") as f:
                f.write("\n".join(sorted(res["selected_files"])))
            print(f"[sweep] manifest -> {manifest}")

    if args.simulate_only:
        return

    #  DESI IO
    RV_T, FM_T = DESIData().load()
    print(f"[i] DESI tables loaded: RV={len(RV_T)} FM={len(FM_T)}")

    # Basic main selection (your original flags)
    main_sel = RV_T['PRIMARY'] & (RV_T['RVS_WARN'] == 0) & (RV_T['RR_SPECTYPE'] == 'STAR')
    RV_T_sel = RV_T[main_sel]
    FM_T_sel = FM_T[main_sel]

    #  Positional cone cut
    mask_boxcut, RV_T_pos, FM_T_pos = DESISelector.positional_cone_cut(
        FM_T_sel, RV_T_sel, sim_stream_tab, sep_max_deg=args.sep_max
    )
    print(f"[i] Positional cone cut count: {mask_boxcut.sum()}")

    #  GC region masks/splits
    row = gc[0]
    res = DESISelector.gc_masks_and_splits(
        RV_T_pos_sel=RV_T_pos,
        FM_T_pos_sel=FM_T_pos,
        ra_deg=row['RA'], dec_deg=row['DEC'],
        rsun_pc=row['Rsun'].to(u.pc), rt_pc=row['rt']
    )
    print(f"[i] Stars in GC region: {res.gc_region_mask.sum()}")

    #  Plots
    os.makedirs("figures", exist_ok=True)
    Plotter.fm_density(res.FM_T_final, save_path=f"figures/{args.gc}_final_candidate_density.png")
    Plotter.density_with_sim(res.FM_T_final, sim_stream_tab, gc,
                             save_path=f"figures/{args.gc}_candidates_with_sim_stream.png")

    #  Exports
    os.makedirs("stream_candidates", exist_ok=True)
    Exporter.write_candidates(f"stream_candidates/stream_candidates_{args.gc}.fits",
                              res.RV_T_final, res.FM_T_final)
    print(f"[v] wrote candidates to: stream_candidates/stream_candidates_{args.gc}.fits")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Time taken: {time.time()-t0:.2f}s")

