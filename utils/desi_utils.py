# desi_utils.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable, Set, List

import os
import shutil
import numpy as np
import astropy.units as u
from astropy.table import QTable, Table
import agama

# Your existing utils
from utils import mock_stream_utils, coordinate_utils, selection_utils, plot_utils


# ---------------------------
# GC parameters & potentials
# ---------------------------

@dataclass
class GCParams:
    """Helper to load the GC parameter table and select a cluster row."""
    table_path: str = "data/mw_gc_parameters_orbital_structural_time.ecsv"

    def get_row(self, cluster_name: str) -> QTable:
        tab = QTable.read(self.table_path)
        sel = tab[tab["Cluster"] == cluster_name]
        if len(sel) == 0:
            raise ValueError(f"Cluster '{cluster_name}' not found in: {self.table_path}")
        return sel[0:1]  # keep as 1-row table for convenience


class PotentialFactory:
    """Create host & satellite potentials, and ensure Agama unit setup."""
    def __init__(self, potentials_dir: str = "potentials"):
        self.potentials_dir = potentials_dir
        self._ensure_units()

    @staticmethod
    def _ensure_units():
        # Matches your original: code assumes ini files are already in these units
        agama.setUnits(length=1, velocity=1, mass=1)

    def host(self, potential_name: str) -> agama.GalaPotential:
        ini = Path(self.potentials_dir) / f"{potential_name}.ini"
        if not ini.is_file():
            raise FileNotFoundError(f"Host potential ini not found: {ini}")
        return agama.GalaPotential(str(ini))

    @staticmethod
    def satellite_plummer(mass: float, rhm: float) -> agama.GalaPotential:
        # scaleRadius expects same length units as Agama config (here unitless, consistent with setUnits above)
        return agama.GalaPotential(type="Plummer", mass=mass, scaleRadius=rhm)


# ---------------------------
# Stream simulation
# ---------------------------

class StreamSimulator:
    """Simulate a tidal stream and return both phase-space and observables table."""
    def __init__(self, factory: PotentialFactory):
        self.factory = factory

    @staticmethod
    def _total_time_gyr(orbit_period_max: float, n_orbits: int) -> float:
        # Keep your original heuristic:
        return (-n_orbits * orbit_period_max) / 978.0 if orbit_period_max < 1000 else -3.0

    def simulate(
        self,
        gc_row: QTable,
        potential_name: str,
        n_particles: int,
        n_orbits: int,
        rng_seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, QTable]:
        row = gc_row[0]

        # Extract GC info (match your column names)
        ra_q   = row["RA"].value
        dec_q  = row["DEC"].value
        dist_q = row["Rsun"].value
        pmra   = row["mualpha"].value
        pmdec  = row["mu_delta"].value
        mass   = row["Mass"].value
        rv     = row["<RV>"].value
        rhm    = row["rh,m"].value
        orbit_t = row["orbit_period_max"].value

        # Potentials & initial conditions
        pot_host = self.factory.host(potential_name)
        pot_sat  = self.factory.satellite_plummer(mass, rhm)
        prog_w0  = coordinate_utils.get_galactocentric_coords(ra_q, dec_q, dist_q, rv, pmra, pmdec)[0]

        # Simulate
        time_total = self._total_time_gyr(orbit_t, n_orbits)
        time_sat, orbit_sat, xv_stream, ic_stream = mock_stream_utils.create_stream(
            mock_stream_utils.create_initial_condition_fardal15,
            np.random.default_rng(rng_seed),
            time_total, n_particles,
            pot_host, prog_w0, mass,
            pot_sat=pot_sat
        )

        # To observables table
        ra, dec, vlos, pmra_o, pmdec_o, dist = coordinate_utils.get_observed_coords(xv_stream)
        sim_stream_tab = QTable({
            "RA":   ra * u.deg,
            "DEC":  dec * u.deg,
            "PMRA": pmra_o * u.mas/u.yr,
            "PMDEC": pmdec_o * u.mas/u.yr,
            "VLOS": vlos * u.km/u.s,
            "DIST": dist * u.kpc,
            "X":  xv_stream[:, 0] * u.kpc,
            "Y":  xv_stream[:, 1] * u.kpc,
            "Z":  xv_stream[:, 2] * u.kpc,
            "Vx": xv_stream[:, 3] * u.km/u.s,
            "Vy": xv_stream[:, 4] * u.km/u.s,
            "Vz": xv_stream[:, 5] * u.km/u.s,
        })

        return time_sat, orbit_sat, xv_stream, ic_stream, sim_stream_tab


# ---------------------------
# DESI data IO & selection
# ---------------------------

class DESIData:
    """Load DESI-derived tables you use in the pipeline."""
    def __init__(self, rv_fm_fits_path: str = "data/mwsall-pix-iron-rv-corrected.fits"):
        self.rv_fm_fits_path = rv_fm_fits_path

    def load(self) -> Tuple[Table, Table]:
        RV_T = Table.read(self.rv_fm_fits_path, "RVTAB", mask_invalid=False)
        FM_T = Table.read(self.rv_fm_fits_path, "FIBERMAP", mask_invalid=False)
        return RV_T, FM_T


@dataclass
class DESISelectionResult:
    RV_T_final: Table
    FM_T_final: Table
    RV_T_around_GC: Table
    FM_T_around_GC: Table
    mask_boxcut: np.ndarray
    gc_region_mask: np.ndarray


class DESISelector:
    """All positional & GC-region selections, grouped here."""
    @staticmethod
    def positional_cone_cut(FM_T: Table, RV_T: Table, sim_stream_tab: QTable, sep_max_deg: float) -> Tuple[np.ndarray, Table, Table]:
        mask_boxcut, _ = selection_utils.cone_cut_DESI(FM_T, RV_T, sim_stream_tab, sep_max=sep_max_deg * u.deg)
        RV_T_pos_sel = RV_T[mask_boxcut]
        FM_T_pos_sel = FM_T[mask_boxcut]
        return mask_boxcut, RV_T_pos_sel, FM_T_pos_sel

    @staticmethod
    def gc_masks_and_splits(
        RV_T_pos_sel: Table,
        FM_T_pos_sel: Table,
        ra_deg: u.Quantity,
        dec_deg: u.Quantity,
        rsun_pc: u.Quantity,
        rt_pc: u.Quantity,
        around_gc_sep_deg: float = 0.5,
    ) -> DESISelectionResult:
        gc_region_mask = selection_utils.mask_gc_region_DESI(FM_T_pos_sel, ra_deg, dec_deg, rsun_pc, rt_pc)
        outside_gc_mask = ~gc_region_mask

        around_gc_mask = selection_utils.around_gc_region_DESI(FM_T_pos_sel, ra_deg, dec_deg, sep=around_gc_sep_deg * u.deg)

        RV_T_around_GC = RV_T_pos_sel[around_gc_mask]
        FM_T_around_GC = FM_T_pos_sel[around_gc_mask]

        RV_T_final = RV_T_pos_sel[outside_gc_mask]
        FM_T_final = FM_T_pos_sel[outside_gc_mask]

        return DESISelectionResult(
            RV_T_final=RV_T_final, FM_T_final=FM_T_final,
            RV_T_around_GC=RV_T_around_GC, FM_T_around_GC=FM_T_around_GC,
            mask_boxcut=gc_region_mask,  # (kept for reference)
            gc_region_mask=gc_region_mask
        )


# ---------------------------
# Plotting wrappers
# ---------------------------

class Plotter:
    """Thin wrappers over your existing plot_utils to keep main clean."""
    @staticmethod
    def sim_stream(sim_stream_tab: QTable, gc_df: QTable, save_path: str, potential_name: str):
        plot_utils.plot_sim_stream(sim_stream_tab=sim_stream_tab, gc_df=gc_df,
                                   save_path=save_path, potential_name=potential_name)

    @staticmethod
    def fm_density(FM_T: Table, save_path: str):
        plot_utils.plot_fm_density(FM_T, save_path=save_path)

    @staticmethod
    def density_with_sim(FM_T: Table, sim_stream_tab: QTable, gc_df: QTable, save_path: str):
        plot_utils.plot_density_with_sim_stream(FM_T, sim_stream_tab, gc_df, save_path=save_path)


# ---------------------------
# Export helpers
# ---------------------------

class Exporter:
    """Write out FITS products for candidates, around-GC, and simulated stream."""
    @staticmethod
    def write_candidates(path: str, RV_T_final: Table, FM_T_final: Table) -> None:
        from astropy.io import fits
        h0 = fits.PrimaryHDU()
        h1 = fits.table_to_hdu(RV_T_final); h1.name = "RVTAB"
        h2 = fits.table_to_hdu(FM_T_final); h2.name = "FIBERMAP"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList([h0, h1, h2]).writeto(path, overwrite=True)

    @staticmethod
    def write_table(path: str, table: Table) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        table.write(path, overwrite=True)

    @staticmethod
    def write_around_gc(path: str, RV_T_around: Table, FM_T_around: Table) -> None:
        from astropy.io import fits
        h0 = fits.PrimaryHDU()
        h1 = fits.table_to_hdu(RV_T_around); h1.name = "RVTAB"
        h2 = fits.table_to_hdu(FM_T_around); h2.name = "FIBERMAP"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList([h0, h1, h2]).writeto(path, overwrite=True)

    @staticmethod
    def write_sim_stream(path: str, sim_stream_tab: QTable) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sim_stream_tab.write(path, overwrite=True)


# ---------------------------
# Sweep brick fetcher (from RA/Dec footprint)
# ---------------------------

def _to_deg_array(x) -> np.ndarray:
    if isinstance(x, u.Quantity):
        return np.asarray(x.to(u.deg).value, dtype=np.float64)
    return np.asarray(x, dtype=np.float64)

class SweepFetcher:
    """
    Map stream RA/Dec to LS DR9 sweep bricks via a lookup CSV
    and copy the bricks locally. Missing bricks are recorded, not fatal.
    """
    def __init__(self, sweep_lookup_csv: str):
        self.lookup = Table.read(sweep_lookup_csv, format="csv")
        need = {"RA", "Dec", "file_location"}
        if not need.issubset(self.lookup.colnames):
            raise ValueError(f"Lookup CSV must have columns {need}")

        # (RA_int, Dec_int) -> path
        self._lut: Dict[Tuple[int, int], str] = {
            (int(r["RA"]), int(r["Dec"])): str(r["file_location"]) for r in self.lookup
        }

    def select_files(
        self, ra: Iterable[float], dec: Iterable[float], min_count: int = 50
    ) -> Tuple[Set[str], Dict[str, int]]:
        ra_floor = np.floor(_to_deg_array(ra)).astype(int)
        dec_floor = np.floor(_to_deg_array(dec)).astype(int)

        hits: List[Optional[str]] = [self._lut.get((ri, di), None) for ri, di in zip(ra_floor, dec_floor)]
        from collections import Counter
        counts = Counter([h for h in hits if h is not None])
        selected = {f for f, c in counts.items() if c >= int(min_count)}
        return selected, dict(counts)
    
    @staticmethod
    def _download_missing_from_mirror(
        missing_src_paths: Iterable[str],
        dest_dir: str,
        base_url: str = "https://casdc.china-vo.org/mirror/DESI/cosmo/data/legacysurvey/dr9/south/sweep/9.0/",
        skip_if_exists: bool = True,
        retries: int = 2,
        timeout: float = 30.0,
    ) -> tuple[list[str], dict[str, str]]:
        """
        For each missing source path, download <base_url>/<basename> to dest_dir.
        Returns (downloaded_dest_paths, errors_by_srcpath).
        """
        import time
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError

        downloaded: list[str] = []
        errors: dict[str, str] = {}
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        for src in sorted(set(map(str, missing_src_paths))):
            fname = Path(src).name
            url = base_url.rstrip("/") + "/" + fname
            dst = dest / fname

            if dst.exists() and skip_if_exists:
                # already present locally (perhaps from a previous run)
                downloaded.append(str(dst))
                print(f"[sweep]  -> already present after mirror check: {dst}")
                continue

            print(f"[sweep]  -> downloading from mirror: {url}")
            last_err = None
            for attempt in range(1, retries + 2):  # e.g. retries=2 -> up to 3 tries
                try:
                    # polite UA; some mirrors dislike default Python UA
                    req = Request(url, headers={"User-Agent": "Streamcutter/1.0 (urllib)"})
                    with urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
                        while True:
                            chunk = r.read(1024 * 1024)  # 1 MiB chunks
                            if not chunk:
                                break
                            f.write(chunk)
                    print(f"[sweep]  -> downloaded to: {dst}")
                    downloaded.append(str(dst))
                    last_err = None
                    break
                except (HTTPError, URLError, TimeoutError) as e:
                    last_err = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
                    print(f"[sweep]     mirror attempt {attempt} failed: {last_err}")
                    # clean up partial
                    try:
                        if dst.exists():
                            dst.unlink()
                    except Exception:
                        pass
                    if attempt <= retries:
                        time.sleep(0.5 * attempt)  # simple backoff
                        continue
                except Exception as e:
                    last_err = repr(e)
                    print(f"[sweep]     mirror attempt {attempt} failed: {last_err}")
                    try:
                        if dst.exists():
                            dst.unlink()
                    except Exception:
                        pass
                    if attempt <= retries:
                        time.sleep(0.5 * attempt)
                        continue

            if last_err:
                errors[str(src)] = last_err
                print(f"[sweep]  -> mirror download failed for {src}: {last_err}")

        return downloaded, errors


    @staticmethod
    def copy_files(
        src_paths: Iterable[str], dest_dir: str, skip_if_exists: bool = True
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Copy sweep bricks, but first probe each source with `ls` to surface remote FS errors
        like 'Cannot send after transport endpoint shutdown'. Any nonzero `ls` exit code
        is treated as 'missing' and skipped.
        """
        import subprocess
        import shutil
        from pathlib import Path
        import os

        # How long to wait for `ls` before skipping (seconds)
        LS_TIMEOUT_S = float(os.environ.get("SWEEP_LS_TIMEOUT_S", "2.0"))

        def _ls_ok(path: str) -> Tuple[bool, str]:
            """
            Probe with `/bin/ls -ld` in a separate process.
            Decide by message content; never hang the caller.
            """
            import os, subprocess, multiprocessing as mp

            TIMEOUT_S = float(os.environ.get("SWEEP_LS_TIMEOUT_S", "20.0"))

            def _worker(pth: str, q: "mp.Queue[str]") -> None:
                env = dict(os.environ)
                env.setdefault("LC_ALL", "C")  # stable diagnostics
                try:
                    r = subprocess.run(
                        ["/bin/ls", "-ld", "--", pth],   # absolute ls
                        capture_output=True,
                        text=True,
                        check=False,
                        env=env,
                    )
                    msg_full = ((r.stderr or "") + "\n" + (r.stdout or "")).strip()
                    q.put(msg_full)
                except Exception as e:
                    q.put(f"ls exception: {e!r}")

            q: "mp.Queue[str]" = mp.Queue(maxsize=1)
            proc = mp.Process(target=_worker, args=(path, q), daemon=True)
            proc.start()
            proc.join(TIMEOUT_S)

            if proc.is_alive():
                # Don’t wait on a wedged child; abandon and move on.
                try:
                    proc.terminate()
                except Exception:
                    pass
                return False, f"ls timed out after {TIMEOUT_S:.1f}s"

            try:
                msg_full = q.get_nowait()
            except Exception:
                return False, "ls produced no output"

            low = msg_full.lower()
            # Your real cluster outputs look like:
            # '/bin/ls: cannot access ...: Cannot send after transport endpoint shutdown'
            if (
                "transport endpoint" in low
                or "cannot access" in low
                or "no such file or directory" in low
                or low.startswith("ls:")
                or low.startswith("/bin/ls:")
            ):
                return False, msg_full

            # If it printed a normal mode line (starts with -, d, or l), treat as OK
            # even if return code was weird (we’re matching text only).
            if any(line.lstrip().startswith(("-", "d", "l")) for line in msg_full.splitlines()):
                return True, msg_full

            # Otherwise, assume OK if nothing suspicious showed up.
            return True, msg_full



        print(f"[sweep] copying {len(set(src_paths))} bricks to: {dest_dir}")
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        copied, skipped, missing = [], [], []

        for src in sorted(set(map(str, src_paths))):
            print(f"[sweep] processing: {src}")
            sp = Path(src)

            # Probe with `ls` to detect flaky remote mount errors
            ok, ls_msg = _ls_ok(str(sp))
            print(f"[sweep]  -> ls check: ok={ok} msg='{ls_msg}'")
            if not ok:
                print(f"[sweep]  -> ls failed: {ls_msg} ; skipping {sp}")
                missing.append(str(sp))
                continue

            # Optional extra guard (cheap and won't hang under normal conditions)
            if not sp.is_file():
                print(f"[sweep]  -> missing source file (not a file): {sp}")
                missing.append(str(sp))
                continue

            dp = dest / sp.name
            if dp.exists() and skip_if_exists:
                print(f"[sweep]  -> skipping existing: {dp}")
                skipped.append(str(dp))
                continue

            try:
                shutil.copy2(str(sp), str(dp))
                print(f"[sweep]  -> copied to: {dp}")
                copied.append(str(dp))
            except Exception as e:
                # If copy fails despite ls passing, treat as missing to keep signature
                print(f"[sweep]  -> copy failed ({e}); skipping {sp}")
                missing.append(str(sp))
                # best-effort cleanup of partial
                try:
                    if dp.exists():
                        dp.unlink()
                except Exception:
                    pass
                continue

        return copied, skipped, missing


    def fetch(
        self,
        ra,
        dec,
        dest_dir: str = "./data/sweep_catalogue",
        min_count: int = 50,
        skip_if_exists: bool = True,
        dry_run: bool = False,
        mirror_base_url: str = "https://casdc.china-vo.org/mirror/DESI/cosmo/data/legacysurvey/dr9/south/sweep/9.0/",
    ) -> Dict[str, object]:
        """
        Copy needed sweep bricks from their original locations; if any are missing or
        inaccessible, try to download by basename from the China-VO mirror.
        """
        selected, counts = self.select_files(ra, dec, min_count=min_count)
        copied: list[str] = []
        skipped: list[str] = []
        missing: list[str] = []
        downloaded: list[str] = []
        dl_errors: dict[str, str] = {}

        if not dry_run:
            # First try to copy from source paths
            copied, skipped, missing = self.copy_files(
                selected, dest_dir, skip_if_exists=skip_if_exists
            )

            # Then try to download the ones that were missing
            if missing:
                dl_ok, dl_errs = self._download_missing_from_mirror(
                    missing_src_paths=missing,
                    dest_dir=dest_dir,
                    base_url=mirror_base_url,
                    skip_if_exists=skip_if_exists,
                    retries=2,
                    timeout=30.0,
                )
                downloaded.extend(dl_ok)
                dl_errors.update(dl_errs)
                # Anything that we failed to download stays "missing"
                still_missing = set(missing) - set(dl_errs.keys())
                # `still_missing` are those successfully downloaded; remove them
                missing = [m for m in missing if m in dl_errs]

        return {
            "selected_files": selected,
            "counts": counts,
            "copied_files": copied,
            "skipped_existing": skipped,
            "missing_source_files": missing,   # after mirror attempts
            "downloaded_files": downloaded,    # new: from mirror
            "download_errors": dl_errors,      # new: per-src error message
        }