from __future__ import annotations
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Galactocentric, CartesianDifferential, CartesianRepresentation, ICRS
import agama
from utils.coordinate_utils import get_observed_coords
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
from utils import coordinate_utils


def galactocentric_to_icrs(x, y, z, vx, vy, vz):
    
    galactocentric_frame = Galactocentric()
    c = SkyCoord(x=x,
                 y=y,
                 z=z,
                 v_x = vx,
                 v_y = vy,
                 v_z = vz,
                 frame=galactocentric_frame)
    icrs_coord = c.transform_to('icrs')
    return icrs_coord.ra.value, icrs_coord.dec.value, icrs_coord.distance, icrs_coord.radial_velocity



def get_rotational_matrix(x, y, z, vx, vy, vz):
    """
    Compute rotation matrices, angular momentum magnitudes, and radii for transforming from the host to the satellite frame.

    Parameters
    ----------
    x, y, z : Positions of the satellite.
    vx, vy, vz : Velocities of the satellite.

    Returns
    -------
    R : Rotation matrices for each point.
    L : Angular momentum magnitudes.
    r : Distances (radii) for each point.
    """

    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx
    r = (x*x + y*y + z*z)**0.5
    L = (Lx*Lx + Ly*Ly + Lz*Lz)**0.5
    R = np.zeros((len(x), 3, 3))
    R[:,0,0] = x/r
    R[:,0,1] = y/r
    R[:,0,2] = z/r
    R[:,2,0] = Lx/L
    R[:,2,1] = Ly/L
    R[:,2,2] = Lz/L
    R[:,1,0] = R[:,0,2] * R[:,2,1] - R[:,0,1] * R[:,2,2]
    R[:,1,1] = R[:,0,0] * R[:,2,2] - R[:,0,2] * R[:,2,0]
    R[:,1,2] = R[:,0,1] * R[:,2,0] - R[:,0,0] * R[:,2,1]
    return R, L, r

def get_d2Phi_dr2(pot_host, x, y, z): 
    """
    Compute the second derivative of the gravitational potential with respect to radius at given positions.

    Parameters
    ----------
    pot_host : Host galaxy potential.
    x, y, z : Positions to evaluate the derivative.

    Returns
    -------
    d2Phi_dr2 : Second derivative of the potential at each position.
    """
    r = (x*x + y*y + z*z)**0.5 #radius
    der = pot_host.forceDeriv(np.column_stack([x,y,z]))[1]
    d2Phi_dr2 = -(x**2  * der[:,0] + y**2  * der[:,1] + z**2  * der[:,2] +
                  2*x*y * der[:,3] + 2*y*z * der[:,4] + 2*z*x * der[:,5]) / r**2
    return d2Phi_dr2

# For comparison Fardal+15 method
# Originally implemented by Eugene Vasiliev
def create_initial_condition_fardal15(rng, pot_host, orb_sat, mass_sat, gala_modified=True):
    N = len(orb_sat)
    x, y, z, vx, vy, vz = orb_sat.T
    R, L, r = get_rotational_matrix(x, y, z, vx, vy, vz)
    d2Phi_dr2 = get_d2Phi_dr2(pot_host, x, y, z)
    
    # compute the Jacobi radius and the relative velocity at this radius for each point on the trajectory
    Omega = L / r**2
    rj = (agama.G * mass_sat / (Omega**2 - d2Phi_dr2))**(1./3)
    vj = Omega * rj
    
    # assign positions and velocities (in the satellite reference frame) of particles
    # leaving the satellite at both lagrange points.
    rj = np.repeat(rj, 2) * np.tile([1, -1], N)
    vj = np.repeat(vj, 2) * np.tile([1, -1], N)
    mean_x  = 2.0
    disp_x  = 0.5 if gala_modified else 0.4
    disp_z  = 0.5
    mean_vy = 0.3
    disp_vy = 0.5 if gala_modified else 0.4
    disp_vz = 0.5
    
    rx  = rng.normal(size=2*N) * disp_x + mean_x
    rz  = rng.normal(size=2*N) * disp_z * rj
    rvy =(rng.normal(size=2*N) * disp_vy + mean_vy) * vj * (rx if gala_modified else 1)
    rvz = rng.normal(size=2*N) * disp_vz * vj
    rx *= rj
    ic_stream = np.tile(orb_sat, 2).reshape(2*N, 6)
    ic_stream[:,0:3] += np.einsum('ni,nij->nj',
        np.column_stack([rx,  rx*0, rz ]), np.repeat(R, 2, axis=0))
    ic_stream[:,3:6] += np.einsum('ni,nij->nj',
        np.column_stack([rx*0, rvy, rvz]), np.repeat(R, 2, axis=0))
    return ic_stream

def create_stream(create_ic_method, rng, time_total, num_particles, pot_host, posvel_sat, mass_sat, pot_sat=None, **kwargs):
    """
    Generate a tidal stream by simulating the orbital trajectory of a progenitor and creating particles released at its Lagrange points.

    Parameters
    ----------
    create_ic_method : A function to generate initial conditions (Fardal or Chen)
    rng : Random number generator instance used for initializing particle positions and velocities.
    time_total : Total integration time for the progenitor's orbit. A negative value integrates the orbit backward in time.
    num_particles : Number of particles to generate for the tidal stream.
    pot_host : The gravitational potential of the host galaxy.
    posvel_sat : The initial 6D phase-space coordinates (position and velocity) of the progenitor at the present time.
    mass_sat : The mass of the progenitor satellite.
    pot_sat : The gravitational potential of the progenitor satellite. If `None`, the satellite's potential is neglected. (optional)
    **kwargs : Additional parameters passed to `create_ic_method`. (optional)

    Returns
    -------
    time_sat : 1D array of time points along the progenitor's orbit.
    orbit_sat : 2D array of shape (num_steps, 6) representing the progenitor's orbit (position and velocity at each time step).
    xv_stream : 2D array of shape (num_particles, 6) representing the 6D phase-space coordinates (position and velocity) of the particles in the tidal stream.
    ic_stream : 2D array of shape (num_particles, 6) containing the initial conditions of the particles released from the progenitor at the Lagrange points.
    """
    
    # integrate the orbit of the progenitor from its present-day posvel (at time t=0)
    # back in time for an interval time_total, storing the trajectory at num_steps points
    # here the potential of satellite can be neglected
    time_sat, orbit_sat = agama.orbit(potential=pot_host, ic=posvel_sat,
        time=time_total, trajsize=num_particles//2)

    # plt.plot(orbit_sat[:,0], orbit_sat[:,1])
    
    if time_total < 0:
        # reverse the arrays to make them increasing in time
        time_sat  = time_sat [::-1]
        orbit_sat = orbit_sat[::-1]

    # at each point on the trajectory, create a pair of seed initial conditions
    # for particles released at Lagrange points
    ic_stream = create_ic_method(rng, pot_host, orbit_sat, mass_sat, **kwargs)

    time_seed = np.repeat(time_sat, 2)
    
    if pot_sat is None:
        pot_tot = pot_host
    else:
        # include the progenitor's potential
        traj = np.column_stack([time_sat, orbit_sat])
        pot_traj = agama.Potential(potential=pot_sat, center=traj)
        pot_tot = agama.Potential(pot_host, pot_traj)
        
    xv_stream = np.vstack(agama.orbit(potential=pot_tot,
        ic=ic_stream, time=-time_seed if time_total<0 else time_total-time_seed, timestart=time_seed, trajsize=1)[:,1])
    return time_sat, orbit_sat, xv_stream, ic_stream

class StreamSimulator:
    """Simulate a tidal stream (restricted or full N-body) and return phase-space + observables."""
    def __init__(
        self,
        factory,
        sim_mode: str = "restricted",   # "restricted" (default) or "full"
        # knobs for FULL N-body mode (ignored in restricted)
        vcirc10: float = 200.0,         # host normalization: v_circ(10 kpc) [km/s]
        eps: float = 0.1,               # softening length [kpc]
        tau: Optional[float] = None,    # leapfrog step in AGAMA time units (kpc / km/s)
        use_df: bool = True,            # apply Chandrasekhar DF to CoM in full mode
        total_time: Optional[float] = None,  # total time in Gyr (if provided)
    ):
        self.factory  = factory
        self.sim_mode = sim_mode
        self.vcirc10  = float(vcirc10)
        self.eps      = float(eps)
        self.tau      = tau
        self.use_df   = bool(use_df)
        self.total_time = total_time  # Store total time if provided

    @staticmethod
    def _total_time_gyr(orbit_period_max: float, n_orbits: int) -> float:
        # Keep your original heuristic:
        return (-n_orbits * orbit_period_max) / 978.0 if orbit_period_max < 1000 else -3.0

    def simulate(
        self,
        gc_row: QTable,
        potential_name: str,
        n_particles: int,
        n_orbits: Optional[int] = 3,
        total_time: Optional[float] = None,
        rng_seed: int = 0,
        sim_mode: Optional[str] = None,  # optional per-call override
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, QTable]:
        mode = (sim_mode or self.sim_mode).lower()
        if mode == "full":
            # Not yet implemented
            raise NotImplementedError("Full N-body stream simulation is not yet implemented in pipeline.")
        return self._simulate_restricted(gc_row, potential_name, n_particles, n_orbits, rng_seed, total_time)

    def _simulate_restricted(
        self,
        gc_row: QTable,
        potential_name: str,
        n_particles: int,
        n_orbits: int,
        rng_seed: int,
        total_time: Optional[float] = None,  # Added parameter
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        dist: Optional[float] = None,
        pmra: Optional[float] = None,
        pmdec: Optional[float] = None,
        mass: Optional[float] = None,
        rv: Optional[float] = None,
        rhm: Optional[float] = None,
        orbit_t: Optional[float] = None,
    ):
        row = gc_row[0]

        # Extract GC info (match your column names) or use defaults
        ra_q   = ra if ra is not None else row["RA"].value
        dec_q  = dec if dec is not None else row["DEC"].value
        dist_q = dist if dist is not None else row["Rsun"].value
        pmra   = pmra if pmra is not None else row["mualpha"].value
        pmdec  = pmdec if pmdec is not None else row["mu_delta"].value
        mass   = mass if mass is not None else row["Mass"].value
        rv     = rv if rv is not None else row["<RV>"].value
        rhm    = rhm if rhm is not None else row["rh,m"].value
        orbit_t = orbit_t if orbit_t is not None else row["orbit_period_max"].value
        print(f"This GC orbital period is {orbit_t:.2f} Myr")

        # Potentials & initial conditions (your factory)
        pot_host = self.factory.host(potential_name)
        pot_sat  = self.factory.satellite_plummer(mass, rhm)
        prog_w0  = coordinate_utils.get_galactocentric_coords(ra_q, dec_q, dist_q, rv, pmra, pmdec)[0]

        # Determine total time
        if total_time is not None:
            time_total = total_time  / 978.0 # Use provided total time directly
        elif self.total_time is not None:
            time_total = self.total_time  / 978.0# Use stored total time
        else:
            time_total = self._total_time_gyr(orbit_t, n_orbits)  # Calculate from orbit

        # Simulate (your existing helper)
        time_sat, orbit_sat, xv_stream, ic_stream = create_stream(
            create_initial_condition_fardal15,
            np.random.default_rng(rng_seed),
            time_total, n_particles,
            pot_host, prog_w0, mass,
            pot_sat=pot_sat
        )

        # To observables table (unchanged columns)
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

        sim_stream_meta = {
            "gc_name": row["Cluster"],
            "potential": potential_name,
            "n_particles": n_particles,
            "n_orbits": n_orbits,
            "rng_seed": rng_seed,
            "mass_sat": mass,
            "rhm": rhm,
            "time_total_gyr": time_total * 978.0,
        }

        return time_sat, orbit_sat, xv_stream, ic_stream, sim_stream_tab, sim_stream_meta