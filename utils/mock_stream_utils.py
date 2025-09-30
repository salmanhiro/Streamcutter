import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Galactocentric, CartesianDifferential, CartesianRepresentation, ICRS
import agama
from utils.coordinate_utils import get_observed_coords


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
