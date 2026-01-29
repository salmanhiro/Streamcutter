#!/usr/bin/python
'''
Simulation of a disrupting satellite galaxy orbiting inside a host galaxy and
exposed to tidal stripping and dynamical friction.
The most straightforward approach would be to represent both galaxies as N-body
systems and evolve them with a conventional N-body code (in this example, we use
pyfalcon - the Python interface to Gyrfalcon - a fast-multipole code by W.Dehnen).
But this is rather expensive, since the host galaxy would need to be resolved
by many more particles than the satellite in order to correctly simulate
the dynamical friction (DF).
Instead, we rely on the Chandrasekhar approximation of the DF force, and simulate
only the satellite as an N-body system. It is embedded in a static external
potential of the host galaxy, which creates a realistic tidal force, and we add
the DF force manually, using the instantaneous position, velocity and mass of
the satellite galaxy recomputed at each timestep.
This approach is still expensive, so another approximation is put forward
under the name "restricted N-body simulation".
Here the satellite galaxy moves in the host galaxy as a single massive body
experiencing dynamical friction. To produce the tidal stream and to determine
the mass evolution of the satellite, it is also represented by test particles,
which do not interact between themselves, but move in the time-dependent
potential created by the host and the moving satellite.
The potential and mass of the satellite are then recomputed every so often
from these particles, closing the loop. The update interval is much longer
than the timestep of the full N-body simulation, because the orbits of particles
are evolved with high precision using an adaptive-timestep integrator, and
the frequency of updates should be sufficient to resolve the mass and structural
evolution of the satellite rather than particle trajectories themselves.
Despite the obviously approximate nature of this approach, it produces
a realistic tidal stream and roughly matches the evolution of the system
in the full N-body + DF simulation.
This script implements both approaches, but the full N-body is run only if
pyfalcon is available.
'''
# From E.Vasiliev
import os, numpy, agama, scipy.special, scipy.integrate, matplotlib.pyplot as plt
import pyfalcon
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric, ICRS, CartesianRepresentation, CartesianDifferential
plot = True
import numpy as np

agama.setUnits(length=1, velocity=1, mass=1)

pot_host = agama.Potential("potentials/MWPotential2014.ini")  # alternative: a more realistic Milky Way potential


# prepare an interpolation table for the host velocity dispersion profile
df_host = agama.DistributionFunction(type='quasispherical', potential=pot_host)
grid_r  = numpy.logspace(-1, 2, 16)  # grid from 0.1 to 100 kpc
grid_sig= agama.GalaxyModel(pot_host, df_host).moments(
    numpy.column_stack((grid_r, grid_r*0, grid_r*0)), dens=False, vel=False, vel2=True)[:,0]**0.5
logspl  = agama.Spline(numpy.log(grid_r), numpy.log(grid_sig))  # log-scaled spline
sigma   = lambda r: numpy.exp(logspl(numpy.log(r)))   # and the un-scaled interpolator

# initial potential of the satellite (a single Dehnen component with a Gaussian cutoff)
pot_sat  = agama.Potential(type='Plummer', scaleRadius=0.05, mass=1e5)
initmass = pot_sat.totalMass()

# create a spherical isotropic DF for the satellite and sample it with particles
df_sat = agama.DistributionFunction(type='quasispherical', potential=pot_sat)
Nbody = 10000
xv, mass = agama.GalaxyModel(pot_sat, df_sat).sample(Nbody)

# acquired from orbital simulation
r_center = numpy.array([
    9.87557601928711,   11.236837387084961,  5.983392238616943,   # x, y, z   [kpc]
    -13.987908363342285, -124.69666290283203, 130.86196899414062   # vx, vy, vz [km/s]
], dtype=float)

# shift particle ICs into the Galactocentric frame at Pal 5's current phase-space point
xv += r_center

# parameters for the simulation
tend = 1.0   # total simulation time
tupd = 2**-3 # interval for plotting and updating the satellite mass for the restricted N-body simulation
tau  = 2**-9 # timestep of the full N-body sim (typically should be smaller than eps/v, where v is characteristic internal velocity)
eps  = 0.01  # softening length for the full N-body simulation

def dynfricAccel(pos, vel, mass):
    # compute the Chandrasekhar's dynamical friction acceleration for a point mass in the host galaxy
    r   = sum(pos**2)**0.5
    v   = sum(vel**2)**0.5
    rho = pot_host.density(pos)
    coulombLog = 3.0
    X = v / (2**0.5 * sigma(r))
    return -vel / v * (4*numpy.pi * agama.G**2 * mass * rho * coulombLog *
        (scipy.special.erf(X) - 2/numpy.pi**.5 * X * numpy.exp(-X*X)) / v**2)

def orbitDF(ic, time, timestart, trajsize, mass):
    # integrate the orbit of a massive particle in the host galaxy, accounting for dynamical friction
    if mass == 0:
        return agama.orbit(ic=ic, potential=pot_host, time=time, timestart=timestart, trajsize=trajsize)
    times = numpy.linspace(timestart, timestart+time, trajsize)
    traj = scipy.integrate.odeint(
        lambda xv, t: numpy.hstack((xv[3:6], pot_host.force(xv[0:3], t=t) + dynfricAccel(xv[0:3], xv[3:6], initmass) )),
        ic, times)

    # traj = scipy.integrate.odeint(
    #     lambda xv, t: numpy.hstack((xv[3:6], pot_host.force(xv[0:3], t=t))),
    #     ic, times)
    return times, traj

def orbit_host_agama(ic, time, timestart, trajsize):
    # Agama orbit integrator in the given potential
    times, traj = agama.orbit(ic=ic, potential=pot_host, time=time,
                             timestart=timestart, trajsize=trajsize)
    return times, traj


# simulate the evolution of the disrupting satellite using two methods:
# "restricted N-body" (r_ prefix) and "full N-body" (if available, f_ prefix)

r_mass   = [initmass]
r_traj   = [r_center]
r_xv     = xv.copy()
time     = 0.0   # current simulation time
times_t  = [time]
times_u  = [time]
f_center = r_center.copy()
f_mass   = [initmass]
f_traj   = [f_center]
f_xv     = xv.copy()
f_bound  = numpy.ones(len(xv), bool)

plt.figure(figsize=(9,6), dpi=100)
ax1=plt.axes([0.07, 0.08, 0.36, 0.54])
ax2=plt.axes([0.57, 0.08, 0.36, 0.54])
bx1=plt.axes([0.07, 0.72, 0.36, 0.24])
bx2=plt.axes([0.57, 0.72, 0.36, 0.24])
cx1=bx1.twinx()
cx2=bx2.twinx()
if plot:
    plt.ion()

print('time  mass' + ('  mass(Nbody)' if pyfalcon else ''))
while time < tend:
    # Method 1: restricted N-body
    # first determine the trajectory of the satellite centre in the host potential
    # (assuming that it moves as a single massive particle)
    time_center, orbit_center = orbitDF(ic=r_center, time=tupd, timestart=time, trajsize=round(tupd/tau) + 1, mass=r_mass[-1])
    
    #time_center, orbit_center = orbit_host_agama(ic=r_center, time=tupd, timestart=time,
    #    trajsize=round(tupd/tau) + 1
    #)
    times_u.append(time_center[-1])
    times_t.extend(time_center[1:])
    r_traj.extend(orbit_center[1:])
    r_center = orbit_center[-1]  # current position and velocity of satellite CoM
    # initialize the time-dependent total potential (host + moving sat) on this time interval
    pot_total = agama.Potential(pot_host,
        agama.Potential(potential=pot_sat, center=numpy.column_stack((time_center, orbit_center))))
    # compute the trajectories of all particles moving in the combined potential of the host galaxy and the moving satellite
    r_xv = numpy.vstack(agama.orbit(ic=r_xv, potential=pot_total, time=tupd, timestart=time, trajsize=1)[:,1])
    # update the potential of the satellite (using a spherical monopole approximation)
    pot_sat = agama.Potential(type='multipole', particles=(r_xv[:,0:3] - r_center[0:3], mass), symmetry='s')
    # determine which particles remain bound to the satellite
    r_bound = pot_sat.potential(r_xv[:,0:3] - r_center[0:3]) + 0.5 * numpy.sum((r_xv[:,3:6] - r_center[3:6])**2, axis=1) < 0
    r_mass.append(numpy.sum(mass[r_bound]))

    # Method 2: full N-body
    if time==0:   # initialize accelerations and potential
        f_acc, f_pot = pyfalcon.gravity(f_xv[:,0:3], agama.G * mass, eps)
        f_acc += pot_host.force(f_xv[:,0:3]) + dynfricAccel(f_center[0:3], f_center[3:6], initmass)

    # advance the N-body sim in smaller steps
    f_time = 0
    while f_time < tupd:
        # kick-drift-kick leapfrog method:
        # kick for half-step, using accelerations computed at the end of the previous step
        f_xv[:,3:6] += f_acc * (tau/2)
        # drift for full step
        f_xv[:,0:3] += f_xv[:,3:6] * tau
        # recompute accelerations from self-gravity of the satellite
        # NB: falcON works with natural N-body units in which G=1, so we multiply particle mass passed to falcon by G
        f_acc, f_pot = pyfalcon.gravity(f_xv[:,0:3], agama.G * mass, eps)
        # add accelerations from the host galaxy
        f_acc += pot_host.force(f_xv[:,0:3])
        # add acceleration from dynamical friction (for simplicity, to all particles, not only the bound ones)
        f_acc += dynfricAccel(f_center[0:3], f_center[3:6], numpy.sum(mass[f_bound]))
        # kick again for half-step
        f_xv[:,3:6] += f_acc * (tau/2)

        # recompute the location and velocity of the satellite centre and its remaining bound mass
        f_center[0:3] += tau * f_center[3:6]  # linearly extrapolate from the previous timestep to get the first estimate
        Rmax = 10.0
        use  = numpy.sum((f_xv[:,0:3] - f_center[0:3])**2, axis=1) < Rmax**2
        # iteratively refine the selection, retaining only bound particles (which have
        # negative total energy in the satellite-centered frame using its own potential only)
        prev_f_center = f_center
        for i in range(10):
            f_center = numpy.median(f_xv[use], axis=0)
            f_bound = f_pot + 0.5 * numpy.sum((f_xv[:,3:6] - f_center[3:6])**2, axis=1) < 0
            if numpy.sum(f_bound)<=1 or all(f_center==prev_f_center): break
            use = f_bound * (numpy.sum((f_xv[:,0:3] - f_center[0:3])**2, axis=1) < Rmax**2)
            prev_f_center = f_center

        f_traj.append(f_center)
        f_mass.append(numpy.sum(mass[f_bound]))
        f_time += tau

    time += tupd
    print('%5.3f  %.4g' % (time, r_mass[-1]/initmass) + ('  %.4g' % (f_mass[-1]/initmass) if pyfalcon else ''))

    if plot or time==tend:
        ax1.cla()
        ax2.cla()
        bx1.cla()
        bx2.cla()
        cx1.cla()
        cx2.cla()
        bx1.text(0.5, 0.1, 'restricted N-body', ha='center', va='bottom', transform=bx1.transAxes)
        bx2.text(0.5, 0.1, 'full N-body' + ('' if pyfalcon else ' (n/a)'), ha='center', va='bottom', transform=bx2.transAxes)
        ax1.scatter(r_xv[:,0], r_xv[:,1], marker='o', s=1, linewidths=0, edgecolors='none',
            c=r_bound, cmap='bwr_r', vmin=0, vmax=1)  # blue: bound, red: unbound particles
        ax1.plot(numpy.vstack(r_traj)[:,0], numpy.vstack(r_traj)[:,1], 'g')
        if pyfalcon:
            ax2.scatter(f_xv[:,0], f_xv[:,1], marker='o', s=1, linewidths=0, edgecolors='none',
                c=f_bound, cmap='bwr_r', vmin=0, vmax=1)
            ax2.plot(numpy.vstack(f_traj)[:,0], numpy.vstack(f_traj)[:,1], 'g')
        ax1.set_xlim(-20,20)
        ax1.set_ylim(-20,20)
        ax2.set_xlim(-20,20)
        ax2.set_ylim(-20,20)
        ax1.set_xlabel('x')
        ax2.set_xlabel('x')
        ax1.set_ylabel('y', labelpad=0)
        ax2.set_ylabel('y', labelpad=0)
        bx1.plot(times_t, numpy.sum(numpy.vstack(r_traj)[:,0:2]**2, axis=1)**0.5, 'r')
        cx1.plot(times_u, r_mass, 'b')
        if pyfalcon:
            bx2.plot(times_t, numpy.sum(numpy.vstack(f_traj)[:,0:2]**2, axis=1)**0.5, 'r')
            cx2.plot(times_t, f_mass, 'b')
        bx1.set_ylim(0, 20)
        bx2.set_ylim(0, 20)
        cx1.set_ylim(0, initmass)
        cx2.set_ylim(0, initmass)
        bx1.set_xlim(0, tend)
        bx2.set_xlim(0, tend)
        bx1.set_xlabel('time')
        bx2.set_xlabel('time')
        bx1.set_ylabel('distance')
        bx2.set_ylabel('distance')
        cx1.set_ylabel('bound mass')
        cx2.set_ylabel('bound mass')
        cx1.yaxis.set_label_position('right')
        cx2.yaxis.set_label_position('right')
        plt.draw()
        plt.pause(.01)

plt.savefig('full_nbody.png')


def xv_to_icrs(xv, galcen_frame):
    """
    xv: (N,6) array in Galactocentric Cartesian:
        [x,y,z] in kpc, [vx,vy,vz] in km/s
    returns: SkyCoord in ICRS
    """
    xv = np.asarray(xv)
    if xv.ndim == 1:
        xv = xv.reshape(1, 6)
    if xv.shape[1] != 6:
        raise ValueError(f"xv must be (N,6); got {xv.shape}")

    rep = CartesianRepresentation(
        x=xv[:,0]*u.kpc, y=xv[:,1]*u.kpc, z=xv[:,2]*u.kpc,
        differentials=CartesianDifferential(
            d_x=xv[:,3]*u.km/u.s,
            d_y=xv[:,4]*u.km/u.s,
            d_z=xv[:,5]*u.km/u.s
        )
    )
    c_gal = SkyCoord(rep, frame=galcen_frame)
    return c_gal.icrs

galcen_frame = Galactocentric()

c_r = xv_to_icrs(r_xv, galcen_frame)
ra_r, dec_r = c_r.ra.deg, c_r.dec.deg

have_full = ("f_xv" in globals()) and (f_xv is not None) and (len(f_xv) == len(r_xv))
if have_full:
    c_f = xv_to_icrs(f_xv, galcen_frame)
    ra_f, dec_f = c_f.ra.deg, c_f.dec.deg

# Convert COM positions too
c_rcom = xv_to_icrs(r_center, galcen_frame)
ra_rcom, dec_rcom = c_rcom.ra.deg[0], c_rcom.dec.deg[0]

if have_full:
    c_fcom = xv_to_icrs(f_center, galcen_frame)
    ra_fcom, dec_fcom = c_fcom.ra.deg[0], c_fcom.dec.deg[0]

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
axL, axR = axes

# Restricted
axL.scatter(ra_r, dec_r, s=1, c=r_bound, cmap="bwr_r", vmin=0, vmax=1, linewidths=0)
axL.scatter([ra_rcom], [dec_rcom], s=80, marker="*", edgecolors="k")  # COM marker
axL.set_title("Restricted (ICRS)")
axL.set_xlabel("RA [deg]")
axL.set_ylabel("Dec [deg]")
axL.invert_xaxis()

# Full N-body
if have_full:
    axR.scatter(ra_f, dec_f, s=1, c=f_bound, cmap="bwr_r", vmin=0, vmax=1, linewidths=0)
    axR.scatter([ra_fcom], [dec_fcom], s=80, marker="*", edgecolors="k")  # COM marker
else:
    axR.text(0.5, 0.5, "Full N-body not available", ha="center", va="center", transform=axR.transAxes)

axR.set_title("Full N-body (ICRS)")
axR.set_xlabel("RA [deg]")
axR.invert_xaxis()

fig.tight_layout()
fig.savefig("icrs_restricted_vs_full_with_com.png", dpi=200)
