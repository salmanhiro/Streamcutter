#!/usr/bin/env python3
"""
Pal 5 toy: back-integrate the observed orbit by 1 Gyr, then run a restricted N-body
stream simulation forward to the present and report the final position error.

Requirements: numpy, scipy, matplotlib, astropy, agama (and optionally pyfalcon for full N-body)
"""

import os, numpy as np, scipy.special, scipy.integrate, matplotlib.pyplot as plt
import agama
import pyfalcon

import os, numpy as np, scipy.special, scipy.integrate
import matplotlib
import matplotlib.pyplot as plt
import agama

import astropy.units as u
from astropy.coordinates import (
    SkyCoord, Galactocentric,
    CartesianRepresentation, CartesianDifferential
)

# where to save stuff
OUTDIR   = os.environ.get("PAL5_OUT", "pal5_outputs")
FRAMEDIR = os.path.join(OUTDIR, "frames_side_by_side")
os.makedirs(FRAMEDIR, exist_ok=True)

import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric, CartesianDifferential

plot_movie = True                     # True: live movie while running
T_BACK_GYR = 1.0                      # back-integration time [Gyr]
Nbody = 1000
GC_mass     = 3.0e4                   # Msun (toy; tune as you like)
GC_scale_kpc= 0.020                   # ~20 pc Plummer-like scale
GC_r_cut_kpc= 0.080                   # outer cutoff (~tidal-ish)
# Pal 5 observed “today” (fill with literature values you prefer)
PAL5 = dict(
    ra_deg   = 229.018,               # deg
    dec_deg  = -0.124,                # deg
    dist_kpc = 23.5,                  # kpc
    pmra_masyr  = -2.736,             # mas/yr  (μ_α*)
    pmdec_masyr = -2.646,             # mas/yr
    vlos_kms = -58.7                  # km/s (heliocentric, +away)
)
# Solar position/velocity for transforms
SOLAR = dict(
    R0_kpc = 8.122,
    z_sun_kpc = 0.0208,
    v_sun_uvw_kms = (11.1, 248.5, 7.25)   # (U,V,W)⊙; pick your favorite set
)

# Working units: 1 kpc, 1 km/s, 1 Msun => 1 time unit = kpc/(km/s) ≈ 0.9777922217 Gyr
agama.setUnits(length=1, velocity=1, mass=1)
UNIT_T_GYR = 0.97779222168
def gyr_to_code(t_gyr): return t_gyr / UNIT_T_GYR

# ================ HOST POTENTIAL: Dehnen & Binney (1998) Model 1 ================
def make_DB98_potential():
    comps = [
        # --- Discs ---
        dict(type='disk', surfacedensity=4.76e8, scaleradius=2.0, scaleheight=0.04, innercutoffradius=4.0),  # ISM
        dict(type='disk', surfacedensity=1.334e9, scaleradius=2.0, scaleheight=0.18),                         # Thin
        dict(type='disk', surfacedensity=9.53e7, scaleradius=2.0, scaleheight=1.0),                           # Thick
        # --- Bulge ---
        dict(type='spheroid', densitynorm=4.27e8, axisratioz=0.6, gamma=1.8, beta=1.8,
             scaleradius=1.0, outercutoffradius=1.9),
        # --- Halo ---
        dict(type='spheroid', densitynorm=7.11e8, axisratioz=0.8, gamma=-2.0, beta=2.959,
             scaleradius=3.83, outercutoffradius=1000.0),
    ]
    # Units already set to Msun, kpc, km/s above.
    return agama.Potential(*comps)

pot_host = make_DB98_potential()

# (optional sanity check)
Rcheck = SOLAR['R0_kpc']
Fr = np.dot(pot_host.force([Rcheck, 0, 0]), np.array([1.0, 0.0, 0.0]))  # radial force at (R,0,0)
Vc = np.sqrt(-Fr * Rcheck) if Fr < 0 else np.nan
print(f"[DB98] Vc(R0={Rcheck:.3f} kpc) ≈ {Vc:.1f} km/s")

def dynfricAccel(pos, vel, mass_for_df):
    """
    Chandrasekhar DF acceleration in the composite DB98 potential.
    Uses a simple sigma_eff ~ Vc/sqrt(2) estimated from the local circular speed.
    Returns zero if mass_for_df <= 0 (e.g., globular clusters).
    """
    if mass_for_df <= 0:
        return np.zeros_like(pos)

    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    if r == 0 or v == 0:
        return np.zeros_like(pos)

    # local radial force projection -> effective circular speed
    F = pot_host.force(pos)
    Fr = np.dot(F, pos) / r
    if Fr >= 0:
        return np.zeros_like(pos)  # unphysical corner case; bail out

    Vc = np.sqrt(-Fr * r)
    sigma_eff = Vc / np.sqrt(2.0)

    rho = pot_host.density(pos)
    lnL = 3.0
    X = v / (np.sqrt(2.0) * sigma_eff)
    pref = 4 * np.pi * agama.G**2 * mass_for_df * rho * lnL
    chi = scipy.special.erf(X) - (2 / np.sqrt(np.pi)) * X * np.exp(-X*X)
    return -(pref * chi / v**3) * vel


# Use a Dehnen with gamma=0,beta=5 (Plummer-like) plus a gentle cutoff
pot_sat  = agama.Potential(type='spheroid',
                           scaleradius=GC_scale_kpc, 
                           outercutoffradius=GC_r_cut_kpc, 
                           mass=GC_mass)
initmass = pot_sat.totalMass()
df_sat   = agama.DistributionFunction(type='quasispherical', potential=pot_sat)
xv_gc, pmass = agama.GalaxyModel(pot_sat, df_sat).sample(Nbody)   # (x,y,z,vx,vy,vz) + masses

def observed_pal5_galcen_state(PAL5, SOLAR):
    sc = SkyCoord(ra=PAL5['ra_deg']*u.deg, dec=PAL5['dec_deg']*u.deg,
                  distance=PAL5['dist_kpc']*u.kpc,
                  pm_ra_cosdec=PAL5['pmra_masyr']*u.mas/u.yr,
                  pm_dec=PAL5['pmdec_masyr']*u.mas/u.yr,
                  radial_velocity=PAL5['vlos_kms']*u.km/u.s)
    U,V,W = SOLAR['v_sun_uvw_kms']
    galcen_frame = Galactocentric(galcen_distance=SOLAR['R0_kpc']*u.kpc,
                                  z_sun=SOLAR['z_sun_kpc']*u.kpc,
                                  galcen_v_sun=CartesianDifferential(U*u.km/u.s, V*u.km/u.s, W*u.km/u.s))
    gc = sc.transform_to(galcen_frame)
    x,y,z = gc.cartesian.x.to_value(u.kpc), gc.cartesian.y.to_value(u.kpc), gc.cartesian.z.to_value(u.kpc)
    vx,vy,vz = gc.v_x.to_value(u.km/u.s), gc.v_y.to_value(u.km/u.s), gc.v_z.to_value(u.km/u.s)
    return np.array([x,y,z,vx,vy,vz]), sc  # also return SkyCoord(obs) for later comparison

def galcen_state_to_icrs(state, SOLAR):
    import astropy.units as u
    from astropy.coordinates import (
        SkyCoord, Galactocentric,
        CartesianRepresentation, CartesianDifferential
    )

    x, y, z, vx, vy, vz = state
    U, V, W = SOLAR['v_sun_uvw_kms']

    galcen_frame = Galactocentric(
        galcen_distance=SOLAR['R0_kpc']*u.kpc,
        z_sun=SOLAR['z_sun_kpc']*u.kpc,
        galcen_v_sun=CartesianDifferential(U*u.km/u.s, V*u.km/u.s, W*u.km/u.s)
    )

    rep  = CartesianRepresentation(x*u.kpc, y*u.kpc, z*u.kpc)
    diff = CartesianDifferential(vx*u.km/u.s, vy*u.km/u.s, vz*u.km/u.s)
    rep  = rep.with_differentials(diff)

    gc = galcen_frame.realize_frame(rep)
    return SkyCoord(gc).transform_to('icrs')

def galcen_particles_to_icrs(xv_array, SOLAR):
    """Vectorized Galactocentric (x,y,z,vx,vy,vz) -> ICRS SkyCoord for all particles."""

    x  = xv_array[:,0] * u.kpc
    y  = xv_array[:,1] * u.kpc
    z  = xv_array[:,2] * u.kpc
    vx = xv_array[:,3] * u.km/u.s
    vy = xv_array[:,4] * u.km/u.s
    vz = xv_array[:,5] * u.km/u.s

    rep  = CartesianRepresentation(x, y, z)
    diff = CartesianDifferential(vx, vy, vz)
    rep  = rep.with_differentials(diff)

    U, V, W = SOLAR['v_sun_uvw_kms']
    galcen_frame = Galactocentric(
        galcen_distance=SOLAR['R0_kpc']*u.kpc,
        z_sun=SOLAR['z_sun_kpc']*u.kpc,
        galcen_v_sun=CartesianDifferential(U*u.km/u.s, V*u.km/u.s, W*u.km/u.s)
    )

    gc = galcen_frame.realize_frame(rep)
    return SkyCoord(gc).transform_to('icrs')


obs_state_gc, obs_sc_icrs = observed_pal5_galcen_state(PAL5, SOLAR)

t_back_units = gyr_to_code(T_BACK_GYR)          # code time units
trajsize_back = max(512, int(256*T_BACK_GYR))   # reasonable resolution
times_back, traj_back = agama.orbit(ic=obs_state_gc, potential=pot_host,
                                    time=-t_back_units, trajsize=trajsize_back)
r_center = traj_back[-1]  # starting center (1 Gyr ago)
print(f"Back-integration done. Start @ t=-{T_BACK_GYR:.2f} Gyr: R={np.linalg.norm(r_center[:3]):.2f} kpc")

tend = t_back_units
tupd = tend/64.0              # update GC monopole & plot cadence
tau = tend / 256.0
MASS_FOR_DF = 0.0             # DF OFF for globular cluster scale

# Interpolator for host+moving-GC potential over each update interval
def advance_center_no_df(ic, dt, tstart, steps):
    return agama.orbit(ic=ic, potential=pot_host, time=dt, timestart=tstart, trajsize=steps)

# initialize arrays
r_mass   = [initmass]
r_traj   = [r_center.copy()]
r_xv     = xv_gc + r_center  # place sampled stars at the GC center 1 Gyr ago
time     = 0.0
times_t  = [time]
times_u  = [time]

fig = plt.figure(figsize=(9,6), dpi=110)
ax1=plt.axes([0.07, 0.08, 0.36, 0.54])
ax2=plt.axes([0.57, 0.08, 0.36, 0.54])
bx1=plt.axes([0.07, 0.72, 0.36, 0.24])
bx2=plt.axes([0.57, 0.72, 0.36, 0.24])
cx1=bx1.twinx()
cx2=bx2.twinx()
plot_movie = True          # ok to leave True; no windows will open
frame_idx = 0              # for saving frames

if plot_movie:
    plt.ion()

print("Using pyfalcon for full N-body comparison.")
f_center = r_center.copy()
f_xv     = r_xv.copy()
f_traj   = [f_center.copy()]
f_bound  = np.ones(len(f_xv), bool)
f_mass   = [initmass]
f_times  = [time]

while time < tend - 1e-12:
    steps_center = max(4, int(tupd/max(tau, tupd/64.0)))
    # 1) evolve GC COM in the host for one update interval (no DF)
    time_center, orbit_center = advance_center_no_df(ic=r_center, dt=tupd, tstart=time, steps=steps_center)
    times_u.append(time_center[-1])
    times_t.extend(time_center[1:])
    r_traj.extend(orbit_center[1:])
    r_center = orbit_center[-1]

    # 2) time-dependent potential host + moving GC monopole over this interval
    pot_total = agama.Potential(pot_host,
        agama.Potential(potential=pot_sat, center=np.column_stack((time_center, orbit_center))))

    # 3) evolve *test particles* in this combined potential over the interval
    r_xv = np.vstack(agama.orbit(ic=r_xv, potential=pot_total, time=tupd, timestart=time, trajsize=1)[:,1])

    # 4) update the GC potential from currently bound particles (spherical monopole)
    pot_sat = agama.Potential(type='multipole', particles=(r_xv[:,0:3] - r_center[:3], pmass), symmetry='s')

    # 5) bound selection in the satellite frame
    E_rel = pot_sat.potential(r_xv[:,0:3] - r_center[:3]) + 0.5*np.sum((r_xv[:,3:6] - r_center[3:6])**2, axis=1)
    r_bound = E_rel < 0.0
    r_mass.append(np.sum(pmass[r_bound]))

    # 6) full N-body comparison block
    # leapfrog for full N-body (self-gravity + host + DF on COM mass if desired)
    f_time = 0.0
    if time == 0.0:
        f_acc, f_pot = pyfalcon.gravity(f_xv[:,0:3], agama.G * pmass, GC_scale_kpc/2)
        f_acc += pot_host.force(f_xv[:,0:3]) + dynfricAccel(f_center[:3], f_center[3:6], MASS_FOR_DF)
    step = 0
    while f_time < tupd - 1e-15:
        h = min(tau, tupd - f_time)
        f_xv[:,3:6] += f_acc * (h/2)
        f_xv[:,0:3] += f_xv[:,3:6] * h
        if step % 4 == 0:
            f_acc, f_pot = pyfalcon.gravity(f_xv[:,0:3], agama.G * pmass, GC_scale_kpc/2)
        f_acc += pot_host.force(f_xv[:,0:3])
        f_acc += dynfricAccel(f_center[:3], f_center[3:6], MASS_FOR_DF)
        f_xv[:,3:6] += f_acc * (h/2)
        # crude COM+bound update for plotting/comparison
        f_center[:3] += h * f_center[3:6]
        Rmax = 0.2  # kpc
        use = np.sum((f_xv[:,0:3] - f_center[:3])**2, axis=1) < Rmax**2
        prev = f_center.copy()
        for _ in range(6):
            if not np.any(use): break
            f_center = np.median(f_xv[use], axis=0)
            f_bound = f_pot + 0.5*np.sum((f_xv[:,3:6]-f_center[3:6])**2, axis=1) < 0
            use = f_bound & (np.sum((f_xv[:,0:3]-f_center[:3])**2, axis=1) < Rmax**2)
            if np.allclose(f_center, prev): break
            prev = f_center.copy()
        f_traj.append(f_center.copy())
        f_mass.append(np.sum(pmass[f_bound]))
        f_time += h
        f_times.append(time + f_time)
        step += 1

time += tupd

    

import numpy as np

# Convert trajectories to arrays for easy handling
r_traj = np.array(r_traj)
f_traj = np.array(f_traj)
times_t = np.array(times_t)
f_times = np.array(f_times)

# --- 1. Interpolate the full N-body trajectory to restricted time sampling
from scipy.interpolate import interp1d

f_interp = interp1d(f_times, f_traj.T, bounds_error=False, fill_value="extrapolate")
f_traj_interp = f_interp(times_t).T

# --- 2. Compute COM separation and velocity difference
pos_err = np.linalg.norm(r_traj[:, :3] - f_traj_interp[:, :3], axis=1)
vel_err = np.linalg.norm(r_traj[:, 3:] - f_traj_interp[:, 3:], axis=1)

mean_pos_err = np.mean(pos_err)
mean_vel_err = np.mean(vel_err)

print(f"Mean COM offset: {mean_pos_err:.3f} kpc")
print(f"Mean COM velocity offset: {mean_vel_err:.3f} km/s")

# --- 3. Bound mass comparison
r_mass = np.array(r_mass)
f_mass = np.array(f_mass[:len(r_mass)])  # adjust length if necessary
mass_frac_diff = np.abs(r_mass - f_mass) / r_mass[0]
print(f"Final bound mass difference: {mass_frac_diff[-1]*100:.2f}% of initial")

# --- 4. Optional: stream morphology comparison at final time
r_icrs = galcen_particles_to_icrs(r_xv, SOLAR)
f_icrs = galcen_particles_to_icrs(f_xv, SOLAR)
sep = r_icrs.separation(f_icrs).arcsec
print(f"Median particle separation on sky: {np.median(sep):.2f} arcsec")
