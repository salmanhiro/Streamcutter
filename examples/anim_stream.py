#!/usr/bin/env python3
# pal5_frames_timeseries.py — many JPG frames; GC orbits host; stream shown each frame

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import astropy.units as u
import astropy.coordinates as coord
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic

# shims
try:
    from gala.dynamics.mockstream import MockStreamGenerator
except Exception:
    from gala.dynamics import MockStreamGenerator
try:
    from gala.dynamics.streamdf import FardalStreamDF
except Exception:
    from gala.dynamics import FardalStreamDF

# ---------------- 1) Pal 5 state ----------------
pal5_c = coord.SkyCoord(ra=229.018*u.deg, dec=-0.124*u.deg,
                        distance=22.9*u.kpc,
                        pm_ra_cosdec=-2.296*u.mas/u.yr,
                        pm_dec=-2.257*u.mas/u.yr,
                        radial_velocity=-58.7*u.km/u.s)
rep = pal5_c.transform_to(coord.Galactocentric()).data
pal5_w0 = gd.PhaseSpacePosition(rep)

pal5_mass = 2.5e4 * u.Msun
pal5_pot  = gp.PlummerPotential(m=pal5_mass, b=4*u.pc, units=galactic)
mw        = gp.MilkyWayPotential()
df        = FardalStreamDF(gala_modified=True)

# ---------------- 2) Progenitor time series (this gives frames!) ----------------
# Small timestep, many steps, and save_all=True to keep the full track.
dt_prog   = -1 * u.Myr
n_steps_p = 1000
prog_orb  = mw.integrate_orbit(pal5_w0, dt_prog, n_steps_p, save_all=True)

# Convenience arrays
x_prog = prog_orb.pos.x.to_value(u.kpc)  # shape (n_time,)
y_prog = prog_orb.pos.y.to_value(u.kpc)
n_frames = x_prog.size

# ---------------- 3) One mock stream snapshot (static point cloud) ----------------
gen = MockStreamGenerator(df, mw, progenitor_potential=pal5_pot)
stream_snapshot, _ = gen.run(pal5_w0, pal5_mass, dt=-1*u.Myr, n_steps=4000, release_every=5)

# Stream positions now (constant each frame)
xs = stream_snapshot.pos.x.to_value(u.kpc).ravel()
ys = stream_snapshot.pos.y.to_value(u.kpc).ravel()
m  = np.isfinite(xs) & np.isfinite(ys)
xs, ys = xs[m], ys[m]

# ---------------- 4) Figure & static host context ----------------
OUTDIR = "frames_pal5_xy"
os.makedirs(OUTDIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(7.0, 7.0))
ax.set_aspect("equal")
ax.set_xlabel("X [kpc]")
ax.set_ylabel("Y [kpc]")

# Host: GC, disk ring, Sun
ax.scatter([0],[0], s=60, marker="*", color="gold", edgecolors="k", zorder=5, label="Galactic Center")
theta = np.linspace(0, 2*np.pi, 512)
Rdisk = 15.0
ax.plot(Rdisk*np.cos(theta), Rdisk*np.sin(theta), color="0.85", lw=1.2, zorder=0, label="MW disk (viz)")
ax.scatter([-8.2],[0], s=30, color="tab:orange", edgecolors="k", zorder=5, label="Sun")

# Full progenitor path
ax.plot(x_prog, y_prog, lw=1.2, color="0.8", zorder=1, label="Progenitor orbit")

# Dynamic artists
trail_len = 200
prog_pt,  = ax.plot([], [], "o", ms=6, color="tab:red", zorder=6, label="Progenitor (now)")
trail_ln, = ax.plot([], [], "-", lw=2.0, color="tab:red", alpha=0.4, zorder=2)
str_sc    = ax.scatter(xs, ys, s=2, color="tab:blue", alpha=0.9, zorder=3, label="Stream stars")

# Limits
xy   = np.vstack((x_prog, y_prog)).T
xmid, ymid = xy.mean(axis=0)
span = np.max(np.ptp(xy, axis=0))
R    = 0.7*span + 8.0
ax.set_xlim(xmid - R, xmid + R)
ax.set_ylim(ymid - R, ymid + R)
ax.legend(loc="upper right", framealpha=0.92)

# ---------------- 5) Save frames ----------------
stride = 1
dpi    = 120
for k in range(0, n_frames, stride):
    # moving progenitor + short trail
    prog_pt.set_data([x_prog[k]], [y_prog[k]])
    k0 = max(0, k - trail_len)
    trail_ln.set_data(x_prog[k0:k+1], y_prog[k0:k+1])

    ax.set_title(f"Pal 5 mock stream (lookback {k*abs(dt_prog.to_value(u.Myr)):.0f} Myr)")
    out = os.path.join(OUTDIR, f"frame_{k:05d}.jpg")
    plt.savefig(out, dpi=dpi, facecolor="white", bbox_inches="tight")

print(f"[done] Wrote {len(range(0, n_frames, stride))} JPG frames to {OUTDIR}/")
