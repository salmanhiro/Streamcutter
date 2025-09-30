
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric, CartesianRepresentation, CartesianDifferential


def get_observed_coords(
    xv,
    assume='galactocentric_vxyz',
    R0_kpc=8.5, # Schonrich et al 2010
    Zsun_kpc=0.1,
):
    """
    Convert Galactocentric phase-space coords -> heliocentric observables.

    Parameters
    ----------
    xv : (N,6) array
        Columns: [X, Y, Z, V1, V2, V3].
        If assume='galactocentric_vxyz', V1,V2,V3 are v_x,v_y,v_z in the **Astropy Galactocentric frame**.
        If assume='heliocentric_UVW',  V1,V2,V3 are classic (U,V,W) with U toward GC, V with rotation, W to NGP.
    assume : str
        'galactocentric_vxyz' (default) or 'heliocentric_UVW'.
    R0_kpc, Zsun_kpc : floats
        Solar position parameters for the Galactocentric frame.

    Returns
    -------
    ra, dec : deg
    vlos    : km/s
    pmra, pmdec : mas/yr  (pmRA* cos(dec) convention)
    dist    : kpc
    """

    # positions
    x = xv[:, 0] * u.kpc
    y = xv[:, 1] * u.kpc
    z = xv[:, 2] * u.kpc

    # velocities
    V1 = xv[:, 3] * (u.km/u.s)
    V2 = xv[:, 4] * (u.km/u.s)
    V3 = xv[:, 5] * (u.km/u.s)

    if assume.lower() == 'heliocentric_uvw':
        # Convert classic UVW -> Galactocentric (v_x,v_y,v_z)
        # Conventions: U (+) toward GC, V (+) in rotation, W (+) to NGP
        # Astropy Galactocentric has +x from GC toward Sun, +y in rotation, +z to NGP.
        vx = -V1    # -U
        vy =  V2    #  V
        vz =  V3    #  W
    else:
        vx, vy, vz = V1, V2, V3

    # Build a Galactocentric frame with explicit Sun position (keeps you honest)
    gc_frame = Galactocentric(
        galcen_distance=R0_kpc * u.kpc,
        z_sun=Zsun_kpc * u.kpc,
    )

    # Use explicit Cartesian rep/diff (no transposes; pass named components)
    rep = CartesianRepresentation(x=x, y=y, z=z)
    dif = CartesianDifferential(d_x=vx, d_y=vy, d_z=vz)
    rep = rep.with_differentials(dif)

    gc = SkyCoord(rep, frame=gc_frame)
    icrs = gc.icrs

    # Observables
    ra   = icrs.ra.to(u.deg).value
    dec  = icrs.dec.to(u.deg).value
    dist = icrs.distance.to(u.kpc).value
    vlos = icrs.radial_velocity.to(u.km/u.s).value
    pmra = icrs.pm_ra_cosdec.to(u.mas/u.yr).value
    pmde = icrs.pm_dec.to(u.mas/u.yr).value

    # If many points have vtan_pm << vtan_in, suspect frame/velocity mapping mistakes.
    # You might print or log np.median(vtan_pm / (vtan_in+1e-8)) to ensure ~O(1).

    return ra, dec, vlos, pmra, pmde, dist
