
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric, CartesianRepresentation, CartesianDifferential

def get_observed_coords(
    xv,
    assume='galactocentric_vxyz',
    R0_kpc=8.5, # Bland-Hawthorn & Gerhard 2016
    Zsun_pc=25, # Bland-Hawthorn & Gerhard 2016
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
    vx = xv[:, 3] * (u.km/u.s)
    vy = xv[:, 4] * (u.km/u.s)
    vz = xv[:, 5] * (u.km/u.s)

    # Build a Galactocentric frame with explicit Sun position (keeps you honest)
    gc_frame = Galactocentric(
        galcen_distance=R0_kpc * u.kpc,
        z_sun=Zsun_pc * u.pc,
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

def get_galactocentric_coords(
    ra_deg,
    dec_deg,
    distance_kpc,
    vlos_kms,
    pmra_masyr,
    pmdec_masyr,
    R0_kpc=8.5,  # Bland-Hawthorn & Gerhard 2016
    Zsun_pc=25  # Bland-Hawthorn & Gerhard 2016
):
    """
    Convert heliocentric observables -> Galactocentric Cartesian (x, y, z, vx, vy, vz).

    Parameters
    ----------
    ra_deg, dec_deg : float or array
        Right Ascension and Declination in degrees.
    distance_kpc : float or array
        Distance in kpc.
    vlos_kms : float or array
        Radial velocity in km/s.
    pmra_masyr, pmdec_masyr : float or array
        Proper motions in mas/yr.
    R0_kpc : float
        Distance from Sun to Galactic Center (default: 8.5 kpc).
    Zsun_pc : float
        Height of Sun above the Galactic plane (default: 25 pc).

    Returns
    -------
    x, y, z : kpc
    vx, vy, vz : km/s
        Galactocentric Cartesian position and velocity.
    """

    # Construct the ICRS (RA/Dec) frame with velocities
    icrs = SkyCoord(
        ra=ra_deg * u.deg,
        dec=dec_deg * u.deg,
        distance=distance_kpc * u.kpc,
        pm_ra_cosdec=pmra_masyr * u.mas/u.yr,
        pm_dec=pmdec_masyr * u.mas/u.yr,
        radial_velocity=vlos_kms * u.km/u.s
    )

    # Define the Galactocentric frame with solar parameters
    gc_frame = Galactocentric(
        galcen_distance=R0_kpc * u.kpc,
        z_sun=Zsun_pc * u.pc,
    )

    # Transform to Galactocentric frame
    gc = icrs.transform_to(gc_frame)

    # Extract Cartesian position and velocity
    x = gc.cartesian.x.to(u.kpc).value
    y = gc.cartesian.y.to(u.kpc).value
    z = gc.cartesian.z.to(u.kpc).value

    vx = gc.velocity.d_x.to(u.km/u.s).value
    vy = gc.velocity.d_y.to(u.km/u.s).value
    vz = gc.velocity.d_z.to(u.km/u.s).value

    return np.column_stack([x, y, z, vx, vy, vz])
