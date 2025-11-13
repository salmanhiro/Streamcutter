import sys, agama, numpy, scipy.integrate, scipy.ndimage, scipy.special, matplotlib, matplotlib.pyplot as plt

def build_potential():
    agama.setUnits(length=1, velocity=1, mass=1)  # work in units of 1 kpc, 1 km/s, 1 Msun)
    Trewind = -10.0  # initial time [Gyr] - the LMC orbit is computed back to that time
    Tcurr   =  0.0  # current time

    # heliocentric ICRS celestial coordinates and velocity of the LMC
    # (PM from Luri+ 2021, distance from Pietrzynski+ 2019, center and velocity from van der Marel+ 2002)
    ra, dec, dist, pmra, pmdec, vlos = 81.28, -69.78, 49.6, 1.858, 0.385, 262.2

    # transform to Galactocentric cartesian position/velocity, using built-in routines from Agama
    # (hence the manual conversion factors from degrees to radians and from mas/yr to km/s/kpc)
    l, b, pml, pmb = agama.transformCelestialCoords(agama.fromICRStoGalactic,
        ra * numpy.pi/180, dec * numpy.pi/180, pmra, pmdec)
    posvelLMC = agama.getGalactocentricFromGalactic(l, b, dist, pml*4.74, pmb*4.74, vlos)

    # Create a simple but realistic model of the Milky Way with a bulge, a single disk, and a spherical dark halo
    paramBulge = dict(
        type              = 'Spheroid',
        mass              = 1.2e10,
        scaleRadius       = 0.2,
        outerCutoffRadius = 1.8,
        gamma             = 0.0,
        beta              = 1.8)
    paramDisk  = dict(
        type='Disk',
        mass              = 5.0e10,
        scaleRadius       = 3.0,
        scaleHeight       = -0.4)
    paramHalo  = dict(
        type              = 'Spheroid',
        densityNorm       = 1.35e7,
        scaleRadius       = 14,
        outerCutoffRadius = 300,
        cutoffStrength    = 4,
        gamma             = 1,
        beta              = 3)
    densMWhalo = agama.Density(paramHalo)
    potMW      = agama.Potential(paramBulge, paramDisk, paramHalo)

    # create a sphericalized MW potential and a corresponding isotropic halo distribution function
    potMWsph   = agama.Potential(type='Multipole', potential=potMW, lmax=0, rmin=0.01, rmax=1000)
    gmHalo     = agama.GalaxyModel(potMWsph, agama.DistributionFunction(type='quasispherical', density=densMWhalo, potential=potMWsph))

    # compute the velocity dispersion in the MW halo needed for the dynamical friction
    rgrid      = numpy.logspace(1, 3, 16)
    xyzgrid    = numpy.column_stack([rgrid, rgrid*0, rgrid*0])
    sigmafnc   = agama.Spline(rgrid, gmHalo.moments(xyzgrid, dens=False, vel=False, vel2=True)[:,0]**0.5)

    # Create the LMC potential - a spherical truncated NFW profile with mass and radius
    # related by the equation below, which produces approximately the same enclosed mass
    # profile in the inner region, satisfying the observational constraints, as shown
    # in Fig.3 of Vasiliev,Belokurov&Erkal 2021.
    massLMC    = 1.5e11
    radiusLMC  = (massLMC/1e11)**0.6 * 8.5
    bminCouLog = radiusLMC * 2.0   # minimum impact parameter in the Coulomb logarithm
    potLMC     = agama.Potential(
        type              = 'spheroid',
        mass              = massLMC,
        scaleradius       = radiusLMC,
        outercutoffradius = radiusLMC*10,
        gamma             = 1,
        beta              = 3)

    def difeq(vars, t):
        x0    = vars[0:3]          # MW position
        v0    = vars[3:6]          # MW velocity
        x1    = vars[6:9]          # LMC position
        v1    = vars[9:12]         # LMC velocity
        dx    = x1-x0              # relative offset
        dv    = v1-v0              # relative velocity
        dist  = sum(dx**2)**0.5    # distance between the galaxies
        vmag  = sum(dv**2)**0.5    # magnitude of relative velocity
        f0    = potLMC.force(-dx)  # force from LMC acting on the MW center
        f1    = potMW .force( dx)  # force from MW acting on the LMC
        rho   = potMW.density(dx)  # actual MW density at this point
        sigma = sigmafnc(dist)     # approximate MW velocity dispersion at this point
        # distance-dependent Coulomb logarithm
        # (an approximation that best matches the results of N-body simulations)
        couLog= max(0, numpy.log(dist / bminCouLog)**0.5)
        X     = vmag / (sigma * 2**.5)
        drag  = -(4*numpy.pi * rho * dv / vmag *
            (scipy.special.erf(X) - 2/numpy.pi**.5 * X * numpy.exp(-X*X)) *
            massLMC * agama.G**2 / vmag**2 * couLog)   # dynamical friction force
        return numpy.hstack((v0, f0, v1, f1 + drag))

    Tstep   = 1./64
    tgrid   = numpy.linspace(Trewind, Tcurr, round((Tcurr-Trewind)/Tstep)+1)
    ic      = numpy.hstack((numpy.zeros(6), posvelLMC))
    sol     = scipy.integrate.odeint(difeq, ic, tgrid[::-1])[::-1]

    print("Computing the past orbits of the Milky Way and the LMC")

    Tstep   = 1./64
    tgrid   = numpy.linspace(Trewind, Tcurr, round((Tcurr-Trewind)/Tstep)+1)
    ic      = numpy.hstack((numpy.zeros(6), posvelLMC))
    sol     = scipy.integrate.odeint(difeq, ic, tgrid[::-1])[::-1]

    # LMC trajectory in the MW-centric (non-inertial) reference frame
    # (7 columns: time, 3 position and 3 velocity components)
    trajLMC = numpy.column_stack([tgrid, sol[:,6:12] - sol[:,0:6]])
    # MW trajectory in the inertial frame
    trajMWx = agama.Spline(tgrid, sol[:,0], der=sol[:,3])
    trajMWy = agama.Spline(tgrid, sol[:,1], der=sol[:,4])
    trajMWz = agama.Spline(tgrid, sol[:,2], der=sol[:,5])
    # MW centre acceleration is minus the second derivative of its trajectory in the inertial frame
    accMW   = numpy.column_stack([tgrid, -trajMWx(tgrid, 2), -trajMWy(tgrid, 2), -trajMWz(tgrid, 2)])
    potacc  = agama.Potential(type='UniformAcceleration', file=accMW)
    potLMCm = agama.Potential(potential=potLMC, center=trajLMC)  # potential of the moving LMC

    # finally, the total time-dependent potential in the non-inertial MW-centric reference frame
    potTotal= agama.Potential(potMW, potLMCm, potacc)
    return potTotal