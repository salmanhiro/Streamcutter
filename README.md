# Globular Cluster Tidal Disruption

This repository compares observed GC tidal stream with full and restricted N-body simulation to simulate the tidal disruption of GCs.

## Installation

Some dependencies (`pyfalcon` and `agama`) must be installed separately before installing this package.

**1. Install pyfalcon** (requires a Fortran compiler and FFTW):

```bash
pip install --no-build-isolation git+https://github.com/GalacticDynamics-Oxford/pyfalcon.git
```

**2. Install agama** — follow the [agama installation guide](https://github.com/GalacticDynamics-Oxford/Agama).

**3. Install the remaining dependencies and this package:**

```bash
pip install -e .
```

**4. Verify the installation:**

```bash
python scripts/basic.py
# streamcutter version: 0.1.0
# Basic check passed.
```

## Publishing to PyPI

See [PUBLISHING.md](PUBLISHING.md) for step-by-step instructions on how to register the package on PyPI with trusted publishing and cut a release.

## Reference

Vasiliev, E. (2019). AGAMA: action-based galaxy modelling architecture. Monthly Notices of the Royal Astronomical Society, 482(2), 1525-1544.

Chen, Y., Li, H., & Gnedin, O. Y. (2025). Stellar streams reveal the mass loss of globular clusters (arXiv:2411.19899v2). arXiv. https://doi.org/10.48550/arXiv.2411.19899

Palau, C. G., Wang, W., & Han, J. (2025). Modelling the M68 stellar stream with realistic mass loss and frequency distributions in angle-action coordinates (arXiv:2508.21408). arXiv. https://doi.org/10.48550/arXiv.2508.21408

Fardal, M. A., Huang, S., & Weinberg, M. D. (2014). Generation of mock tidal streams (arXiv:1410.1861). arXiv. https://doi.org/10.48550/arXiv.1410.1861
