# Streamcutter

Streamcutter is a toolkit for detecting and modeling diffuse stellar streams from globular cluster debris.
It performs filtering using proper-motion box cuts, spatial cone cuts, and tidal-radius exclusion to isolate likely stream member candidates from Gaia and DESI datasets.
This project is ongoing.

## Overview

The workflow combines observational filtering, mock stream simulations, and comparisons across different Milky Way potential models.

Main capabilities:

* Filter and select stream candidates from Gaia/LS data using PM + CMD constraints

* Generate mock stellar streams using restricted N-body or AGAMA simulations

* Compare observed streams to simulated ones

* Evaluate potential footprints for DESI or other survey fields (currently only DESI and LS)

## How to Use
1. Select Stream Candidates

Run the candidate selection process for a specific globular cluster (GC), in example Palomar 5:

```
python select_stream_candidates.py --GC Pal_5
```

Use underscore naming for the GC (e.g., Pal_5, NGC_5466, NGC_5634).

This will apply dynamical simulation of GC tidal stripping under assumed potential.

2. Run for All Globular Clusters

To process all GCs in batch mode:

```
python plot_all_streams.py

```
This will generate simulated stream maps for each GC

3. Compare Potential Models 
To test how different Milky Way potentials affect stream morphology:

```
python simulate_compare_potentials.py
```

This compares the same stream evolved in different Galactic potentials
(e.g., MWPotential2022 vs. barred or triaxial models).

4. Retrieve DESI Tractor Bricks

To extract DESI Tractor footprints along a simulated stream:

```
python get_tractor_footprint.py \
  --sim-file simulated_streams/simulated_stream_Pal_12.fits \
  --min-stars 100 \
  --concat \
  --outdir output_dir \
  --target-radius 1.0 \
  --env-radius 2.0
```

Arguments:

```
--sim-file : path to the simulated stream FITS file

--min-stars : minimum number of simulated stars per brick

--concat : merge all tiles into a single output

--target-radius : target field radius (deg)

--env-radius : background environment radius (deg)
```

## Dependencies

Make sure you have the following installed:

* numpy, scipy, matplotlib, astropy
* pandas, h5py, fitsio
* agama (for dynamical modeling)
* astroquery (for Gaia/DESI queries)
* pyfalcon (optional, for full N-body validation)

## Credit
If you use Streamcutter in your work (which currently the program is also in preparation), please cite:

```
Farisi S. A. and Cooper A. P. (in prep.), Diffuse Globular Cluster Debris in the DESI Milky Way Survey.
```

## Example Workflow
### Generate simulated streams
```
python simulate_compare_potentials.py
```
### Extract stream candidates for Pal 5
```
python select_stream_candidates.py --GC Pal_5
```
### Build DESI footprints for Pal 5
```
python get_tractor_footprint.py --sim-file simulated_streams/simulated_stream_Pal_5.fits --concat --outdir ./tractor_outputs
```
## Author

For further inquiries please contact: 

```
Salman A. Farisi
Institute of Astronomy, National Tsing Hua University
📧 salman@gapp.nthu.edu.tw
```
