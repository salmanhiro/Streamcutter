from astropy.table import Table
import numpy as np
from utils import feh_correct
from astropy.io import fits

print("Reading RV table and applying corrections...")
# Koposov https://academic.oup.com/mnras/article/533/1/1012/7724389
# Open the FITS file
RV_OFFSET_TAB =  Table().read('data/backup_correction.fits')

# Note these are pretty big tables
RV_T = Table().read('data/mwsall-pix-iron.fits',
                        'RVTAB',
                        mask_invalid=False)

FM_T = Table().read('data/mwsall-pix-iron.fits',
                        'FIBERMAP',
                        mask_invalid=False)

SP_T = Table().read('data/mwsall-pix-iron.fits',
                        'SPTAB',
                        mask_invalid=False)

GAIA_T = Table().read('data/mwsall-pix-iron.fits',
                        'GAIA',
                        mask_invalid=False)

RV_T['FEH_CORRECTED'] = feh_correct.calibrate(RV_T['FEH'], RV_T['TEFF'], RV_T['LOGG'])

# For backup program, need to correct RV since there is RV offset. Otherwise, use RV value
RV_OFFSET_TAB =  Table().read('data/backup_correction.fits')

# Keep original RV_T intact
VRAD_BIAS = np.full(len(RV_T), np.nan)

# Build map from correction table
bias_map = dict(zip(RV_OFFSET_TAB["TARGETID"], RV_OFFSET_TAB["VRAD_BIAS"]))

# Apply correction by mapping TARGETID
for i, tid in enumerate(RV_T["TARGETID"]):
    if tid in bias_map:
        VRAD_BIAS[i] = bias_map[tid]

RV_T["VRAD_BIAS"] = VRAD_BIAS
print(f"VRAD_BIAS examples: {VRAD_BIAS[~np.isnan(VRAD_BIAS)][:5]}")
RV_T["VRAD_CORRECTED"] = np.where(
    np.isnan(VRAD_BIAS),
    RV_T["VRAD"],
    RV_T["VRAD"] + VRAD_BIAS
)


print(f"Total rows: {len(RV_T):,}")
print(f"Rows corrected (with VRAD_BIAS): {(~np.isnan(RV_T['VRAD_BIAS'])).sum():,} ({(~np.isnan(RV_T['VRAD_BIAS'])).mean() * 100:.2f}%)")
print(f"Rows uncorrected (missing VRAD_BIAS): {(np.isnan(RV_T['VRAD_BIAS'])).sum():,} ({(np.isnan(RV_T['VRAD_BIAS'])).mean() * 100:.2f}%)")
# make sure RV_T["VRAD_CORRECTED"] is all completed (either from using VRAD value or VRAD + VRAD_BIAS)
assert np.all(~np.isnan(RV_T["VRAD_CORRECTED"]))
print(f"VRAD_CORRECTED stats: mean={np.nanmean(RV_T['VRAD_CORRECTED']):.2f}, std={np.nanstd(RV_T['VRAD_CORRECTED']):.2f}, min={np.nanmin(RV_T['VRAD_CORRECTED']):.2f}, max={np.nanmax(RV_T['VRAD_CORRECTED']):.2f}")

corrected_fits_path = 'data/mwsall-pix-iron-rv-corrected.fits'

h0 = fits.PrimaryHDU()
h1 = fits.table_to_hdu(RV_T); h1.name = 'RVTAB'      # corrected RV table
h2 = fits.table_to_hdu(SP_T); h2.name = 'SPTAB'   # COPY of SP_T, unchanged
h3 = fits.table_to_hdu(FM_T); h3.name = 'FIBERMAP'   # COPY of FM_T, unchanged
h5 = fits.table_to_hdu(GAIA_T); h5.name = 'GAIA'      # COPY of GAIA_T, unchanged

fits.HDUList([h0, h1, h2, h3, h5]).writeto(corrected_fits_path, overwrite=True)
print(f"[v] wrote multi-HDU file with RVTAB+FIBERMAP to {corrected_fits_path}")

# Re-read to verify
RV_T = Table.read(corrected_fits_path, 'RVTAB',    mask_invalid=False)
SP_T = Table.read(corrected_fits_path, 'SPTAB',    mask_invalid=False)
FM_T = Table.read(corrected_fits_path, 'FIBERMAP', mask_invalid=False)
GAIA_T = Table.read(corrected_fits_path, 'GAIA',    mask_invalid=False)

# print first 5 data of FM_T and RV_T and check their TARGETID is the same
print("First 5 rows of FM_T TARGETID:", FM_T['TARGETID'][:5])
print("First 5 rows of SP_T TARGETID:", SP_T['TARGETID'][:5])
print("First 5 rows of RV_T TARGETID:", RV_T['TARGETID'][:5])
print("Last 5 rows of FM_T TARGETID:", FM_T['TARGETID'][-5:])
print("Last 5 rows of SP_T TARGETID:", SP_T['TARGETID'][-5:])
print("Last 5 rows of RV_T TARGETID:", RV_T['TARGETID'][-5:])
print("Checking TARGETID match between Tables...")
assert np.all(FM_T['TARGETID'] == RV_T['TARGETID'] == SP_T['TARGETID'])
assert len(FM_T) == len(RV_T) == len(SP_T)

print("TARGETID match confirmed.")

print("RV and [Fe/H] corrections applied and saved.")