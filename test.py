# pylint: disable=no-member
from glob import glob

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

from rapp.api import find_star, match

catalog_path = '.local/catalog.fits'
fits_path = '.local/CSST_sim_0.fits'
bias_path = '.local/bias_0.fits'
flat_path = '.local/flat_0.fits'

hdu = fits.open(fits_path)
bis = fits.open(bias_path)
flt = fits.open(flat_path)
result = find_star(hdu, bis, flt, 5)
hdu.close()
bis.close()
flt.close()


with fits.open(catalog_path) as hdu:
    result = match(result, hdu)
res = result[result['mag_m'] < 16]
print(res['mag'])
plt.plot(res['mag'])
plt.plot(res['mag_m'])
plt.show()
plt.close()
