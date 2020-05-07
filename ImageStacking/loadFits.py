import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
  
def load_fits(file):
  data = fits.open(file)[0].data
  return np.unravel_index(np.argmax(data, axis=None), data.shape)


if __name__ == '__main__':
 
  bright = load_fits('image1.fits')
  print(bright)

  hdulist = fits.open('image1.fits')
  data = hdulist[0].data

  # Plot the 2D image data
  plt.imshow(data.T, cmap=plt.cm.viridis)
  plt.colorbar()
  plt.show()
