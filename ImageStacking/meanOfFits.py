import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def mean_fits(files):
  count = 0
  for file in files:
    newdata = fits.open(file)[0].data
    if count == 0:
      data = newdata
    else:
     data = data + newdata
    count = count + 1
 
  data = data/count
  
  return data



if __name__ == '__main__':
  
  data  = mean_fits(['image0.fits', 'image1.fits', 'image2.fits'])
  print(data[100, 100])

  plt.imshow(data.T, cmap=plt.cm.viridis)
  plt.colorbar()
  plt.show()
