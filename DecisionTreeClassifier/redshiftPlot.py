import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    data = np.load('sdss_galaxy_colors.npy')
    
    # get colour map
    cmap1 = plt.get_cmap('YlOrRd')

    dx_ug = data['u'] - data['g']
    dx_ri = data['r'] - data['i']
    
    redshift = data['redshift']
   
    plt.scatter(dx_ug, dx_ri, c=redshift, lw=0, s=1, cmap=cmap1)
    colorbar = plt.colorbar().set_label("Redshift")
    
    plt.xlabel= "Colour index u-g"
    plt.ylabel= "Colour index r-i"
    plt.title = "Redshift (colour) u-g versus r-i"
    
    plt.axis([-0.5, 2.5, -0.5, 1])
    
    plt.show()
