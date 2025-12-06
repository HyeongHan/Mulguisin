import numpy as np
from astropy.cosmology import FlatLambdaCDM

default_cosmo = FlatLambdaCDM(
    H0=67.66,      
    Om0=0.3111,    
    Tcmb0=2.7255,  
    Ob0=0.049      
)


def tan_project(ra, dec, ra0, dec0):
    ra  = np.radians(ra)
    dec = np.radians(dec)
    ra0 = np.radians(ra0)
    dec0 = np.radians(dec0)

    cosc = np.sin(dec0)*np.sin(dec) + np.cos(dec0)*np.cos(dec)*np.cos(ra - ra0)
    
    x =  np.cos(dec) * np.sin(ra - ra0) / cosc      # radians
    y = (np.cos(dec0)*np.sin(dec) - np.sin(dec0)*np.cos(dec)*np.cos(ra - ra0)) / cosc

    return x, y

def sky_to_comoving_xy(ra, dec, ra0, dec0, z, cosmology=cosmo):
    """
    RA, Dec, RA0, Dec0 in degrees, z is the redshift at which you want comoving coords.
    Returns X, Y in comoving Mpc relative to (ra0, dec0).
    """
    # Step 1: angular TAN projection (radians)
    x_rad, y_rad = tan_project(ra, dec, ra0, dec0)

    # Step 2: transverse comoving distance at redshift z (Mpc)
    D_M = cosmology.comoving_transverse_distance(z).value  # Mpc

    # Step 3: convert to comoving Mpc
    X = D_M * x_rad
    Y = D_M * y_rad

    return X, Y
