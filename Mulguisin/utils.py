import numpy as np
from astropy.cosmology import FlatLambdaCDM

default_cosmo = FlatLambdaCDM(
    H0=67.66,      
    Om0=0.3111,    
    Tcmb0=2.7255,  
    Ob0=0.049      
)

def radec_to_xyz(ra, dec, z, cosmo=default_cosmo):
    """
    ra: degrees
    dec:  degrees
    z: redshift 
    """
    ra  = np.radians(ra)
    dec = np.radians(dec)

    r = cosmo.comoving_distance(z).value  # Mpc

    X = r * np.cos(dec) * np.cos(ra)
    Y = r * np.cos(dec) * np.sin(ra)
    Z = r * np.sin(dec)

    return X, Y, Z

def relative_xyz(ra, dec, z, ra0, dec0, z0, cosmo=default_cosmo):

    X, Y, Z = radec_to_xyz(ra, dec, z, cosmo)
    
    # reference coordinate
    X0, Y0, Z0 = radec_to_xyz(ra0, dec0, z0, cosmo)

    return X - X0, Y - Y0, Z - Z0

