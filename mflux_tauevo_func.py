import numpy as np
from .taueff_evo import taueff_evo

def mflux_tauevo_func(x, *p, zqso=None):
    """
    Mean flux evolution function for Lyman-alpha forest continuum correction
    
    Parameters:
    -----------
    x : array-like
        Rest-frame wavelength
    p : float
        Free parameter delta for correction
    zqso : float
        Quasar redshift
    
    Returns:
    --------
    Array of mean flux tau evolution corrections
    
    Notes:
    ------
    Computes mean flux evolution * exp(delta * (lambda/1280 - 1))
    Designed for use with curve_fit or similar optimization methods
    """
    # Validate inputs
    if zqso is None:
        raise ValueError("Quasar redshift (zqso) must be provided")
    
    # Calculate forest redshift
    z_for = (x / 1216.0) * (1.0 + zqso) - 1.0
    
    # Compute effective optical depth
    tau = taueff_evo(z_for)
    
    # Compute mean flux
    fmean = np.exp(-tau)
    
    # Apply mean flux correction 
    mfluxtauevo = fmean * mfluxcorr(x, p)
    
    return mfluxtauevo

def mfluxcorr(x, p):
    """
    Correction factor to power-law fit
    
    Parameters:
    -----------
    x : array-like
        Input wavelength values
    p : array-like
        Correction parameters
    
    Returns:
    --------
    Array of correction factors
    """
    lamb_piv = 1113.0  # Pivot point in the restframe spectrum
    return p[0] + p[1] * (x / lamb_piv - 1.)

