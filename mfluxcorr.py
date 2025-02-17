import numpy as np

def mfluxcorr(x, p):
    """
    Correction factor to power-law fit
    
    Parameters:
    -----------
    x : array-like
        Input wavelength values
    p : array-like
        Correction parameters
        p[0]: constant offset
        p[1]: slope parameter
    
    Returns:
    --------
    Array of correction factors
    
    Notes:
    ------
    Pivot wavelength is set to 1113 Angstroms in the rest frame spectrum
    """
    lamb_piv = 1113.0  # Pivot point in the restframe spectrum
    return p[0] + p[1] * (x / lamb_piv - 1.)

