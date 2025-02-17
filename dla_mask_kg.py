import numpy as np

def ew_dla(n_hi):
    """
    Calculate equivalent width of damped hydrogen Lya absorber.
    
    Uses formula from Eq 9.22 of Drain's 'Physics of the ISM'
    
    Parameters:
    -----------
    n_hi : float
        Neutral hydrogen column density
    
    Returns:
    --------
    float
        Equivalent width of the DLA
    """
    # Physical constants
    e = 4.80320e-10  # electron charge
    m_e = 9.109389e-28  # electron mass
    c = 2.99792e10  # speed of light
    
    # Lyman-alpha transition parameters
    gamma_lambda = 7616.0
    lambda_cm = 1.21567e-5 
    f_lu = 0.4164
    
    # Calculate equivalent width
    ew = np.sqrt((e**2 / (m_e * c**2)) * n_hi * f_lu * lambda_cm * 
                 gamma_lambda / c)
    
    return ew * 1215.67

def dla_mask_kg(wave, n_hi, z_abs, log_column=False):
    """
    Create a mask for DLA region based on equivalent width.
    
    Parameters:
    -----------
    wave : array-like
        Wavelength vector in observed frame
    n_hi : float
        Neutral hydrogen column density of DLA
    z_abs : float
        Absorption redshift of DLA
    log_column : bool, optional
        If True, n_hi is interpreted as log10(N_HI)
    
    Returns:
    --------
    numpy.ndarray
        Mask where 1 is unmasked, 0 is masked
    float
        Equivalent width of the DLA
    """
    # Convert column density if needed
    nhi_tmp = 10**n_hi if log_column else n_hi
    
    # Translate observed wavelength scale to DLA restframe
    wave_dla = wave / (1 + z_abs)
    
    # Calculate equivalent width
    ew = ew_dla(nhi_tmp)
    
    # Create mask 
    maskflag = np.ones_like(wave, dtype=float)
    
    # Mask region around Lyman-alpha
    masklist = np.where((wave_dla >= (1215.67 - ew/2)) & 
                        (wave_dla < (1215.67 + ew/2)))
    
    maskflag[masklist] = 0.0
    
    return maskflag, ew
