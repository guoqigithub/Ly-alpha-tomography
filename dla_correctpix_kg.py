import numpy as np

def tau_lorentz(deltawave, n_hi):
    """
    Return optical depth at some rest wavelength separation DELTAWAVE
    (in angstroms) for a column density of N_HI
    
    Parameters:
    -----------
    deltawave : float or numpy.ndarray
        Wavelength separation from Lyman-alpha rest wavelength
    n_hi : float
        Neutral hydrogen column density
    
    Returns:
    --------
    float or numpy.ndarray
        Optical depth values
    """
    # Physical constants
    e = 4.80320e-10  # electron charge
    m_e = 9.109389e-28  # electron mass
    c = 2.99792e10  # speed of light
    
    # Lyman-alpha line parameters
    gamma_lambda = 7616.0
    lambda_cm = 1.21567e-5 
    f_lu = 0.4164
    
    # Ensure n_hi is a scalar
    if np.isscalar(n_hi):
        n_hi = np.full_like(deltawave, n_hi)
    
    # Compute optical depth
    emc_factor = (e**2) / (m_e * c**3)
    tau = emc_factor / (4 * np.pi) * f_lu * gamma_lambda * lambda_cm * \
          n_hi * (1215.67)**2 / deltawave**2
    
    return tau.astype(float)

def dla_correctpix_kg(wave_in, n_hi, z_abs=None, log_column=False):
    """
    For an input wavelength vector, return a vector of exp(tau) 
    to allow corrections for the damping wings of a DLA.
    
    Parameters:
    -----------
    wave_in : numpy.ndarray
        Wavelength vector, either in observed frame or DLA restframe
    n_hi : float or numpy.ndarray
        Neutral hydrogen column density
    z_abs : float, optional
        Redshift of DLA (used if wavelength is in observed frame)
    log_column : bool, optional
        Set if input N_HI is in log10
    
    Returns:
    --------
    numpy.ndarray
        Vector of exp(tau) corresponding to input wavelength vector
    """
    # Convert to rest frame wavelength if redshift is provided
    if z_abs is not None:
        wave = wave_in / (1 + z_abs)
    else:
        wave = wave_in
    
    # Convert column density if in log10
    if log_column:
        nhi = 10**n_hi
    else:
        nhi = n_hi
    
    # Compute wavelength difference from Lyman-alpha
    dwave = wave - 1215.67
    
    # Compute and return exp(tau)
    exptau = np.exp(tau_lorentz(dwave, nhi))
    
    return exptau
