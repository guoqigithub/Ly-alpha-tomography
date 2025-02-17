import numpy as np
from scipy.optimize import curve_fit
from .forestvar_func import forestvar_func
from .mfluxcorr import mfluxcorr
from .taueff_evo import taueff_evo
from .pca_chisq import pca_func
from .pca_chisq import pca_chisq

def mf_pca(ff_in, lambda_r_in, sigma_in=None, ivar_in=None, 
           delta_mf=None, pcaparams=None,afluxerr=None, dr7eigen=None, zqso=None, 
           wavemin=None,m_fit=None):
    """
    Mean-Flux Regulated PCA Fitting for Quasar Spectra
    
    Parameters:
    -----------
    ff_in : array
        Input quasar flux
    lambda_r_in : array
        Rest-frame wavelength
    sigma_in : array, optional
        Flux uncertainties
    ivar_in : array, optional
        Inverse variances
    delta_mf : array, optional
        Mean flux correction parameter
    pcaparams : array, optional
        Pre-computed PCA parameters
    dr7eigen : optional
        DR7 eigenspectra
    zqso : float, optional
        Quasar redshift
    wavemin : float, optional
        Minimum wavelength
    
    Returns:
    --------
    dict
        Dictionary containing PCA results
    """
    # Input handling
    ff = np.array(ff_in)
    lambda_r = np.array(lambda_r_in)
    
    # Inverse variance handling
    if ivar_in is not None:
        ivar = np.array(ivar_in)
    else:
        ivar = 1.0 / (np.array(sigma_in)**2 + (np.array(sigma_in) == 0))
    
    # PCA parameter fitting
    if pcaparams is None:
        fitpca,afluxerr = pca_chisq(ff, lambda_r, sigma_in, ivar,
                           dr7eigen=dr7eigen)
    else:
        fitpca = np.array(pcaparams)
    
    # Generate PCA continuum
    pcaflux = pca_func(lambda_r, fitpca, dr7eigen)
    
    # Forest wavelength range
    for_low, for_hi = 1041., 1185.
    
    # Normalization
    normpix1 = np.where((lambda_r >= 1275) & (lambda_r <= 1285))[0]
    if len(normpix1) > 0 and np.sum(ivar[normpix1]) != 0:
        normfac = np.average(ff[normpix1] * ivar[normpix1]) / np.average(ivar[normpix1])
    else:
        normpix2 = np.where((lambda_r >= 1450) & (lambda_r <= 1470))[0]
        normfac = np.average(ff[normpix2] * ivar[normpix2]) / np.average(ivar[normpix2])
    
    # Normalize flux and inverse variance
    ivar = ivar * normfac**2
    ff = ff / normfac
    pcaflux = pcaflux / normfac
    
    # Forest region extraction
    forestrange = np.where(
        (lambda_r >= for_low) & 
        (lambda_r <= for_hi) & 
        (lambda_r * (1 + zqso) > wavemin)      
    )[0]
    
    lamb_forest = lambda_r[forestrange]
    z_for = (lamb_forest / 1216.7) * (1 + zqso) - 1
    
    fforest = ff[forestrange] / pcaflux[forestrange]
    ivarforest = ivar[forestrange] * pcaflux[forestrange]**2
    
    # Weight estimation
    var_F = forestvar_func(z_for) * (np.exp(-taueff_evo(z_for)))**2
    var_noise = np.where(ivarforest != 0, 1 / ivarforest, 0)
    var_total = var_F + var_noise
    weights_forest = np.where(var_total != 0, 1 / var_total, 0)
    
    # Mask out problematic pixels
    maskedpix = np.where(ivarforest == 0)[0]
    if len(maskedpix) > 0:
        weights_forest[maskedpix] = 0
    
    # Fit mean flux correction
    delta_mf_guess = [0., 0.]

    def mflux_tauevo_func(x, *p, zqso=zqso):
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

    
    delta_mf, _ = curve_fit(
        mflux_tauevo_func, 
        lamb_forest, 
        fforest, 
        p0=delta_mf_guess, 
        sigma=1/np.sqrt(weights_forest)
    )
    
    return ivar, delta_mf, fitpca, afluxerr