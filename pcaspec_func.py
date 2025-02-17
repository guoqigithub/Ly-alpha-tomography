import numpy as np
from .dla_mask_kg import dla_mask_kg
from .dla_correctpix_kg import dla_correctpix_kg
from .mfluxcorr import mfluxcorr
from .pca_chisq import pca_func
from .pca_chisq import pca_chisq
from .mf_pca import mf_pca

def pcaspec_func(lambda_r_in, ff_in, sigma_in=None, ivar_in=None, 
                 pca_only=False, pcacont=None, dr7eigen=False, 
                 m_fit=8, zqso1=None, mfflag1=2, wavemin1=3600.0, 
                 delta_mf=None, pcaparams=None, 
                 afluxerr=None, dla_in=None):
    """
    Wrapper function for PCA spectral continuum fitting with optional 
    mean-flux regulation and DLA correction.
    
    Parameters:
    -----------
    lambda_r_in : array-like
        Wavelength grid in quasar rest frame
    ff_in : array-like
        Quasar spectrum flux
    sigma_in : array-like, optional
        Flux errors (converted to inverse variance internally)
    ivar_in : array-like, optional
        Inverse variance of spectrum
    pca_only : bool, optional
        Do continuum fitting using only PCA
    pcacont : array-like, optional
        Pure PCA continuum
    dr7eigen : bool, optional
        Use Paris et al 2011 eigenspectra templates
    m_fit : int, optional
        Number of principal components to use (default 8)
    zqso1 : float, optional
        Quasar redshift
    mfflag1 : int, optional
        Mean-flux measurement flag
    wavemin1 : float, optional
        Minimum observed wavelength
    delta_mf : array-like, optional
        Mean-flux regulation parameters
    pcaparams : array-like, optional
        PCA parameters
    afluxerr : array-like, optional
        Absolute flux error
    dla_in : array-like, optional
        DLA absorber parameters
    
    Returns:
    --------
    array
        Best-fit continuum spectrum
    """
    # Input validation
    if zqso1 is None:
        raise ValueError("ZQSO needs to be set")
    
    # Set default parameters
    zqso = zqso1
    mfflag = mfflag1 if mfflag1 is not None else 2
    wavemin = wavemin1 if wavemin1 is not None else 3600.0
    
    # Check input errors/inverse variance
    if sigma_in is None and ivar_in is None:
        raise ValueError("Either flux errors or inverse variances must be input")
    
    # Convert sigma to inverse variance if needed
    if ivar_in is None:
        ivar = np.where(sigma_in != 0, 1 / (sigma_in**2), 0)
    else:
        ivar = ivar_in
    
    # Create working copy of flux and wavelength
    ff = ff_in.copy()
    lambda_r = lambda_r_in.copy()
    
    # DLA handling
    ff_dla = ff.copy()
    if dla_in is not None:
        # Assuming dla_in is a 2D array with [redshift, log10(NHI)]
        n_dla = dla_in.shape[1] if dla_in.ndim > 1 else 1
        
        # Masked pixels
        masked_pix = np.where(ivar == 0)[0]
        ff_dla[masked_pix] = 0.
        
        for rr in range(n_dla):
            zabs_tmp = dla_in[0, rr] if n_dla > 1 else dla_in[0]
            nhi = 10 ** (dla_in[1, rr] if n_dla > 1 else dla_in[1])
            
            # DLA masking
            dlamask = dla_mask_kg(lambda_r * (1 + zqso), nhi, zabs_tmp)     # check return in dla_mask_kg
            cut_pix = np.where(dlamask < 1e-6)[0]
            dla_notmask = np.setdiff1d(np.arange(len(lambda_r)), cut_pix)
            
            ff_dla[cut_pix] = 0.
            ivar[cut_pix] = 0.
            
            # DLA correction
            exptau = dla_correctpix_kg(lambda_r[dla_notmask] * (1 + zqso), nhi, zabs_tmp)
            ff[dla_notmask] *= exptau
            ff_dla[dla_notmask] *= exptau
            ivar[dla_notmask] /= exptau**2
    
    # Handle wavelength range
    wave_excess = np.where((lambda_r < 1030) | (lambda_r > 1600))[0]
    if len(wave_excess) > 0:
        lambda_r_full = lambda_r_in.copy()
        pcaspec_full = np.zeros_like(ff_in)
        wave_cut = np.setdiff1d(np.arange(len(lambda_r)), wave_excess)
        
        lambda_r = lambda_r[wave_cut]
        ff = ff[wave_cut]
        ivar = ivar[wave_cut]
    
    # PCA Continuum Fitting
    if not pca_only:
        # Mean-flux regulated PCA
        ivar, delta_mf, pcaparams, afluxerr = mf_pca(ff, lambda_r, sigma_in, ivar_in=ivar, 
                            delta_mf=delta_mf, 
                            pcaparams=pcaparams, 
                            afluxerr=afluxerr, 
                            dr7eigen=dr7eigen,
                            zqso=zqso, 
                            wavemin=wavemin,
                            m_fit=m_fit)      # check return
        
        cz = pcaparams[0]
        pcaspec = pca_func(lambda_r, pcaparams, dr7eigen=dr7eigen)
        pcacont = pcaspec.copy()
        
        # Apply mean-flux correction to wavelengths <= 1185
        le_1185 = np.where(lambda_r <= 1185)[0]
        pcaspec[le_1185] *= mfluxcorr(lambda_r[le_1185], delta_mf)
    
    else:
        # PCA-only continuum
        pcaparams,afluxerr = pca_chisq(ff, lambda_r, sigma_in, ivar=ivar_in, 
                              afluxerr=afluxerr, quiet=True, 
                              dr7eigen=dr7eigen, m_fit=m_fit)       # check pca_chisq return 
        
        cz = pcaparams[0]
        pcaspec = pca_func(lambda_r, pcaparams, dr7eigen=dr7eigen)
        pcacont = pcaspec.copy()
    
    # Restore full wavelength range if needed
    if len(wave_excess) > 0:
        pcaspec_full[wave_cut] = pcaspec
        pcaspec = pcaspec_full
        
        pcacont_tmp = np.zeros_like(pcaspec)
        pcacont_tmp[wave_cut] = pcacont
        pcacont = pcacont_tmp
    
    return pcaspec
