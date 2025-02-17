import numpy as np
import os
import matplotlib.pyplot as plt
# from .pca_func import pca_func
import scipy.optimize as optimize
from scipy import interpolate
import warnings
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, splev, splrep

def smooth(signal, owidth, edge_truncate=False):
    """Replicates the IDL ``SMOOTH()`` function.

    Parameters
    ----------
    signal : array-like
        The array to be smoothed.
    owidth : :class:`int` or array-like
        Width of the smoothing window.  Can be a scalar or an array with
        length equal to the number of dimensions of `signal`.
    edge_truncate : :class:`bool`, optional
        Set `edge_truncate` to ``True`` to apply smoothing to all points.
        Points near the edge are normally excluded from smoothing.

    Returns
    -------
    array-like
        A smoothed array with the same dimesions and type as `signal`.

    References
    ----------
    https://www.nv5geospatialsoftware.com/docs/smooth.html

    """
    if owidth % 2 == 0:
        width = owidth + 1
    else:
        width = owidth
    if width < 3:
        return signal
    n = signal.size
    istart = int((width-1)/2)
    iend = n - int((width+1)/2)
    w2 = int(width/2)
    s = signal.copy()
    for i in range(n):
        if i < istart:
            if edge_truncate:
                s[i] = (signal[0:istart+i+1].sum() +
                        (istart-i)*signal[0])/float(width)
        elif i > iend:
            if edge_truncate:
                s[i] = (signal[i-istart:n].sum() +
                        (i-iend)*signal[n-1])/float(width)
        else:
            s[i] = signal[i-w2:i+w2+1].sum()/float(width)
    return s

###############################################################################################################################################
def pca_func(x, p, dr7eigen=False):
    """
    Return normalized flux for a given set of PCA weights.
    
    Parameters:
    -----------
    x : array_like
        Wavelength in quasar restframe
    p : array_like
        Parameter array
        p[0]: wavelength correction factor
        p[1]: flux normalization correction
        p[2]: power law component
        p[3:]: PCA weight coefficients
    dr7eigen : bool, optional
        Use DR7 eigenspectra
    
    Returns:
    --------
    ff : ndarray
        Normalized flux
    """
    # Select appropriate PCA data based on dr7eigen flag
    if dr7eigen:
        lambxi_tmp = SDSSPCABlock.lambxi_sdss
        mu_pca_tmp = SDSSPCABlock.mu_pca_sdss
        xi_pca_tmp = SDSSPCABlock.xi_pca_sdss
        mmax_tmp = SDSSPCABlock.mmax_sdss
    else:
        lambxi_tmp = PCABlock.lambxi
        mu_pca_tmp = PCABlock.mu_pca
        xi_pca_tmp = PCABlock.xi_pca
        mmax_tmp = PCABlock.mmax

    # Extract parameters
    cij = p[3:]
    cz = p[0]
    fnorm = p[1]
    alpha = p[2]
    
    # Compute weighted principal components
    weightedpcs = xi_pca_tmp * cij[:, np.newaxis]
    sum_pcs = np.sum(weightedpcs, axis=0)
    
    # Reconstruct PCA spectrum
    pcaspec = mu_pca_tmp + sum_pcs
    
    # Normalize to mean flux between 1275-1285
    norm_mask = (lambxi_tmp >= 1275) & (lambxi_tmp <= 1285)
    pcaspec = pcaspec / np.mean(pcaspec[norm_mask])
    
    # Interpolate and apply power law correction
    ff = fnorm * np.interp(cz * x, lambxi_tmp, pcaspec) * (cz * x / 1280)**alpha
    
    return ff

def pca_func_dr7(x, p, lambxi_sdss, mu_pca_sdss, sig_pca_sdss, xi_pca_sdss, mmax_sdss):
    """
    PCA function specifically intended for fitting with Paris+11 eigenspectra.
    
    Parameters:
    x : array-like
        Input wavelength values
    p : array-like
        Parameters for the PCA function:
        - p[0]: cz (redshift)
        - p[1]: fnorm (flux normalization)
        - p[2]: alpha (spectral index)
        - p[3:]: cij (eigenspectrum coefficients)
    lambxi_sdss : array-like
        Wavelength array
    mu_pca_sdss : array-like
        Mean PCA spectrum
    sig_pca_sdss : array-like
        PCA spectrum standard deviation
    xi_pca_sdss : array-like
        PCA eigenvectors
    mmax_sdss : int
        Maximum number of eigenvectors
    
    Returns:
    array-like: Normalized flux values
    """
    # Extract parameters
    cz = p[0]
    fnorm = p[1]
    alpha = p[2]
    cij = p[3:]
    
    # Copy eigenvectors
    weightedpcs = xi_pca_sdss.copy()
    
    # Weight eigenvectors
    for mm in range(mmax_sdss):
        weightedpcs[mm, :] = cij[mm] * xi_pca_sdss[mm, :]
    
    # Sum weighted eigenvectors
    sum_pcs = np.sum(weightedpcs, axis=0)
    
    # Compute PCA spectrum
    pcaspec = mu_pca_sdss + sum_pcs
    
    # Normalize spectrum
    norm_region = (lambxi_sdss >= 1275.0) & (lambxi_sdss <= 1285.0)
    pcaspec = pcaspec / np.mean(pcaspec[norm_region])
    
    # Interpolate and apply spectral scaling
    ff = fnorm * np.interp(cz * x, lambxi_sdss, pcaspec) * (cz * x / 1280.0)**alpha
    
    return ff

###############################################################################################################################################

def smooth_flux(flux, sig):
    """
    Smooths the flux by a window of width sig pixels using FFT.
    
    Parameters:
    -----------
    flux : array_like
        Input flux array
    sig : float
        Smoothing width in pixels
    
    Returns:
    --------
    sm : ndarray
        Smoothed flux array
    """
    nn = len(flux)
    kk = np.arange(nn)
    w = kk > nn / 2
    kk[w] = nn - kk[w]
    kk = 2 * np.pi * kk / nn
    
    # Perform FFT, apply Gaussian smoothing, then inverse FFT
    fk = np.fft.fft(flux) * np.exp(-0.5 * (kk * sig)**2)
    sm = np.fft.ifft(fk)
    
    return np.float32(sm.real)

def calc_chisq(ff, sigma, lambda_arr, pcaparams, dof=None, masklist=None):
    """
    Evaluate the reduced chi-squared from the best-fit PCA parameters.
    
    Parameters:
    -----------
    ff : array_like
        Observed flux
    sigma : array_like
        Flux errors
    lambda_arr : array_like
        Wavelength array
    pcaparams : array_like
        Best-fit PCA parameters
    dof : int, optional
        Degrees of freedom
    masklist : array_like, optional
        Mask list
    
    Returns:
    --------
    chisq_dof : float
        Reduced chi-squared value
    """
    # Select wavelengths between 1220 and 1600
    mask_red = (lambda_arr > 1220) & (lambda_arr <= 1600)
    lambda_red = lambda_arr[mask_red]
    ff_red = ff[mask_red]
    sigma_red = sigma[mask_red]
    
    cz = pcaparams[0]
    ff_model = pca_func(lambda_red, pcaparams)
    
    if dof is None:
        dof = len(ff_red) - len(pcaparams)
    
    chisq = ((ff_red - ff_model)**2) / (sigma_red**2)
    chisq_dof = np.sum(chisq) / dof
    
    return chisq_dof

def absfluxerr(ff_in, lambda_arr, pcaparams, dr7eigen=None):
    """
    Compute absolute flux error between reconstruction and prediction.
    
    Parameters:
    -----------
    ff_in : array_like
        Input flux
    lambda_arr : array_like
        Wavelength array
    pcaparams : array_like
        Best-fit PCA parameters
    dr7eigen : array_like, optional
        DR7 eigenspectra
    
    Returns:
    --------
    float
        Average absolute flux deviation
    """
    # Apply smoothing
    ff = np.convolve(ff_in, np.ones(15)/15, mode='same')
    # ff = smooth(ff_in,15)
    
    cz = pcaparams[0]
    # Select wavelengths between 1230*cz and 1600*cz
    mask_red = (lambda_arr > 1230 * cz) & (lambda_arr <= 1600 * cz)
    lambda_red = lambda_arr[mask_red]
    ff_red = ff[mask_red]
    
    ff_model = pca_func(lambda_red, pcaparams, dr7eigen=dr7eigen)
    ff_dev = np.abs((ff_model / ff_red) - 1)
    
    return np.mean(ff_dev)

def read_pca_hst(mmax_in=10, plotcomp=None):
    """
    Read Nao Suzuki's HST PCA eigenspectra.
    
    Parameters:
    -----------
    mmax_in : int, optional
        Maximum number of components (default: 10)
    plotcomp : int, optional
        Component to plot
    
    Returns:
    --------
    pca_block : PCABlock
        PCA data block
    """
    # pca_block = PCABlock(mmax_in)
    
    # Get PCA data directory from environment variable
    xidir = '/lustre/work/guoqi/code/PFS/CLAMATO/clamato_test/mf_pca/data/' 
    xihst_file = os.path.join(xidir, 'xi_hst.txt')
    
    # Load data, skipping header lines
    data = np.genfromtxt(xihst_file, skip_header=15)
    
    lambxi = data[:, 0]
    mu_pca = data[:, 1]
    sig_pca = data[:, 2]
    
    # Reshape eigenspectra into a 2D array
    xi_tmp = data[:, 3:(3+mmax_in)].T
    
    xi_pca = xi_tmp[:mmax_in]
    
    # Optional plotting
    if plotcomp is not None:
        plt.plot(lambxi, xi_pca[plotcomp-1])
        plt.title(f'PCA Component {plotcomp}')
        plt.xlabel('Wavelength')
        plt.ylabel('Eigenspectrum')
        plt.show()

    # print("pca_hst")
    # print(lambxi,mu_pca,sig_pca,xi_pca,mmax_in)
    return lambxi,mu_pca,sig_pca,xi_pca,mmax_in

def read_pca_sdss(mmax_in=10, plotcomp=None):
    """
    Read Isabelle Paris' SDSS PCA eigenspectra.
    
    Parameters:
    -----------
    mmax_in : int, optional
        Maximum number of components (default: 10)
    plotcomp : int, optional
        Component to plot
    
    Returns:
    --------
    pca_block : SDSSPCABlock
        SDSS PCA data block
    """
    # pca_block = SDSSPCABlock(mmax_in)
    
    # Get PCA data directory from environment variable
    xidir = '/lustre/work/guoqi/code/PFS/CLAMATO/clamato_test/mf_pca/data/' 
    xisdss_file = os.path.join(xidir, 'xi_sdss.txt')
    
    # Load data
    data = np.genfromtxt(xisdss_file)
    
    lambxi_sdss = data[:, 0]
    mu_pca_sdss = data[:, 1]
    
    # Reshape eigenspectra into a 2D array
    xi_tmp = data[:, 2:(2+mmax_in)].T
    
    # Wavelength cutoff (<=1600A)
    wave_cut = lambxi_sdss <= 1600
    
    lambxi_sdss = lambxi_sdss[wave_cut]
    mu_pca_sdss = mu_pca_sdss[wave_cut]
    xi_tmp = xi_tmp[:, wave_cut]
    
    # Initialize standard deviation array
    sig_pca_sdss = np.zeros_like(mu_pca_sdss)
    
    xi_pca_sdss = xi_tmp[:mmax_in]
    
    # Optional plotting
    if plotcomp is not None:
        plt.plot(lambxi_sdss, xi_pca_sdss[plotcomp-1])
        plt.title(f'SDSS PCA Component {plotcomp}')
        plt.xlabel('Wavelength')
        plt.ylabel('Eigenspectrum')
        plt.show()

    # print("pca_sdss")
    # print(lambxi_sdss,mu_pca_sdss,sig_pca_sdss,xi_pca_sdss,mmax_in)
    
    return lambxi_sdss,mu_pca_sdss,sig_pca_sdss,xi_pca_sdss,mmax_in

class PCABlock:
    lambxi, mu_pca, sig_pca, xi_pca, mmax = read_pca_hst(mmax_in=10)

class SDSSPCABlock:
    lambxi_sdss, mu_pca_sdss, sig_pca_sdss, xi_pca_sdss, mmax_sdss = read_pca_sdss(mmax_in=10)

def param_constraints(fixz=False, dr7eigen=False, mmax=10):
    """
    Returns parameter constraints for MPFIT routine.
    
    Parameters:
    -----------
    fixz : bool, optional
        Fix redshift (default: False)
    dr7eigen : bool, optional
        Use DR7 eigenspectra (default: False)
    mmax : int, optional
        Number of PCA components (default: 10)
    
    Returns:
    --------
    dict
        Dictionary of parameter constraints
    """
    n_notpca = 3
    
    # Initialize parameter constraints
    paramconst = [{
        'fixed': 0,
        'limited': [0, 0],
        'limits': [0.0, 0.0]
    } for _ in range(mmax + n_notpca)]
    
    # Fix redshift if requested
    if fixz:
        paramconst[0]['fixed'] = 1
    
    # Constrain redshift correction factor
    paramconst[0]['limited'] = [1, 1]
    paramconst[0]['limits'] = [0.9, 1.1]
    
    # Constrain power law index
    paramconst[2]['limited'] = [1, 1]
    paramconst[2]['limits'] = [-1.0, 1.0]
    
    # Constrain PCA weights
    for i in range(n_notpca, mmax + n_notpca):
        paramconst[i]['limited'] = [1, 1]
    
    # Specific constraints based on eigenspectra
    if not dr7eigen:
        # Suzuki 2006 constraints
        if mmax >= 1:
            paramconst[n_notpca]['limits'] = [-15.0, 15.0]
        if mmax >= 2:
            for i in range(n_notpca + 1, mmax + n_notpca):
                paramconst[i]['limits'] = [-7.2, 7.2]
        if mmax >= 3:
            for i in range(n_notpca + 2, mmax + n_notpca):
                paramconst[i]['limits'] = [-5.0, 5.0]
        if mmax >= 5:
            for i in range(n_notpca + 4, mmax + n_notpca):
                paramconst[i]['limits'] = [-3.0, 3.0]
        if mmax >= 7:
            for i in range(n_notpca + 6, mmax + n_notpca):
                paramconst[i]['limits'] = [-2.0, 2.0]
        if mmax >= 8:
            for i in range(n_notpca + 7, mmax + n_notpca):
                paramconst[i]['limits'] = [-1.5, 1.5]
    else:
        # DR7 eigenspectra constraints
        if mmax >= 1:
            paramconst[n_notpca]['limits'] = [-7.0, 5.0]
        if mmax >= 2:
            for i in range(n_notpca + 1, mmax + n_notpca):
                paramconst[i]['limits'] = [-1.6, 1.6]
        if mmax >= 5:
            for i in range(n_notpca + 3, mmax + n_notpca):
                paramconst[i]['limits'] = [-1.0, 1.0]
        if mmax >= 7:
            for i in range(n_notpca + 6, mmax + n_notpca):
                paramconst[i]['limits'] = [-0.65, 0.65]
    
    return paramconst


def pca_weights(ff_in, lambda_in, interpflag=0, dr7eigen=False):
    """
    Estimate PCA weights c_ij from PCA components.
    
    Parameters:
    -----------
    ff_in : array_like
        Flux spectrum of the quasar
    lambda_in : array_like
        Wavelength in restframe of quasar
    interpflag : int, optional
        Interpolation method (0: linear, 1: spline)
    dr7eigen : bool, optional
        Use DR7 eigenspectra
    
    Returns:
    --------
    c_ij : ndarray
        PCA weights vector
    """
    # Select appropriate PCA data based on dr7eigen flag
    if dr7eigen:
        lambxi_tmp = lambda_tmp = SDSSPCABlock.lambxi_sdss
        mu_pca_tmp = SDSSPCABlock.mu_pca_sdss
        xi_pca_tmp = SDSSPCABlock.xi_pca_sdss
        mmax_tmp = SDSSPCABlock.mmax_sdss
    else:
        lambxi_tmp = PCABlock.lambxi
        mu_pca_tmp = PCABlock.mu_pca
        xi_pca_tmp = PCABlock.xi_pca
        mmax_tmp = PCABlock.mmax

    # print("pca_weights")
    # print(lambxi_tmp,mu_pca_tmp,xi_pca_tmp,mmax_tmp)
    # Smooth and normalize spectrum
    # ff = np.convolve(ff_in, np.ones(10)/10, mode='same')
    ff = smooth(ff_in,10)
    
    # Normalization
    norm_mask = (lambda_in >= 1275) & (lambda_in <= 1285)
    if np.any(norm_mask):
        normfac = np.mean(ff[norm_mask])
    else:
        norm_mask = (lambda_in >= 1450) & (lambda_in <= 1470)
        normfac = np.mean(ff[norm_mask])
    
    ff = ff / normfac
    
    # Select wavelength range redwards of Lya
    red_mask = (lambda_in >= 1210) & (lambda_in < 1600)
    lamb_red = lambda_in[red_mask]
    ff_red = ff[red_mask]
    
    # Interpolate mean spectrum
    if interpflag == 0:
        mu = np.interp(lamb_red, lambxi_tmp, mu_pca_tmp)
    else:
        # Use scipy's interpolate for spline
        tck = splrep(lambxi_tmp, mu_pca_tmp)
        mu = splev(lamb_red, tck)
        
    # Interpolate eigenspectra
    xi = np.zeros((mmax_tmp, len(lamb_red)))
    for m in range(mmax_tmp):
        if interpflag == 0:
            xi[m, :] = np.interp(lamb_red, lambxi_tmp, xi_pca_tmp[m, :])
        else:
            tck = splrep(lambxi_tmp, xi_pca_tmp[m, :])
            xi[m, :] = splev(lamb_red, tck)
            
    # Calculate PCA weights via matrix multiplication
    resid = ff_red - mu
    c_ij = np.zeros(mmax_tmp)
    for m in range(mmax_tmp):
        c_ij[m] = np.dot(xi[m, :], resid)  
        
    return c_ij


def pca_chisq(ff, lambda_r, sigma, ivar=None, dof=None, 
              afluxerr=None, quiet=False, contiter=None, 
              maskwave=None, dr7eigen=False, m_fit=None):
    """
    Perform PCA Chi-Square fitting on a quasar spectrum.
    
    Parameters:
    -----------
    ff : array
        Flux values
    lambda_r : array
        Rest-frame wavelength
    sigma : array
        Flux errors
    ivar : array, optional
        Inverse variance
    dof : int, optional
        Degrees of freedom
    afluxerr : float, optional
        Absolute flux error
    quiet : bool, optional
        Suppress printing
    contiter : array, optional
        Continuum iterations
    maskwave : array, optional
        Masked wavelengths
    dr7eigen : bool, optional
        Use DR7 eigenspectra
    m_fit : float, optional
        Limit number of PCA components
    
    Returns:
    --------
    array
        Fitted parameters
    """
    # Normalize flux
    normpix1 = np.where((lambda_r >= 1275.0) & (lambda_r <= 1285.0))[0]
    
    if len(normpix1) > 0 and np.sum(ivar[normpix1]) != 0:
        normfac = np.average(ff[normpix1] * ivar[normpix1]) / np.average(ivar[normpix1])
    else:
        normpix2 = np.where((lambda_r >= 1450.0) & (lambda_r <= 1470.0))[0]
        normfac = np.average(ff[normpix2] * ivar[normpix2]) / np.average(ivar[normpix2])
    
    ff_norm = ff / normfac
    ivar_norm = ivar * normfac**2
    
    # Select pixels redwards of Lya
    redlist = np.where((lambda_r >= 1216.0) & (lambda_r < 1600.0))[0]
    lambda_red = lambda_r[redlist]
    ff_red = ff_norm[redlist]
    ivar_red = ivar_norm[redlist]

    if sigma is not None:
        sigma_norm = sigma / normfac
        sigma_red = sigma_norm[redlist]

    snmed = np.median(ff_red * np.sqrt(ivar_red))
    # Hack: if S/N is more than 8, then increase noise since high-SN
    # objects seem to have crappier fits
    if snmed > 8.0:
        ivar_red *= (8.0 / snmed) ** 2
        if sigma is not None:
            sigma_red *= snmed / 8.0
    
    # Initial PCA weights estimation
    ff_weights = ff_red
    cij = pca_weights(ff_weights, lambda_red, interpflag=1, dr7eigen=dr7eigen)

    # Prepare initial parameters
    n_notpca = 3
    mmax_tmp = 10  # Default from readpca_sdss or readpca_hst
    # Parameter constraints
    paramconst = param_constraints(dr7eigen=dr7eigen)
    
    # Constrain PCA weights to parameter bounds
    cij_constraint = [param["limits"][1] for param in paramconst[n_notpca:]]
    cij_constraint = np.array(cij_constraint)
    
    # Adjust initial PCA weights to be within constraints
    for mm in range(mmax_tmp):
        if abs(cij[mm]) > cij_constraint[mm]:
            signconst = -1 if cij[mm] < 0 else 1
            cij[mm] = cij_constraint[mm] * signconst * 0.98    

    if m_fit is not None:
        m_fit = round(m_fit)
        if m_fit < mmax_tmp:
            if m_fit == 0:
                cij[:mmax_tmp] = 0
            else:
                cij[m_fit - 1:] = 0            
            # In Python, we'd need a way to "fix" parameters
            # This might involve creating a mask or modifying parameter constraints
            # Placeholder for parameter fixing logic
            for i in range(m_fit+3, len(paramconst)):
                paramconst[i]['fixed'] = 1

    # Initial parameter guess
    inparam = np.zeros(mmax_tmp + n_notpca)
    inparam[0] = 1.0  # redshift correction
    inparam[1] = 1.0  # normalization factor
    inparam[2] = 0.0  # power-law index
    inparam[n_notpca:mmax_tmp+n_notpca] = cij

    ivar_mpfit = ivar_red if ivar is not None else None
    # Prepare sigma_red (dummy array for weights)
    sigma_red = np.zeros_like(ivar_red)

    def mpfit_func(x, *p):
        if dr7eigen:
            lambxi_tmp = SDSSPCABlock.lambxi_sdss
            mu_pca_tmp = SDSSPCABlock.mu_pca_sdss
            xi_pca_tmp = SDSSPCABlock.xi_pca_sdss
            mmax_tmp = SDSSPCABlock.mmax_sdss
        else:
            lambxi_tmp = PCABlock.lambxi
            mu_pca_tmp = PCABlock.mu_pca
            xi_pca_tmp = PCABlock.xi_pca
            mmax_tmp = PCABlock.mmax
        # Extract parameters
        cij = p[3:]
        cz = p[0]
        fnorm = p[1]
        alpha = p[2]

        # Compute weighted principal components
        weightedpcs = xi_pca_tmp * np.array(cij)[:, np.newaxis]
        sum_pcs = np.sum(weightedpcs, axis=0)
        
        # Reconstruct PCA spectrum
        pcaspec = mu_pca_tmp + sum_pcs
        
        # Normalize to mean flux between 1275-1285
        norm_mask = (lambxi_tmp >= 1275) & (lambxi_tmp <= 1285)
        pcaspec = pcaspec / np.mean(pcaspec[norm_mask])
        
        # Interpolate and apply power law correction
        ff = fnorm * np.interp(cz * x, lambxi_tmp, pcaspec) * (cz * x / 1280)**alpha
        
        return ff
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pcafit, _ = curve_fit(
            mpfit_func, 
            lambda_red, 
            ff_red, 
            p0=inparam, 
            sigma=1/np.sqrt(ivar_red) if ivar_red is not None else None,
            bounds=([0.,-5,-1,-7,-1.6,-1.6,-1.6,-1,-1,-0.65,-0.65,-0.65,-0.65],[4,5,1,5,1.6,1.6,1.6,1,1,0.65,0.65,0.65,0.65]),
            # absolute_sigma=True
        )     
    pcaspec_first = pca_func(lambda_red, pcafit, dr7eigen=dr7eigen)
    
    cz = pcafit[0]
    # Criterion for throwing out pixels
    nu_abs = 2.5
    # For high-sn pixels, impose more stringent criteria to avoid throwing
    # pixels due to imperfect fitting
    if snmed > 6.0:
        nu_abs = nu_abs * snmed / 6.0

    abslist = np.where(ff_red - pcaspec_first < -nu_abs/np.sqrt(ivar_red))[0]
    
    # Prepare vectors for iteration
    lambda2 = lambda_red.copy()
    ff2 = ff_red.copy()
    ivar2 = ivar_red.copy()
    sigma2 = sigma_red.copy()

    # Remove masked lines if any
    if len(abslist) > 0:
        lambda2 = np.delete(lambda2, abslist)
        ff2 = np.delete(ff2, abslist)
        ivar2 = np.delete(ivar2, abslist)
        sigma2 = np.delete(sigma2, abslist)

    # Iteration parameters
    crit_delta_fits = 0.02
    n_iter_max = 5
    delta_fits = 1.0
    n_iter = 0
    pcaspec_init = pcaspec_first.copy()

    while n_iter < n_iter_max and delta_fits > crit_delta_fits:
        # Placeholder for iterative fitting logic
        # In actual implementation, this would involve sophisticated fitting
        
        # Simulated iteration steps
        pcafit2, _ = curve_fit(
            mpfit_func, 
            lambda2, 
            ff2, 
            p0=pcafit, 
            sigma=1/np.sqrt(ivar2) if ivar2 is not None else None,
            bounds=([0.,-5,-1,-7,-1.6,-1.6,-1.6,-1,-1,-0.65,-0.65,-0.65,-0.65],[4,5,1,5,1.6,1.6,1.6,1,1,0.65,0.65,0.65,0.65]),
            # absolute_sigma=True
        )     
        # pcafit2 = pcafit.copy()
        pcaspec_second = pca_func(lambda_red, pcafit2, dr7eigen=dr7eigen)
        
        delta_fits = np.mean(np.abs(pcaspec_second/pcaspec_first - 1.0))
        abslist2 = np.where((ff_red - pcaspec_second) < (-nu_abs / np.sqrt(ivar_red)))[0]

        lambda2 = lambda_red.copy()
        ff2 = ff_red.copy()
        sigma2 = sigma_red.copy()
        ivar2 = ivar_red.copy()
        # Remove masked lines from copy of vectors
        if len(abslist) > 0:
            lambda2 = np.delete(lambda2, abslist2)
            ff2 = np.delete(ff2, abslist2)
            sigma2 = np.delete(sigma2, abslist2)
            ivar2 = np.delete(ivar2, abslist2)
            
        n_iter += 1
        pcaspec_first = pcaspec_second

    # Finalize fitting
    pcafit_fin = pcafit2
    pcafit_fin[1] *= normfac
    
    # Placeholder for additional computations
    # afluxerr = absfluxerr(...)
    
    # Setup contiter for tracking iterations
    contiter = np.zeros((3, len(redlist)))
    contiter[0, :] = lambda_r[redlist]
    contiter[1, :] = pcaspec_init
    contiter[2, :] = pcaspec_second
    
    # Default maskwave if not set
    if maskwave is None:
        maskwave = np.array([-1])
    
    return pcafit_fin,afluxerr
