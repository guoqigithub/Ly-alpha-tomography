def taueff_evo(zin):
    """
    Calculate effective tau based on redshift.
    
    Parameters:
    -----------
    zin : float or array-like
        Redshift value(s)
    
    Returns:
    --------
    float or array-like
        Effective tau value(s)
    """
    tauevo = 0.001845 * (1 + zin)**3.924
    return tauevo