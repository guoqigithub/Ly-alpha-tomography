import numpy as np

def forestvar_func(z_in: np.ndarray) -> np.ndarray:
    """
    Calculate intrinsic variance of Lyman-alpha forest.
    
    Based on the estimate from McDonald et al. 2006.
    
    Args:
        z_in (np.ndarray): Input redshift array
    
    Returns:
        np.ndarray: Calculated forest variance
    """
    return 0.065 * ((1 + z_in) / (1 + 2.25))**3.8