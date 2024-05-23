import numpy as np

def bn2d(Z, gamma, beta, delta=1e-5, return_cache=False):
    """
    Perform a batch norm operation on a rank-4 input (N x C x H x W).
    
    Parameters
    ----------
    Z : np.ndarray of shape (N, C, H, W)
        Input tensor.  Should be c-contiguous for optimal performance.

    gamma : np.ndarray of shape (C,)
        "Weight" tensor to be broacasted and applied to each channel of 
        the normalized input.
        
    beta : np.ndarray of shape (C,)
        "Bias" tensor to be broacasted and added to each channel of 
        the normalized input.
    
    delta : float, default=1e-5
        Safety constant added to the variance for numerical stability.
    
    return_cache : bool, default=False
        In addition to returning the batch norm of the input, return a tuple 
        of tensors which are useful for the backward pass: (Z_prime, gamma_tilde, 
        sigma_tilde).

    Returns
    -------
    Z_BN : np.ndarray of shape (N, C, H, W)
        The batch norm of the input
    """
    # get shapes
    N, C, H, W = Z.shape
    dtype = Z.dtype
    
    # broadcast gamma and beta
    gamma_tilde = gamma.reshape(C, 1, 1)
    beta_tilde = beta.reshape(C, 1, 1)

    # numpy seems to have a peculiarity where if the values in a float32 ndarray
    # become too small (i.e., when N * H * W is too large), it automatically casts
    # the array to float64, so I make sure to enforce the dtype that I desire with
    # .astype(dtype)
    mu = (np.einsum('bhij->h', Z) / (N * H * W)).astype(dtype)
    mu_tilde = mu.reshape(C, 1, 1)

    sigma = np.sqrt((np.einsum('bhij->h', (Z - mu_tilde) ** 2) / (N * H * W)).astype(dtype) + delta)
    sigma_tilde = sigma.reshape(C, 1, 1)
    
    Z_prime = (Z - mu_tilde) / sigma_tilde
    Z_BN = gamma_tilde * Z_prime + beta_tilde
    
    if return_cache:
        return Z_BN, (Z_prime, gamma_tilde, sigma_tilde)
    return Z_BN

def _dgamma(dZ_BN, Z_prime):
    """
    return dL/dgamma for the conv2d operation
    """
    return np.einsum('bhij->h', Z_prime * dZ_BN)

def _dbeta(dZ_BN):
    """
    return dL/dbeta for the conv2d operation
    """
    return np.einsum('bhij->h', dZ_BN)

def _dZ(dZ_BN, gamma_tilde, sigma_tilde):
    """
    return dL/dZ for the conv2d operation
    """
    return (gamma_tilde * dZ_BN) / sigma_tilde