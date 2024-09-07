import numpy as np
from dougnet.functions._batch_norm._batch_norm_multithreaded import (_bnXd_parallel, 
                                                                        _grads_bnXd_parallel)
from dougnet.functions._batch_norm._batch_norm_singlethreaded import (_bnXd, 
                                                                        _grads_bnXd)


bn_docstring = """
Perform a batch norm operation on a rank-{0} input {1}.

Parameters
----------
Z : np.ndarray of shape {2}
    Input matrix.

gamma : np.ndarray of shape {3}
    "Weight" vector to be broacasted and multiplied with the normalized 
    input matrix.

beta : np.ndarray of shape {3}
    "Bias" vector to be broacasted and added to the normalized 
    input matrix.

eps : float, default=1e-5
    Safety constant added to the variance for numerical stability.
    
parallel : bool, default=True
    Compute batch norm operation with multi-threading.

return_cache : bool, default=False
    In addition to returning the batch norm of the input, return a tuple 
    of tensors which are useful for the backward pass: (Z_prime, sigma, mu, var).

Returns
-------
Z_BN : np.ndarray of shape {2}
    The batch norm of the input
"""


# DEFINE MAIN 1d AND 2d BATCH NORM FUNCS
def bn1d(Z, gamma, beta, eps=1e-5, parallel=True, return_cache=False):
    if parallel:
        Z_BN, Z_prime, sigma, mu, var = _bnXd_parallel(Z, gamma, beta, eps)
    else:
        Z_BN, Z_prime, sigma, mu, var = _bnXd(Z, gamma, beta, eps)
        
    if return_cache:
        return Z_BN, (Z_prime, sigma, mu, var)
    return Z_BN

def bn2d(Z, gamma, beta, eps=1e-5, parallel=True, return_cache=False):
    """same func as above.  using the bn1d/bn2d distinction to be consistent with pytorch"""
    if parallel:
        Z_BN, Z_prime, sigma, mu, var = _bnXd_parallel(Z, gamma, beta, eps)
    else:
        Z_BN, Z_prime, sigma, mu, var = _bnXd(Z, gamma, beta, eps)
        
    if return_cache:
        return Z_BN, (Z_prime, sigma, mu, var)
    return Z_BN

bn1d.__doc__ = bn_docstring.format("2", "m x n", "(m, n)", "m, ")
bn2d.__doc__ = bn_docstring.format("4", "N x C x H x W", "(N, C, H, W)", "C, ")
        

# DEFINE MAIN BATCH NORM FUNCS FOR GRADIENTS
def _grads_bn(dZ_BN, Z_prime, gamma, sigma, parallel=True):
    if parallel:
        return _grads_bnXd_parallel(dZ_BN, Z_prime, gamma, sigma)
    return _grads_bnXd(dZ_BN, Z_prime, gamma, sigma)