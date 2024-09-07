import numpy as np


# DEFINE MAIN HELPER FUNCS FOR NON-PARALLEL MODE
def _bnXd(Z, gamma, beta, eps):
    """batch norm function for 1 and 2 dimensional batch norm operations"""
    if Z.ndim == 4:
        summation_axes = (0, 2, 3)
        reshape_arr = [1, 1]
        B = Z.shape[0] * Z.shape[2] * Z.shape[3] 
    else:
        summation_axes = (1, )
        reshape_arr = [1]
        B = Z.shape[1]
    
    # compute batch statistics (at the expense of memory, compute the statistics in 
    # 64 bit in order to avoid overflow/underflow issues)
    mu = np.mean(Z.astype(np.float64), axis=summation_axes).astype(Z.dtype)
    mu_ = mu.reshape(-1, *reshape_arr)
    var = np.var(Z.astype(np.float64), ddof=0, axis=summation_axes).astype(Z.dtype)
    sigma = np.sqrt(var + eps) 
    
    # reshape to allow proper broadcasting
    sigma_ = sigma.reshape(-1, *reshape_arr)
    gamma_ = gamma.reshape(-1, *reshape_arr)
    beta_ = beta.reshape(-1, *reshape_arr)
    
    # compute the batchnorm output
    Z_prime = (Z - mu_) / sigma_
    Z_BN = gamma_ * Z_prime + beta_
    
    return Z_BN, Z_prime, sigma, mu, var

def _grads_bnXd(dZ_BN, Z_prime, gamma, sigma):
    """batch norm grad function for 1 and 2 dimensional batch norm operations"""
    if dZ_BN.ndim == 4:
        summation_axes = (0, 2, 3)
        reshape_arr = [1, 1]
        B = dZ_BN.shape[0] * dZ_BN.shape[2] * dZ_BN.shape[3] 
    else:
        summation_axes = (1, )
        reshape_arr = [1]
        B = dZ_BN.shape[1]
    
    # compute grads of batchnorm "weight" and "bias"
    dgamma = np.sum(Z_prime * dZ_BN, axis=summation_axes)    
    dbeta = np.sum(dZ_BN, axis=summation_axes)  
    
    # reshape to allow proper broadcasting
    gamma_ = gamma.reshape(-1, *reshape_arr)
    sigma_ = sigma.reshape(-1, *reshape_arr)
    dgamma_ = dgamma.reshape(-1, *reshape_arr)
    dbeta_ = dbeta.reshape(-1, *reshape_arr)
    
    # compute grad of input
    dZ = gamma_ / sigma_ * (dZ_BN - dbeta_ / B - Z_prime * dgamma_ / B)
    
    return dgamma, dbeta, dZ.astype(dZ_BN.dtype)