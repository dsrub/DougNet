import numpy as np
from math import sqrt
from numba import njit, prange


# JITTED HELPER FUNCS FOR 2D BN IN PARALLEL MODE
@njit(parallel=True)
def _moments2d(X):
    """compute E[X] and Var[X] in parallel"""
    N, C, H, W = X.shape
    B = N * H * W
   
    # compute moments in 64 bit in order to avoid overflow/underflow issues
    EX = np.zeros(C, dtype=np.float64)
    EX2 = np.zeros(C, dtype=np.float64)
    for c in prange(C):
        for n in range(N):
            for i in range(H):
                for j in range(W):
                    x = X[n, c, i, j]
                    EX[c] += x
                    EX2[c] += (x ** 2)
    EX /= B
    return EX.astype(X.dtype), (EX2 / B - EX ** 2).astype(X.dtype)

@njit(parallel=True)
def _Z_prime2d(Z, mu, sigma, gamma, beta):
    """compute Z_prime and Z_BN in parallel"""
    N, C, H, W = Z.shape
    
    Z_prime = np.empty_like(Z)
    Z_BN = np.empty_like(Z)
    for c in prange(C):
        mu_c = mu[c]
        sigma_c = sigma[c]
        gamma_c = gamma[c]
        beta_c = beta[c]
        for n in range(N):
            for i in range(H):
                for j in range(W):
                    z_prime = (Z[n, c, i, j] - mu_c) / sigma_c
                    Z_prime[n, c, i, j] = z_prime
                    Z_BN[n, c, i, j] = gamma_c * z_prime + beta_c
    return Z_prime, Z_BN

@njit(parallel=True)
def _dgamma_dbeta2d(Z_prime, dZ_BN):
    """compute dgamma and dbeta in parallel"""
    N, C, H, W = Z_prime.shape
    
    dgamma = np.zeros(C, dtype=dZ_BN.dtype)
    dbeta = np.zeros(C, dtype=dZ_BN.dtype)
    for c in prange(C):
        for n in range(N):
            for i in range(H):
                for j in range(W):
                    dz_bn = dZ_BN[n, c, i, j]
                    dgamma[c] += Z_prime[n, c, i, j] * dz_bn
                    dbeta[c] += dz_bn    
    return dgamma, dbeta

@njit(parallel=True)
def _dZ2d(gamma, sigma, dZ_BN, dbeta, Z_prime, dgamma):
    """
    compute dZ in parallel using the formula: 
    dZ = gamma / sigma * (dZ_BN - dbeta / B - Z_prime * dgamma / B)
    """
    N, C, H, W = Z_prime.shape
    B = N * H * W
    
    dZ = np.empty_like(dZ_BN)
    for c in prange(C):
        g_over_s_c = gamma[c] / sigma[c]
        db_over_B_c = dbeta[c] / B
        dg_over_B_c = dgamma[c] / B
        for n in range(N):
            for i in range(H):
                for j in range(W):
                    dZ[n, c, i, j] = g_over_s_c * (dZ_BN[n, c, i, j] - db_over_B_c - Z_prime[n, c, i, j] * dg_over_B_c)   
    return dZ


# JITTED HELPER FUNCS FOR 1D BN IN PARALLEL MODE
@njit(parallel=True)
def _moments1d(X):
    """compute E[X] and Var[X] in parallel"""
    C, N = X.shape
   
    # compute moments in 64 bit in order to avoid overflow/underflow issues
    EX = np.zeros(C, dtype=np.float64)
    EX2 = np.zeros(C, dtype=np.float64)
    for c in prange(C):
        for n in range(N):
            x = X[c, n]
            EX[c] += x
            EX2[c] += (x ** 2)
    EX /= N
    return EX.astype(X.dtype), (EX2 / N - EX ** 2).astype(X.dtype)


@njit(parallel=True)
def _Z_prime1d(Z, mu, sigma, gamma, beta):
    """compute Z_prime and Z_BN in parallel"""
    C, N = Z.shape
    
    Z_prime = np.empty_like(Z)
    Z_BN = np.empty_like(Z)
    for c in prange(C):
        mu_c = mu[c]
        sigma_c = sigma[c]
        gamma_c = gamma[c]
        beta_c = beta[c]
        for n in range(N):
            z_prime = (Z[c, n] - mu_c) / sigma_c
            Z_prime[c, n] = z_prime
            Z_BN[c, n] = gamma_c * z_prime + beta_c
    return Z_prime, Z_BN

@njit(parallel=True)
def _dgamma_dbeta1d(Z_prime, dZ_BN):
    """compute dgamma and dbeta in parallel"""
    C, N = Z_prime.shape
    
    dgamma = np.zeros(C, dtype=dZ_BN.dtype)
    dbeta = np.zeros(C, dtype=dZ_BN.dtype)
    for c in prange(C):
        for n in range(N):
            dz_bn = dZ_BN[c, n]
            dgamma[c] += Z_prime[c, n] * dz_bn
            dbeta[c] += dz_bn    
    return dgamma, dbeta

@njit(parallel=True)
def _dZ1d(gamma, sigma, dZ_BN, dbeta, Z_prime, dgamma):
    """
    compute dZ in parallel using the formula: 
    dZ = gamma / sigma * (dZ_BN - dbeta / B - Z_prime * dgamma / B)
    """
    C, N = Z_prime.shape
    B = N
    
    dZ = np.empty_like(dZ_BN)
    for c in prange(C):
        g_over_s_c = gamma[c] / sigma[c]
        db_over_B_c = dbeta[c] / B
        dg_over_B_c = dgamma[c] / B
        for n in range(N):
            dZ[c, n] = g_over_s_c * (dZ_BN[c, n] - db_over_B_c - Z_prime[c, n] * dg_over_B_c)   
    return dZ


# DEFINE MAIN HELPER FUNCS FOR PARALLEL MODE
def _bnXd_parallel(Z, gamma, beta, eps):
    """batch norm function for 1 and 2 dimensional batch norm operations in parallel mode"""
    if Z.ndim == 4:
        _moments, _Z_prime  = _moments2d, _Z_prime2d
    else:
        _moments, _Z_prime = _moments1d, _Z_prime1d
    
    # compute statistics
    mu, var = _moments(Z)
    sigma = np.sqrt(var + eps) 
    
    # perform batchnorm normalization
    Z_prime, Z_BN = _Z_prime(Z, mu, sigma, gamma, beta)
    
    return Z_BN, Z_prime, sigma, mu, var

    
def _grads_bnXd_parallel(dZ_BN, Z_prime, gamma, sigma):
    """batch norm grad function for 1 and 2 dimensional batch norm operations in parallel mode"""
    if dZ_BN.ndim == 4:
        _dgamma_dbeta, _dZ = _dgamma_dbeta2d, _dZ2d
    else:
        _dgamma_dbeta, _dZ = _dgamma_dbeta1d, _dZ1d
    
    # compute grads
    dgamma, dbeta = _dgamma_dbeta(Z_prime, dZ_BN) 
    dZ = _dZ(gamma, sigma, dZ_BN, dbeta, Z_prime, dgamma)
    
    return dgamma, dbeta, dZ