import numpy as np
from numba import njit

# DEFINE EMBEDDING NODE HELPER FUNCS   
@njit
def _embed_1d(x, W):
    embed_dim, _ = W.shape
    m = x.shape[0]
    X_embed = np.empty((m, embed_dim), dtype=W.dtype)
    for i in range(m):
        X_embed[i, :] = W[:, x[i]]
    return X_embed

@njit
def _embed_2d(X, W):
    embed_dim, _ = W.shape
    m, n = X.shape
    X_embed = np.empty((m, n, embed_dim), dtype=W.dtype)
    for i in range(m):
        for j in range(n):
            X_embed[i, j, :] = W[:, X[i, j]]
    return X_embed

@njit
def _dW_embed_1d(g, x, W):
    m = x.shape[0]
    dW = np.zeros_like(W)
    for i in range(m):
        dW[:, x[i]] += g[i, :] 
    return dW

@njit
def _dW_embed_2d(g, X, W):
    m, n = X.shape
    dW = np.zeros_like(W)
    for i in range(m):
        for j in range(n):
            dW[:, X[i, j]] += g[i, j, :] 
    return dW

def embed(X, W):
    """adsf"""
    
    if X.ndim == 2:
        return _embed_2d(X, W)
    return _embed_1d(X, W)

def _dW_embed(g, X, W):
    if X.ndim == 2:
        return _dW_embed_2d(g, X, W)
    return _dW_embed_1d(g, X, W)