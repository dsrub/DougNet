import numpy as np
from numba import njit

# DEFINE EMBEDDING HELPER FUNCS   
@njit
def _embed_1d(x, W):
    """1d jitted embed function"""
    embed_dim, _ = W.shape
    m = x.shape[0]
    X_embed = np.empty((m, embed_dim), dtype=W.dtype)
    for i in range(m):
        X_embed[i, :] = W[:, x[i]]
    return X_embed

@njit
def _embed_2d(X, W):
    """2d jitted embed function"""
    embed_dim, _ = W.shape
    m, n = X.shape
    X_embed = np.empty((m, n, embed_dim), dtype=W.dtype)
    for i in range(m):
        for j in range(n):
            X_embed[i, j, :] = W[:, X[i, j]]
    return X_embed

@njit
def _dW_embed_1d(g, x, W):
    """1d jitted function to compute dL/dW"""
    m = x.shape[0]
    dW = np.zeros_like(W)
    for i in range(m):
        dW[:, x[i]] += g[i, :] 
    return dW

@njit
def _dW_embed_2d(g, X, W):
    """2d jitted function to compute dL/dW"""
    m, n = X.shape
    dW = np.zeros_like(W)
    for i in range(m):
        for j in range(n):
            dW[:, X[i, j]] += g[i, j, :] 
    return dW


def embed(X, W):
    """
    Embed a tensor of integers, X, representing categorical data into a real-valued 
    vector space via an embedding matrix, W.  
    
    Parameters
    ----------
    X : np.ndarray
        Categorical data to be embedded.  Must be a 1 or 2 dimensional array of ints.
        All integers should be in {0, 1, ..., K - 1} with K being the cardinality of 
        the categories.
    W : np.ndarray of shape [d x K]
        The embedding matrix, where the ith column of W represents the d-dimensional 
        vector embedding of the ith category.
    
    Returns
    -------
    X_emb : np.ndarray of shape [X.shape x d]
        The embedded data.
    
    Note
    ----
    Embedding can be accomplished by one-hot encoding the data and applying an 
    appropriate tensor contraction with W.  For example, if X is a 1-dimensional 
    integer array of size N, then X_emb = W X_ohe, where X_emb is of shape 
    [d x N] and X_ohe is of shape [K x N].  However, explicitly one-hot encoding the 
    data can be incredibly memory inefficient if the cardinality, K, is large.  
    Therefore, this function avoids one-hot encoding and utilizes the so-called  
    "look-up" approach instead.
    """
    # perform basic data checks
    msg = "the embed function only supports 1 or 2 dimensional data"
    assert (X.ndim == 1) or (X.ndim == 2), msg
    
    msg = "the embed function only supports int64 data"
    assert X.dtype == np.int64, msg
    
    if X.ndim == 2:
        return _embed_2d(X, W)
    return _embed_1d(X, W)

def _dW_embed(g, X, W):
    """
    Compute dL/dW for an embedding operation.
    
    Parameters
    ----------
    g : np.ndarray
        The gradient tensor of the loss with-respect-to X_emb (with shape X.shape x d).
    X : np.ndarray
        Categorical data to be embedded.  Must be a 1 or 2 dimensional array of ints.
        All integers should be in {0, 1, ..., K - 1} with K being the cardinality of the 
        categories.
    W : np.ndarray of shape [d x K]
        The embedding matrix, where the ith column of E represents the d-dimensional 
        vector embedding of the ith category.
    
    Returns
    -------
    dL/dW : np.ndarray of shape [X.shape x d] with dtype = W.dtype
        The gradient of the loss with-respect-to W.
        
    Note
    ----
    As with the embed function, this function avoids one-hot encoding the data by 
    computing the gradient associated with the so-called "look-up" approach.       
    """
    if X.ndim == 2:
        return _dW_embed_2d(g, X, W)
    return _dW_embed_1d(g, X, W)