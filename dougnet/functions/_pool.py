from math import inf
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def _MP2d_njit(Z, H_K, W_K, stride):
    """
    Jitted function to compute a max pooling operation with multi-threading.
    """
    # get shapes
    N, C, H, W = Z.shape
    H_out = (H - H_K + 1) // stride
    W_out = (W - W_K + 1) // stride

    Z_MP = np.empty((N, C, H_out, W_out), dtype=Z.dtype)
    I_max = np.empty((N, C, H_out, W_out), dtype=np.int64)
    J_max = np.empty((N, C, H_out, W_out), dtype=np.int64)
    for b in prange(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    curr_max = -inf
                    for ii in range(H_K):
                        for jj in range(W_K):
                            z = Z[b, c, i * stride + ii, j * stride + jj]
                            if z > curr_max:
                                i_max = i * stride + ii
                                j_max = j * stride + jj
                                curr_max = z
                    Z_MP[b, c, i, j] = curr_max
                    I_max[b, c, i, j] = i_max
                    J_max[b, c, i, j] = j_max
                    
    return Z_MP, I_max, J_max


def mp2d(Z, H_K=3, W_K=3, stride=1, return_inds=False):
    """
    Perform a max pool operation on a rank-4 input (N x C x H x W).
    
    Parameters
    ----------
    Z : np.ndarray of shape (N, C, H, W)
        Input tensor.
        
    H_K : int, default=3
        Kernel height.
        
    W_K : int, default=3
        Kernel width.
        
    stride : int, default=1
        Stride.
    
    return_inds : bool, default=False
        In addition to returning the max pool of the input, return (I_max, J_max), 
        where both I_max and J_max have the same shape as the output of this function, 
        Z_MP.  I_max and J_max provide the spatial location in the input tensor where 
        the max associated with the (i,j)^th position of the output occurs.  In other 
        words, if Z_MP[b, c, i, j] = Z[b, c, i_max, j_max], then 
        I_max[b, c, i, j] = i_max and J_max[b, c, i, j] = j_max.  This information is
        useful for the backwards method of this function.

    Returns
    -------
    Z_MP : np.ndarray of shape (N, C, (H - H_K + 1) / stride, (W - W_K + 1) / stride)
        The max pool of the input
        
    Notes
    -----
    Stride must evenly divide H - H_K + 1 and W - W_K + 1.  Note also, that in the rare 
    case of a tie (i.e., for a given batch example and channel, a max pool region contains 
    multiple spatial locations with the same max value) I_max and J_max return the FIRST 
    spatial location found while iterating through the input image.  The iteration order is 
    given by looping over rows then columns.  This seems to mimic the pytorch behavior in 
    case of a tie.  
    """
    # check shape
    assert Z.ndim == 4, "input must be rank-4 (N x C x H x W)"
    
    # check that stride and H/W are compatible 
    message1 = "stride should evenly divide H - H_K + 1"
    message2 = "stride should evenly divide W - W_K + 1"
    assert (Z.shape[2] - H_K + 1) % stride == 0, message1
    assert (Z.shape[3] - W_K + 1) % stride == 0, message2
    
    # run jitted maxpool operation with multi-threading
    Z_MP, I_max, J_max = _MP2d_njit(Z, H_K, W_K, stride)
 
    if return_inds:  
        return Z_MP, (I_max, J_max)
    return Z_MP


@njit(parallel=True)
def _dZ_mp2d(dZ_MP, Z, I_max, J_max, H_K, W_K, stride=1):
    """
    Given the gradient of the loss wrt the output of the max pool operation, 
    return dL/dZ, the gradient of the loss wrt the input tensor.
    """
    # get shapes
    N, C, H, W = Z.shape
    H_out = (H - H_K + 1) // stride
    W_out = (W - W_K + 1) // stride

    # compute dZ with multi-threading
    dZ = np.zeros((N, C, H, W), dtype=Z.dtype)
    for b in prange(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    i_max = I_max[b, c, i, j]
                    j_max = J_max[b, c, i, j]
                    dZ[b, c, i_max, j_max] += dZ_MP[b, c, i, j]
    return dZ


def gap2d(Z):
    """
    Perform a global average pooling operation on a rank-4 input (N x C x H x W).
    
    Parameters
    ----------
    Z : np.ndarray of shape (N, C, H, W)
        Input tensor.
    
    Returns
    -------
    Z_GAP : np.ndarray of shape (N, C)
        The global average pool of the input.
    """
    # check shape
    assert Z.ndim == 4, "input must be rank-4 (N x C x H x W)"
    
    _, _, H, W = Z.shape
    
    # compute mean in float 64 to avoid overflow/underflow issues
    Z_GAP = np.ascontiguousarray((np.einsum('bhij->hb', Z.astype(np.float64)) / (H * W)
                                  ).astype(Z.dtype))
    return Z_GAP

def _dZ_gap2d(dZ_GAP, Z):
    """
    Given the gradient of the loss wrt the output of the global average pool operation, return 
    dL/dZ, the gradient of the loss wrt the input tensor.
    """
    # get shapes
    N, C, H, W = Z.shape
    
    # broadcast 1 / (H * W) * dL/dM to correct shape
    # do division in float64 to avoid potential underflow
    dZ = np.zeros_like(Z) + (dZ_GAP.T.reshape(N, C, 1, 1).astype(np.float64) / (H * W)).astype(Z.dtype)
    return dZ

