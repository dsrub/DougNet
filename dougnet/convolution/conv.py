import numpy as np
from numba import njit, prange
import math


@njit(cache=True, parallel=True)
def _im2col(V_p, H_K, W_K, H_out, W_out, stride, dilate):
    """
    An im2col routine that uses multi-threading to parallelize over batches.  The 
    routine assumes a c-contiguous input of shape N x C_in x H x W.  
    
    The output is a matrix of size C_in * H_K * W_K x N * H_out * W_out, where H_K 
    and W_K are the kernel height and width respectively and H_out and W_out are the 
    height and width of the convolved output.  This corresponds to patch size x total 
    number of patches, where each "patch" is the 3-dimensional volume in the input 
    where each 3-dimensional kernel is evaluated.
    
    The format of the output of this function has all patches for each example image in 
    the columns sorted by patch, example ascending.  In other words, if c_pb is the column 
    vector associated with patch p and batch example b, and there are P total patches in 
    the input and N total examples in the batch, the output is the matrix:
    
    [   |     |         |     |     |           |          |     |          |  ]
    [   |     |         |     |     |           |          |     |          |  ] 
    [ c_11, c_21, ..., c_P1, c_12, c_22, ..., c_P2, ..., c_1N, c_2N, ..., c_PN ]
    [   |     |         |     |     |           |          |     |          |  ]
    [   |     |         |     |     |           |          |     |          |  ]
    """  
    # get shapes
    N, C_in, _, _ = V_p.shape

    # The following formulas are used for getting the correct column and row 
    # index in the loop:
    # col = exampleID * total_number_patches + current_patchID
    # row = index in current patch
    V_tilde_p = np.empty((C_in * H_K * W_K, N * H_out * W_out), dtype=V_p.dtype)
    for b in prange(N):
        for c in range(C_in):
            for i in range(H_out):
                for j in range(W_out):
                    col = b * H_out * W_out + i * W_out + j
                    for ii in range(H_K):
                        for jj in range(W_K):
                            row = c * H_K * W_K + ii * W_K + jj
                            V_tilde_p[row, col] = V_p[b, c, stride * i + dilate * ii, stride * j + dilate * jj]
    return V_tilde_p

def _conv2d_gemm(V, K, b, pad, stride, dilate):
    """
    Compute convolution by reshaping the input with im2col, matrix-multiplying 
    with a gemm optimized matrix-multiply routine, then reshaping back to the 
    output shape with col2im.  This function uses multi-threading to parallelize 
    over batches in im2col.
    """   
    # get shapes
    N, _, H, W = V.shape
    C_out, _, H_K, W_K = K.shape
    H_out = math.ceil((H + 2 * pad - dilate * (H_K - 1)) / stride)
    W_out = math.ceil((W + 2 * pad - dilate * (W_K - 1)) / stride)
    
    # convert to big matrix via im2col
    V_p = np.pad(V, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    V_tilde_p = _im2col(V_p, H_K, W_K, H_out, W_out, stride, dilate)
    
    # compute convolutions via a gemm
    K_tilde = K.reshape((C_out, -1))
    Z_tilde = K_tilde @ V_tilde_p + b.reshape(-1, 1)

    # reshape to desired output shape (effectively a col2im)
    Z = Z_tilde.reshape(C_out, N, H_out, W_out).transpose(1, 0, 2, 3)
    Z = np.ascontiguousarray(Z)
    
    return Z, V_tilde_p


@njit(cache=True, parallel=True)
def _conv2d_naive(V_p, K, bias, stride, dilate):
    """
    Compute convolution naively using only nested loops through the input and convolution
    kernels with the formula: [V * K]_{ij} = sum_{m,n} V_{i*s + m*d, j*s + n*d} * K_{mn}.
    This function uses multi-threading to parallelize over batches.
    """    
    # get shapes
    N, C_in, H_padded, W_padded = V_p.shape
    C_out, C_in, H_K, W_K = K.shape
    H_out = math.ceil((H_padded - dilate * (H_K - 1) ) / stride)
    W_out = math.ceil((W_padded - dilate * (W_K - 1) ) / stride)
    
    Z = np.zeros((N, C_out, H_out, W_out), dtype=V_p.dtype) + bias.reshape(C_out, 1, 1)
    for b in prange(N):
        for c_in in range(C_in):
            for c_out in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        for m in range(H_K):
                            for n in range(W_K):
                                Z[b, c_out, i, j] += V_p[b, c_in, i * stride + m * dilate, j * stride + n * dilate] * K[c_out, c_in, m, n]
    return Z


def conv2d(V, K, b, pad=0, stride=1, dilate=1, method="gemm", return_Vim2col=False):
    """
    Perform convolution on a multi-channel input, V (N x C_in x H x W), with 
    kernel tensor, K (C_out x C_in x H_K x W_K), and bias vector, b (C_in).  
    The method supports padding, stride and dilation and can be implemented 
    with a fast gemm based method using im2col, or a slow naive method using 
    only nested loops.  Note that this function technically implements 
    cross-correlation, even though this operation is commonly called  
    convolution in the ML community. 
    
    The output tensor :math:`\mathsf{Z}` has shape: 
    .. math::
        \mathsf{Z} \in \mathbf{R}^{N \times c_{out} \times \left \lceil 
        \frac{v_h+2p_h - d_h(k_h-1)}{s_h} \right \rceil \times \left \lceil 
        \frac{v_w+2p_w - d_w(k_w-1)}{s_w} \right \rceil },
    where
    `N` = number of examples in batch
    `C_in` = number of input channels
    `H` = image height
    `W` = image width
    `C_out` = number of output channels
    `H_K` = kernel height
    `W_K` = kernel width

    Parameters
    ----------
    V : np.ndarray of shape (N, C_in, H, W)
        Input tensor.  Should be c-contiguous for optimal performance.

    K : np.ndarray of shape (C_out, C_in, H_K, W_K)
        Kernel tensor.  Should be c-contiguous for optimal performance.

    b : np.ndarray of shape (C_out,)
        Bias vector.

    pad : int, default=0
        The amount of padding to add to each edge of the input. 

    stride : int, default=1
        The convolution stride.

    dilate : int, default=1
        The amount by which to dilate the kernel (dilate = 1 corresponds to
        no dilation).

    method : str, default="gemm"
        The convolution method.  One of "gemm" or "naive".
        
    return_Vim2col : bool, default=False
        Return Vim2col (= im2col(V_padded, s, d)) in addition to the convolved input, Z, 
        as a tuple: Z, Vim2col.  This keyword has no effect if method="naive".

    Returns
    -------
    Z : np.ndarray of shape (N, C_out, H_out, W_out)
        The convolved input in c-contiguous format.  H_out and W_out are 
        defined in the formula above.

    Notes
    -----
    Stride, pad and dilate should be chosen such that stride evenly divides
    H + 2 * pad  - d * (H_K - 1) and W + 2 * pad  - d * (W_K - 1).
    
    Author: Douglas Rubin
    """
    # check shapes
    message1 = "input must be rank-4 (N x C_in x H x W)"
    message2 = "kernel must be rank-4 (C_out x C_in x H_K x W_K)"
    assert V.ndim == 4, message1
    assert K.ndim == 4, message2
    
    # check that method is valid
    assert method in ["gemm", "naive"], 'method must be "gemm" or "naive"'
    
    # check that pad, stride and H/W are compatible 
    message1 = "stride should evenly divide H + 2 * pad  - d * (H_K - 1)"
    message2 = "stride should evenly divide W + 2 * pad  - d * (W_K - 1)"
    assert (V.shape[2] + 2 * pad - dilate * (K.shape[2] - 1)) % stride == 0, message1
    assert (V.shape[3] + 2 * pad - dilate * (K.shape[3] - 1) ) % stride == 0, message2
    
    # make sure bias vector is flattened
    b = b.reshape(-1)
    
    if method == "gemm":
        Z, Vim2col = _conv2d_gemm(V, K, b, pad, stride, dilate)
        if return_Vim2col:
            return Z, Vim2col
        return Z
    
    # numba doesn't support np.pad, so pad here then pass to _conv2d_naive
    V_padded = np.pad(V, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    return _conv2d_naive(V_padded, K, b, stride, dilate)


@njit(cache=True, parallel=True)
def _col2im(dV_tilde_p, N, C_in, H, W, H_K, W_K, H_out, W_out, pad, stride, dilate):
                            
    dV_p = np.zeros((N, C_in, H + 2 * pad, W + 2 * pad), dtype=dV_tilde_p.dtype)                   
    for b in prange(N):
        for c in range(C_in):
            for i in range(H_out):
                for j in range(W_out):
                    col = b * H_out * W_out + i * W_out + j
                    for ii in range(H_K):
                        for jj in range(W_K):
                            row = c * H_K * W_K + ii * W_K + jj
                            dV_p[b, c, stride * i + dilate * ii, stride * j + dilate * jj] += dV_tilde_p[row, col]
    return dV_p

                            
def _db_conv2d(dZ):
    """return dL/db for the conv2d operation"""
    
    # compute db
    db = np.einsum('bhij->h', dZ)
    return db


def _dK_conv2d(dZ, K, V_tilde_p):
    """return dL/dK for the conv2d operation"""
    
    # get shapes
    C_out, C_in, H_K, W_K = K.shape
    
    # compute dK
    dZ_tilde = dZ.transpose(1, 0, 2, 3).reshape(C_out, -1)
    dK_tilde = dZ_tilde @ V_tilde_p.T
    dK = dK_tilde.reshape(C_out, C_in, H_K, W_K)
    
    return dK

def _dV_conv2d(dZ, V, K, pad=0, stride=1, dilate=1):
    """return dL/dV for the conv2d operation"""
    
    # get shapes
    N, C_in, H, W = V.shape
    N, C_out, H_out, W_out = dZ.shape
    C_out, C_in, H_K, W_K = K.shape
    
    # compute dV_padded
    dZ_tilde = dZ.transpose(1, 0, 2, 3).reshape(C_out, -1)
    K_tilde = K.reshape(C_out, -1)
    dV_tilde_p = K_tilde.T @ dZ_tilde    
    dV_p = _col2im(dV_tilde_p, 
                   N, 
                   C_in, 
                   H, 
                   W, 
                   H_K, 
                   W_K, 
                   H_out, 
                   W_out, 
                   pad, 
                   stride, 
                   dilate
                  )   
    
    # extract unpadded portion  
    dV = dV_p
    if pad > 0:
        dV = np.ascontiguousarray(dV_p[:, :, pad:-pad, pad:-pad])
    return dV


@njit
def _dilate(K, d):
    """
    A function to dilate the input tensor, K, (with shape C_out x C_in x H_K x W_K) by d.
    """
    # get shapes
    C_out, C_in, H_K, W_K = K.shape

    dilated_K = np.zeros((C_out, C_in, d * (H_K - 1) + 1, d * (W_K - 1) + 1), dtype=K.dtype)
    for c_out in range(C_out):
        for c_in in range(C_in):
            for i in range(H_K):
                for j in range(W_K):
                    dilated_K[c_out, c_in, i * d, j * d] = K[c_out, c_in, i, j]
                    
    return dilated_K




