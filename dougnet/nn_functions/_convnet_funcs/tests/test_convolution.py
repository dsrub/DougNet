import math
import pytest
import numpy as np
import torch
import torch.nn as nn
from dougnet.nn_functions._convnet_funcs._convolution import (conv2d, 
                                                              _db, 
                                                              _dK, 
                                                              _dV
                                                              )

# CREATE TESTING DATA
N, H, W = 1_000, 30, 30
C_in, C_out = 64, 128
H_K = W_K = 3

SEED = 1984
RANDOM_STATE = np.random.RandomState(SEED)

V_NDARRAY = RANDOM_STATE.normal(0, 1, size=(N, C_in, H, W))
K_NDARRAY = RANDOM_STATE.normal(0, 1, size=(C_out, C_in, H_K, W_K))
B_NDARRAY = RANDOM_STATE.normal(0, 1, size=(C_out,))


@pytest.mark.parametrize("pad", [0, 1, 2], ids=["p0", "p1", "p2"])
@pytest.mark.parametrize("stride", [1, 2], ids=["s1", "s2"])
@pytest.mark.parametrize("dilate", [1, 2, 3], ids=["d1", "d2", "d3"])
@pytest.mark.parametrize("method", ["gemm", "naive"], ids=["g", "n"])
def test_conv2d_params(pad, stride, dilate, method):
    # cast to correct dtype
    V = V_NDARRAY.astype(np.float32)
    K = K_NDARRAY.astype(np.float32)
    b = B_NDARRAY.astype(np.float32)
    
    # create corresponding pytorch tensors
    V_tensor = torch.tensor(V)
    K_tensor = torch.tensor(K)
    b_tensor = torch.tensor(b)
    
    Z = conv2d(V, K, b, pad=pad, stride=stride, dilate=dilate, method=method)
    Z_torch = nn.functional.conv2d(V_tensor, 
                                   K_tensor, 
                                   b_tensor, 
                                   stride=stride, 
                                   padding=pad, 
                                   dilation=dilate
                                   ).numpy()

    assert np.allclose(Z, Z_torch, rtol=1e-4, atol=1e-04)
    
    
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_conv2d_dtype(dtype):
    # cast to correct dtype
    V = V_NDARRAY.astype(dtype)
    K = K_NDARRAY.astype(dtype)
    b = B_NDARRAY.astype(dtype)
    
    # create corresponding pytorch tensors
    V_tensor = torch.tensor(V)
    K_tensor = torch.tensor(K)
    b_tensor = torch.tensor(b)
    
    Z = conv2d(V, K, b)
    Z_torch = nn.functional.conv2d(V_tensor, K_tensor, b_tensor).numpy()

    assert np.allclose(Z, Z_torch, rtol=1e-4, atol=1e-04)
    
    
@pytest.mark.parametrize("pad", [0, 1, 2], ids=["p0", "p1", "p2"])
@pytest.mark.parametrize("stride", [1, 2], ids=["s1", "s2"])
@pytest.mark.parametrize("dilate", [1, 2, 3], ids=["d1", "d2", "d3"])
def test_grads(pad, stride, dilate):
    
    H_out = math.ceil((H + 2 * pad - dilate * (H_K - 1)) / stride)
    W_out = math.ceil((W + 2 * pad - dilate * (W_K - 1)) / stride) 
    
    # define input and output tensors (the output tensor is to compute a loss
    # for testing the gradients)
    torch.manual_seed(SEED)
    V = torch.tensor(V_NDARRAY.astype(np.float32), requires_grad=True)    
    Z_out = torch.randn(N, C_out, H_out, W_out)
    
    # compute gradients with pytorch
    conv_torch = nn.Conv2d(C_in, 
                           C_out,
                           H_K,
                           stride=stride, 
                           padding=pad, 
                           dilation=dilate)
    K = conv_torch.weight.data
    b = conv_torch.bias.data

    Z = conv_torch(V)
    Z.retain_grad()
    l = torch.sum((Z - Z_out) ** 2) / Z_out.numel()
    l.backward()

    dZ = Z.grad
    dV_torch = V.grad
    dK_torch = conv_torch.weight.grad
    db_torch = conv_torch.bias.grad
    
    # compute gradients with dougnet
    Z_dn, V_tilde_p = conv2d(V.detach().numpy(), 
                             K.numpy(), 
                             b.numpy(), 
                             pad=pad, 
                             stride=stride, 
                             dilate=dilate, 
                             return_Vim2col=True
                             )
    db_dn = _db(dZ.numpy())
    dK_dn = _dK(dZ.numpy(), K.numpy(), V_tilde_p)
    dV_dn = _dV(dZ.numpy(), 
                V.detach().numpy(), 
                K.numpy(), 
                pad=pad, 
                stride=stride, 
                dilate=dilate
            )
    
    # compare
    assert np.allclose(Z_dn, Z.detach().numpy(), rtol=1e-5, atol=1e-05)
    assert np.allclose(dK_dn, dK_torch.numpy(), rtol=1e-5, atol=1e-05)
    assert np.allclose(dV_dn, dV_torch.numpy(), rtol=1e-5, atol=1e-05)
    assert np.allclose(db_dn, db_torch.numpy(), rtol=1e-5, atol=1e-05)