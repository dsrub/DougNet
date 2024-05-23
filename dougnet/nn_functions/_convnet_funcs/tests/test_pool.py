import pytest
import numpy as np
import torch
import torch.nn as nn
from dougnet.nn_functions._convnet_funcs._pool import (mp2d, 
                                                       _dZ_mp2d as _dZ,
                                                       gap2d,
                                                       _dZ_gap2d as _dZ_gap
                                                       )

# CREATE TESTING DATA
N, C, H, W = 1_000, 128, 31, 31

SEED = 1984
RANDOM_STATE = np.random.RandomState(SEED)
Z_NDARRAY = RANDOM_STATE.normal(0, 1, size=(N, C, H, W))


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("stride", [1, 2], ids=["s1", "s2"])
@pytest.mark.parametrize("kernel_size", [(2, 2), (4, 4)], ids=["k2", "k4"])
def test_MP2d(dtype, stride, kernel_size):
    """test forward and backward max pool function"""
    
    H_K, W_K = kernel_size
    H_out = (H - H_K + 1) // stride
    W_out = (W - W_K + 1) // stride

    # define output tensor
    torch.manual_seed(SEED)
    Z_out = torch.randn(N, C, H_out, W_out)

    # compute Z_MP and grads with pytorch
    mp = nn.MaxPool2d((H_K, W_K), stride=stride, padding=0)
    Z_torch = torch.tensor(Z_NDARRAY.astype(np.float32), requires_grad=True)

    Z_MP_torch = mp(Z_torch)
    Z_MP_torch.retain_grad()
    l = torch.sum((Z_MP_torch - Z_out) ** 2) / Z_out.numel()
    l.backward()

    dZ_MP = Z_MP_torch.grad
    dZ_torch = Z_torch.grad
    
    # compute Z_MP and grads with dougnet
    Z = Z_NDARRAY.astype(dtype)
    Z_MP, (I_max, J_max) = MP2d(Z, H_K, W_K, stride=stride, return_inds=True)
    dZ = _dZ(dZ_MP.numpy(), Z, I_max, J_max, H_K, W_K, stride=stride)
    
    # check correctness
    assert np.allclose(Z_MP_torch.detach().numpy(), Z_MP, rtol=1e-5, atol=1e-05)
    assert np.allclose(dZ_torch.numpy(), dZ, rtol=1e-5, atol=1e-05)
    
    # check dtype
    assert Z_MP.dtype == dtype
    assert dZ.dtype == dtype
    
    # check if row major
    assert Z_MP.flags['C_CONTIGUOUS']
    assert dZ.flags['C_CONTIGUOUS']
    
    
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
def test_GAP2d(dtype):
    """test forward and backward global average pool function"""
    
    # define output tensor
    torch.manual_seed(SEED)
    Z_out = torch.randn(H,)

    # compute M and grads with pytorch
    gap = nn.AdaptiveAvgPool2d((1, 1))
    Z_torch = torch.tensor(Z_NDARRAY.astype(np.float32), requires_grad=True)

    ZGAP_torch = gap(Z_torch)
    ZGAP_torch.retain_grad()
    l = torch.sum((ZGAP_torch - Z_out) ** 2) / Z_out.numel()
    l.backward()

    dZGAP_torch = ZGAP_torch.grad
    dZ_torch = Z_torch.grad
    
    # compute Z_GAP and grads with dougnet
    Z = Z_NDARRAY.astype(dtype)
    ZGAP = GAP2d(Z)
    dZ = _dZ_gap(dZGAP_torch.numpy().reshape(N, C), Z)
    
    # check correctness
    assert np.allclose(ZGAP_torch.detach().numpy().reshape(N, C), ZGAP, rtol=1e-5, atol=1e-05)
    assert np.allclose(dZ_torch.numpy(), dZ, rtol=1e-5, atol=1e-05)
    
    # check dtype
    assert ZGAP.dtype == dtype
    assert dZ.dtype == dtype
    
    # check if row major
    assert ZGAP.flags['C_CONTIGUOUS']
    assert dZ.flags['C_CONTIGUOUS']