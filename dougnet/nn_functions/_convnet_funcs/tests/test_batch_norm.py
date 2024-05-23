import pytest
import numpy as np
import torch
import torch.nn as nn
from dougnet.nn_functions._convnet_funcs._batch_norm import (bn2d,
                                                             _dbeta,
                                                             _dgamma,
                                                             _dZ
                                                             )


# CREATE TESTING DATA
N, C, H, W = 1_000, 128, 30, 30

SEED = 1984
RANDOM_STATE = np.random.RandomState(SEED)
Z_NDARRAY = RANDOM_STATE.normal(0, 1, size=(N, C, H, W))
GAMMA_NDARRAY = RANDOM_STATE.normal(0, 1, size=(C, ))
BETA_NDARRAY = RANDOM_STATE.normal(0, 1, size=(C, ))


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
def test_BN2d(dtype):
    
    # cast to correct type
    Z = Z_NDARRAY.astype(dtype)
    gamma = GAMMA_NDARRAY.astype(dtype)
    beta = BETA_NDARRAY.astype(dtype)
    
    # define output tensor
    torch.manual_seed(SEED)
    Z_out = torch.randn(N, C, H, W)
    
    # compute Z_BN and grads with pytorch
    BN_torch = nn.BatchNorm2d(C)
    BN_torch.bias = torch.nn.Parameter(torch.tensor(BETA_NDARRAY.astype(np.float32) ))
    BN_torch.weight = torch.nn.Parameter(torch.tensor(GAMMA_NDARRAY.astype(np.float32)))
    Z_torch = torch.tensor(Z_NDARRAY.astype(np.float32), requires_grad=True)

    Z_BN_torch = BN_torch(Z_torch)
    Z_BN_torch.retain_grad()
    l = torch.sum((Z_BN_torch - Z_out) ** 2) / Z_out.numel()
    l.backward()

    dZ_BN = Z_BN_torch.grad
    dgamma_torch = BN_torch.weight.grad
    dbeta_torch = BN_torch.bias.grad
    dZ_torch = Z_torch.grad
    
    # compute Z_BN with dougnet
    Z_BN, (Z_prime, gamma_tilde, sigma_tilde) = BN2d(Z, gamma, beta, return_cache=True)

    # compute grads with dougnet
    dgamma = _dgamma(dZ_BN.numpy().astype(dtype), Z_prime)
    dbeta = _dbeta(dZ_BN.numpy().astype(dtype))
    dZ = _dZ(dZ_BN.numpy().astype(dtype), gamma_tilde, sigma_tilde)
    
    # check correctness
    assert np.allclose(Z_BN_torch.detach().numpy(), Z_BN, rtol=1e-5, atol=1e-05)
    assert np.allclose(dgamma_torch.numpy(), dgamma, rtol=1e-5, atol=1e-05)
    assert np.allclose(dbeta_torch.numpy(), dbeta, rtol=1e-5, atol=1e-05)
    assert np.allclose(dZ_torch.numpy(), dZ, rtol=1e-5, atol=1e-05)
    
    # check dtype
    assert Z_BN.dtype == dtype
    assert dgamma.dtype == dtype
    assert dbeta.dtype == dtype
    assert dZ.dtype == dtype
    
    # check if row major
    assert Z_BN.flags['C_CONTIGUOUS']
    assert dZ.flags['C_CONTIGUOUS']