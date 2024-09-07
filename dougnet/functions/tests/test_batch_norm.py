import pytest
import numpy as np
import torch
import torch.nn as nn
from dougnet.functions import bn1d, bn2d
from dougnet.functions._batch_norm._batch_norm_main import _grads_bn

     
SEED = 1984
RANDOM_STATE = np.random.RandomState(SEED)                                                    

# CREATE 1d TESTING DATA
m, n = 100, 1_000
Z_NDARRAY = RANDOM_STATE.normal(0, 1, size=(m, n))
GAMMA_NDARRAY = RANDOM_STATE.normal(0, 1, size=(m,))
BETA_NDARRAY = RANDOM_STATE.normal(0, 1, size=(m,))

# CREATE 2d TESTING DATA
N, C, H, W = 100, 64, 28, 28
Z_NDARRAY_2d = RANDOM_STATE.normal(0, 1, size=(N, C, H, W))
GAMMA_NDARRAY_2d = RANDOM_STATE.normal(0, 1, size=(C,))
BETA_NDARRAY_2d = RANDOM_STATE.normal(0, 1, size=(C,))


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("eps", [1e-5, 1e-1])
@pytest.mark.parametrize("parallel", [True, False])
def test_bn1d(dtype, eps, parallel):
    
    # cast to correct type
    Z = Z_NDARRAY.astype(dtype)
    gamma = GAMMA_NDARRAY.astype(dtype)
    beta = BETA_NDARRAY.astype(dtype)

    # define output tensor
    torch.manual_seed(SEED)
    Z_out = torch.randn(n, m)

    # compute Z_BN and grads with pytorch
    BN_torch = nn.BatchNorm1d(m, eps=eps)
    BN_torch.bias = torch.nn.Parameter(torch.tensor(BETA_NDARRAY.astype(np.float32)))
    BN_torch.weight = torch.nn.Parameter(torch.tensor(GAMMA_NDARRAY.astype(np.float32)))
    Z_torch = torch.tensor(Z_NDARRAY.T.astype(np.float32), requires_grad=True)

    Z_BN_torch = BN_torch(Z_torch)
    Z_BN_torch.retain_grad()
    l = torch.sum((Z_BN_torch - Z_out) ** 2) / Z_out.numel()
    l.backward()

    dZ_BN = Z_BN_torch.grad
    dgamma_torch = BN_torch.weight.grad
    dbeta_torch = BN_torch.bias.grad
    dZ_torch = Z_torch.grad

    # compute Z_BN and gradswith dougnet
    Z_BN, (Z_prime, sigma, mu, var) = bn1d(Z, gamma, beta, eps=eps, return_cache=True, 
                                           parallel=parallel)
    dgamma, dbeta, dZ = _grads_bn(np.ascontiguousarray(dZ_BN.numpy().T.astype(dtype)), 
                                  Z_prime, gamma, sigma, parallel=parallel)
        
    # check correctness
    assert np.allclose(Z_BN_torch.detach().numpy().T, Z_BN, rtol=1e-5, atol=1e-05)
    assert np.allclose(dgamma_torch.numpy(), dgamma.reshape(-1), rtol=1e-5, atol=1e-05)
    assert np.allclose(dbeta_torch.numpy(), dbeta.reshape(-1), rtol=1e-5, atol=1e-05)
    assert np.allclose(dZ_torch.numpy().T, dZ, rtol=1e-5, atol=1e-05)

    # check dtype
    assert Z_BN.dtype == dtype
    assert dgamma.dtype == dtype
    assert dbeta.dtype == dtype
    assert dZ.dtype == dtype

    # check if row major
    assert Z_BN.flags['C_CONTIGUOUS']
    assert dZ.flags['C_CONTIGUOUS']
    
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("eps", [1e-5, 1e-1])
@pytest.mark.parametrize("parallel", [True, False])
def test_bn2d(dtype, eps, parallel):
                                                         
    # cast to correct type
    Z = Z_NDARRAY_2d.astype(dtype)
    gamma = GAMMA_NDARRAY_2d.astype(dtype)
    beta = BETA_NDARRAY_2d.astype(dtype)

    # define output tensor
    torch.manual_seed(SEED)
    Z_out = torch.randn(N, C, H, W)

    # compute Z_BN and grads with pytorch
    BN_torch = nn.BatchNorm2d(C, eps=eps)
    BN_torch.bias = torch.nn.Parameter(torch.tensor(BETA_NDARRAY_2d.astype(np.float32)))
    BN_torch.weight = torch.nn.Parameter(torch.tensor(GAMMA_NDARRAY_2d.astype(np.float32)))
    Z_torch = torch.tensor(Z_NDARRAY_2d.astype(np.float32), requires_grad=True)

    Z_BN_torch = BN_torch(Z_torch)
    Z_BN_torch.retain_grad()
    l = torch.sum((Z_BN_torch - Z_out) ** 2) / Z_out.numel()
    l.backward()

    dZ_BN = Z_BN_torch.grad
    dgamma_torch = BN_torch.weight.grad
    dbeta_torch = BN_torch.bias.grad
    dZ_torch = Z_torch.grad

    # compute Z_BN and grads with dougnet
    Z_BN, (Z_prime, sigma, mu, var) = bn2d(Z, gamma, beta, eps=eps, return_cache=True, 
                                           parallel=parallel)
    dgamma, dbeta, dZ = _grads_bn(dZ_BN.numpy().astype(dtype), Z_prime, gamma, sigma, 
                                  parallel=parallel)

    # check correctness
    assert np.allclose(Z_BN_torch.detach().numpy(), Z_BN, rtol=1e-4, atol=1e-04)
    assert np.allclose(dgamma_torch.numpy(), dgamma.reshape(-1), rtol=1e-5, atol=1e-05)
    assert np.allclose(dbeta_torch.numpy(), dbeta.reshape(-1), rtol=1e-5, atol=1e-05)
    assert np.allclose(dZ_torch.numpy(), dZ, rtol=1e-5, atol=1e-05)
    
    # check dtype
    assert Z_BN.dtype == dtype
    assert dgamma.dtype == dtype
    assert dbeta.dtype == dtype
    assert dZ.dtype == dtype

    # check if row major
    assert Z_BN.flags['C_CONTIGUOUS']
    assert dZ.flags['C_CONTIGUOUS']