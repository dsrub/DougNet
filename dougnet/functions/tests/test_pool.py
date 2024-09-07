import pytest
import numpy as np
import torch
import torch.nn as nn
from dougnet.data import LoadCIFAR10
from dougnet.functions._pool import (mp2d, 
                                        _dZ_mp2d as _dZ,
                                        gap2d,
                                        _dZ_gap2d as _dZ_gap
                                        )


# DEFINE HELPER FUNC FOR LOADING CIFAR10 DATA
def PrepareData(X, y, n_classes, dtype=np.float32, seed=42):
    # one hot encode Ys
    Y_ohe = np.zeros((y.size, n_classes))
    Y_ohe[np.arange(y.size), y] = 1
    Y_ohe = Y_ohe.T
    
    # standardize
    X = ((X / 255.) - .5) * 2
    
    # get in N x C x H x W form
    X = X.transpose(0, 3, 1, 2)

    ## randomly shuffle images
    random_perm = np.random.RandomState(seed=seed).permutation(X.shape[0])
    X = X[random_perm, :, :, :]
    Y_ohe = Y_ohe[:, random_perm]
    
    return np.ascontiguousarray(X.astype(dtype)), np.ascontiguousarray(Y_ohe.astype(dtype))

# LOAD CIFAR10 DATA AND DO BASIC DATA PREP
X_TRAIN, Y_TRAIN, X_VAL, Y_VAL = LoadCIFAR10()
X_TRAIN, Y_TRAIN = PrepareData(X_TRAIN, Y_TRAIN, 10, dtype=np.float32)
X_TRAIN_TORCH = torch.tensor(X_TRAIN)
Y_TRAIN_DEOHE = np.argmax(Y_TRAIN, axis=0).astype(np.int64)

# CREATE RANDOM TESTING DATA
N, C, H, W = 1_000, 128, 31, 31
SEED = 1984
RANDOM_STATE = np.random.RandomState(SEED)
Z_NDARRAY = RANDOM_STATE.normal(0, 1, size=(N, C, H, W))


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("stride", [1, 2], ids=["s1", "s2"])
@pytest.mark.parametrize("kernel_size", [(2, 2), (4, 4)], ids=["k2", "k4"])
def test_MP2d_randomdata(dtype, stride, kernel_size):
    """test forward and backward max pool function with randomly generated data"""
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
    Z_MP, (I_max, J_max) = mp2d(Z, H_K, W_K, stride=stride, return_inds=True)
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
@pytest.mark.parametrize("stride", [1, 2], ids=["s1", "s2"])
@pytest.mark.parametrize("kernel_size", [(3, 3), (5, 5)], ids=["k2", "k4"])  
def test_MP2d_CIFAR10(dtype, stride, kernel_size):
    """test forward and backward max pool function on CIFAR10 data"""
    H_K, W_K = kernel_size
    H_out = (32 - H_K + 1) // stride
    W_out = (32 - W_K + 1) // stride

    # DEFINE SIMPLE MAXPOOL->FLATTEN->LINEAR MODEL IN PYTORCH
    mp = nn.MaxPool2d((H_K, W_K), stride=stride, padding=0)
    flatten = nn.Flatten()
    linear = nn.Linear(H_out * W_out * 3, 10)
    loss = nn.CrossEntropyLoss()

    # COMPUTE FORWARD PASS IN PYTORCH
    Z_torch = torch.tensor(X_TRAIN.astype(np.float32), requires_grad=True)
    Z_MP_torch = mp(Z_torch)
    Z_MP_torch.retain_grad()
    X = flatten(Z_MP_torch)
    Yhat = linear(X)

    # COMPUTE BACKWARD PASS IN PYTORCH
    l = loss(Yhat, torch.tensor(Y_TRAIN_DEOHE, dtype=torch.long))
    l.backward()

    # GET PYTORCH GRADS
    dZ_MP_torch = Z_MP_torch.grad
    dZ_torch = Z_torch.grad

    # COMPUTE GRADS IN DOUGNET
    Z = X_TRAIN.astype(dtype)
    Z_MP, (I_max, J_max) = mp2d(Z, H_K, W_K, stride=stride, return_inds=True)
    dZ = _dZ(dZ_MP_torch.numpy().astype(dtype), Z, I_max, J_max, H_K, W_K, stride=stride)

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
def test_GAP2d_randomdata(dtype):
    """test forward and backward global average pool function with randomly generated data"""
    # define output tensor
    torch.manual_seed(SEED)
    Z_out = torch.randn(C,)

    # compute M and grads with pytorch
    gap = nn.AdaptiveAvgPool2d((1, 1))
    Z_torch = torch.tensor(Z_NDARRAY.astype(np.float32), requires_grad=True)

    ZGAP_torch = torch.squeeze(gap(Z_torch))
    ZGAP_torch.retain_grad()
    l = torch.sum((ZGAP_torch - Z_out) ** 2) / Z_out.numel()
    l.backward()

    dZGAP_torch = ZGAP_torch.grad
    dZ_torch = Z_torch.grad

    # compute Z_GAP and grads with dougnet
    Z = Z_NDARRAY.astype(dtype)
    ZGAP = gap2d(Z)
    dZ = _dZ_gap(np.ascontiguousarray(dZGAP_torch.numpy().astype(dtype).T), Z)

    # check correctness
    assert np.allclose(ZGAP_torch.detach().numpy().T, ZGAP, rtol=1e-5, atol=1e-05)
    assert np.allclose(dZ_torch.numpy(), dZ, rtol=1e-5, atol=1e-05)

    # check dtype
    assert ZGAP.dtype == dtype
    assert dZ.dtype == dtype

    # check if row major
    assert ZGAP.flags['C_CONTIGUOUS']
    assert dZ.flags['C_CONTIGUOUS']

    
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
def test_GAP2d_CIFAR10(dtype):
    """test forward and backward max pool function on CIFAR10 data"""
    # DEFINE SIMPLE MAXPOOL->LINEAR MODEL IN PYTORCH
    gap = nn.AdaptiveAvgPool2d((1, 1))
    linear = nn.Linear(3, 10)
    loss = nn.CrossEntropyLoss()

    # COMPUTE FORWARD PASS IN PYTORCH
    Z_torch = torch.tensor(X_TRAIN.astype(np.float32), requires_grad=True)
    Z_GAP_torch = torch.squeeze(gap(Z_torch))
    Z_GAP_torch.retain_grad()
    Yhat = linear(Z_GAP_torch)

    # COMPUTE BACKWARD PASS IN PYTORCH
    l = loss(Yhat, torch.tensor(Y_TRAIN_DEOHE, dtype=torch.long))
    l.backward()

    # GET PYTORCH GRADS
    dZ_GAP_torch = Z_GAP_torch.grad
    dZ_torch = Z_torch.grad

    # COMPUTE GRADS IN DOUGNET
    Z = X_TRAIN.astype(dtype)
    Z_GAP = gap2d(Z)
    dZ = _dZ_gap(np.ascontiguousarray(dZ_GAP_torch.numpy().astype(dtype).T), Z)

    # check correctness
    assert np.allclose(Z_GAP_torch.detach().numpy().astype(dtype).T, Z_GAP, rtol=1e-5, atol=1e-05)
    assert np.allclose(dZ_torch.numpy().astype(dtype), dZ, rtol=1e-5, atol=1e-05)

    # check dtype
    assert Z_GAP.dtype == dtype
    assert dZ.dtype == dtype

    # check if row major
    assert Z_GAP.flags['C_CONTIGUOUS']
    assert dZ.flags['C_CONTIGUOUS']