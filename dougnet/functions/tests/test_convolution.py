import math
import pytest
import numpy as np
import torch
import torch.nn as nn
from dougnet.data import LoadCIFAR10
from dougnet.functions._convolution import (conv2d, 
                                            _db, 
                                            _dK, 
                                            _dV)


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


# LOAD CIFAR10 DATA FOR TESTING AND DO BASIC DATA PREP
X_TRAIN, Y_TRAIN, X_VAL, Y_VAL = LoadCIFAR10()
X_TRAIN, _ = PrepareData(X_TRAIN, Y_TRAIN, 10, dtype=np.float32)
X_TRAIN = X_TRAIN[:100, :, :, :] # subset to a smaller number of examples for quicker testing


# CREATE RANDOM TESTING DATA
SEED = 1984
RANDOM_STATE = np.random.RandomState(SEED)
N, H, W = 1_000, 30, 30
C_in, C_out = 64, 128
H_K = W_K = 3
V_NDARRAY = RANDOM_STATE.normal(0, 1, size=(N, C_in, H, W))
K_NDARRAY = RANDOM_STATE.normal(0, 1, size=(C_out, C_in, H_K, W_K))
B_NDARRAY = RANDOM_STATE.normal(0, 1, size=(C_out,))


@pytest.mark.parametrize("pad", [0, 1, 2], ids=["p0", "p1", "p2"])
@pytest.mark.parametrize("stride", [1, 2], ids=["s1", "s2"])
@pytest.mark.parametrize("dilate", [1, 2, 3], ids=["d1", "d2", "d3"])
@pytest.mark.parametrize("method", ["gemm", "naive"], ids=["g", "n"])
def test_conv2d_params_random(pad, stride, dilate, method):
    """test convolution on randomly generated data"""
    # cast to correct dtype
    V = V_NDARRAY.astype(np.float32)
    K = K_NDARRAY.astype(np.float32)
    b = B_NDARRAY.astype(np.float32)

    # create corresponding pytorch tensors
    V_tensor = torch.tensor(V)
    K_tensor = torch.tensor(K)
    b_tensor = torch.tensor(b)

    # compute convolution with both dougnet and pytorch
    Z_dn = conv2d(V, K, b, pad=pad, stride=stride, dilate=dilate, method=method)
    Z_torch = nn.functional.conv2d(V_tensor, 
                                K_tensor, 
                                b_tensor, 
                                stride=stride, 
                                padding=pad, 
                                dilation=dilate
                                ).numpy()

    assert np.allclose(Z_dn, Z_torch, rtol=1e-4, atol=1e-4)
    
    
@pytest.mark.parametrize("pad", [0, 1, 2], ids=["p0", "p1", "p2"])
@pytest.mark.parametrize("stride", [1, 2], ids=["s1", "s2"])
@pytest.mark.parametrize("dilate", [1, 2, 3], ids=["d1", "d2", "d3"])
@pytest.mark.parametrize("method", ["gemm", "naive"], ids=["g", "n"])
def test_conv2d_params_cifar10(pad, stride, dilate, method):
    """test convolution on cifar10 data"""
    # instantiate torch's Conv2d
    seed, output_channels = 1984, 64
    torch.manual_seed(seed)
    conv = nn.Conv2d(3, 
                     output_channels, 
                     H_K, 
                     stride=stride, 
                     padding=pad, 
                     dilation=dilate)

    # grab kernel and bias that pytorch has created
    K = conv.weight.data.numpy().copy().astype(np.float32)
    b = conv.bias.data.numpy().copy().astype(np.float32)

    # compute convolution with both dougnet and pytorch
    Z_dn = conv2d(X_TRAIN, K, b, pad=pad, stride=stride, dilate=dilate, method=method)
    Z_torch = conv(torch.tensor(X_TRAIN)).detach().numpy()

    assert np.allclose(Z_dn, Z_torch, rtol=1e-6, atol=1e-6)
    assert Z_dn.dtype == Z_torch.dtype

    
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_conv2d_dtype_bias_random(dtype, bias):
    """
    test convolution on randomly generated data with specified dtype and with 
    and without bias
    """
    # cast to correct dtype
    V = V_NDARRAY.astype(dtype)
    K = K_NDARRAY.astype(dtype)
    b = B_NDARRAY.astype(dtype)

    # create corresponding pytorch tensors
    V_tensor = torch.tensor(V.astype(np.float32))
    K_tensor = torch.tensor(K.astype(np.float32))
    b_tensor = torch.tensor(b.astype(np.float32))

    # compute convolution with both dougnet and pytorch
    if bias:
        Z_dn = conv2d(V, K, b)
        Z_torch = nn.functional.conv2d(V_tensor, K_tensor, b_tensor).numpy()
    else:
        Z_dn = conv2d(V, K)
        Z_torch = nn.functional.conv2d(V_tensor, K_tensor).numpy()

    assert np.allclose(Z_dn, Z_torch, rtol=1e-4, atol=1e-04)
    assert Z_dn.dtype == dtype
    

@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_conv2d_dtype_bias_cifar10(dtype, bias):
    """
    test convolution on cifar10 data with specified dtype and with and without 
    bias
    """
    # instantiate torch seed to create filter/bias
    seed, output_channels = 1984, 64
    torch.manual_seed(seed)
    
    # compute convolution with both dougnet and pytorch
    if bias:
        conv = nn.Conv2d(3, output_channels, H_K)

        # grab kernel and bias that pytorch has created
        K = conv.weight.data.numpy().copy().astype(dtype)
        b = conv.bias.data.numpy().copy().astype(dtype)

        # compute convolution with both dougnet and pytorch
        Z_dn = conv2d(X_TRAIN.astype(dtype), K, b)
        Z_torch = conv(torch.tensor(X_TRAIN)).detach().numpy()
    else:
        conv = nn.Conv2d(3, output_channels, H_K, bias=False)

        # grab kernel and bias that pytorch has created
        K = conv.weight.data.numpy().copy().astype(dtype)

        # compute convolution with both dougnet and pytorch
        Z_dn = conv2d(X_TRAIN.astype(dtype), K)
        Z_torch = conv(torch.tensor(X_TRAIN)).detach().numpy()

    assert np.allclose(Z_dn, Z_torch, rtol=1e-6, atol=1e-6)
    assert Z_dn.dtype == dtype
    
    
@pytest.mark.parametrize("pad", [0, 1, 2], ids=["p0", "p1", "p2"])
@pytest.mark.parametrize("stride", [1, 2], ids=["s1", "s2"])
@pytest.mark.parametrize("dilate", [1, 2, 3], ids=["d1", "d2", "d3"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_grads_random(pad, stride, dilate, bias):
    """test computation of gradients with randomly generated data"""
    H_out = math.ceil((H + 2 * pad - dilate * (H_K - 1)) / stride)
    W_out = math.ceil((W + 2 * pad - dilate * (W_K - 1)) / stride) 

    # define input and output tensors (the output tensor is to compute a loss
    # for testing the gradients)
    seed = 2
    torch.manual_seed(seed)
    V = torch.tensor(V_NDARRAY.astype(np.float32), requires_grad=True)    
    Z_out = torch.randn(N, C_out, H_out, W_out)

    # compute gradients with pytorch
    conv_torch = nn.Conv2d(C_in, 
                           C_out,
                           H_K,
                           stride=stride, 
                           padding=pad, 
                           dilation=dilate, 
                           bias=bias)
    K = conv_torch.weight.data
    if bias:
        b = conv_torch.bias.data

    Z = conv_torch(V)
    Z.retain_grad()
    l = torch.sum((Z - Z_out) ** 2) / Z_out.numel()
    l.backward()

    dZ = Z.grad
    dV_torch = V.grad
    dK_torch = conv_torch.weight.grad
    if bias:
        db_torch = conv_torch.bias.grad

    # compute gradients with dougnet
    if bias:    
        Z_dn, V_tilde_p = conv2d(V.detach().numpy(), 
                                 K.numpy(), 
                                 b.numpy(), 
                                 pad=pad, 
                                 stride=stride, 
                                 dilate=dilate, 
                                 return_Vim2col=True)
    else:
        Z_dn, V_tilde_p = conv2d(V.detach().numpy(), 
                                 K.numpy(), 
                                 pad=pad, 
                                 stride=stride, 
                                 dilate=dilate, 
                                 return_Vim2col=True)
    dK_dn = _dK(dZ.numpy(), K.numpy(), V_tilde_p)
    dV_dn = _dV(dZ.numpy(), 
                V.detach().numpy(), 
                K.numpy(), 
                pad=pad, 
                stride=stride, 
                dilate=dilate
            )
    if bias: 
        db_dn = _db(dZ.numpy())
    
    # compare
    assert np.allclose(Z_dn, Z.detach().numpy(), rtol=1e-5, atol=1e-05)
    assert np.allclose(dK_dn, dK_torch.numpy(), rtol=1e-5, atol=1e-05)
    assert np.allclose(dV_dn, dV_torch.numpy(), rtol=1e-5, atol=1e-05)
    if bias:
        assert np.allclose(db_dn, db_torch.numpy(), rtol=1e-5, atol=1e-05)
    
    
@pytest.mark.parametrize("pad", [0, 1, 2], ids=["p0", "p1", "p2"])
@pytest.mark.parametrize("stride", [1, 2], ids=["s1", "s2"])
@pytest.mark.parametrize("dilate", [1, 2, 3], ids=["d1", "d2", "d3"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_grads_cifar10(pad, stride, dilate, bias):
    """test computation of gradients with cifar10 data"""
    H_out = math.ceil((32 + 2 * pad - dilate * (H_K - 1)) / stride)
    W_out = math.ceil((32 + 2 * pad - dilate * (W_K - 1)) / stride) 
        
    # define input and output tensors (the output tensor is to compute a loss
    # for testing the gradients)
    seed, output_channels = 1984, 64
    torch.manual_seed(seed)
    V_torch = torch.tensor(X_TRAIN, requires_grad=True)
    Z_out = torch.randn(100, output_channels, H_out, W_out)

    # compute gradients with pytorch
    conv_torch = nn.Conv2d(3, 
                           output_channels,
                           H_K,
                           stride=stride, 
                           padding=pad, 
                           dilation=dilate, 
                           bias=bias)
    K = conv_torch.weight.data.numpy().copy()
    if bias:
        b = conv_torch.bias.data.numpy().copy()

    Z = conv_torch(V_torch)
    Z.retain_grad()
    l = torch.sum((Z - Z_out) ** 2) / Z_out.numel()
    l.backward()

    dZ = Z.grad
    dV_torch = V_torch.grad
    dK_torch = conv_torch.weight.grad
    if bias:
        db_torch = conv_torch.bias.grad

    # compute gradients with dougnet
    if bias:
        Z_dn, V_tilde_p = conv2d(X_TRAIN, 
                                 K, 
                                 b, 
                                 pad=pad, 
                                 stride=stride, 
                                 dilate=dilate, 
                                 return_Vim2col=True)
    else:
        Z_dn, V_tilde_p = conv2d(X_TRAIN, 
                                 K, 
                                 pad=pad, 
                                 stride=stride, 
                                 dilate=dilate, 
                                 return_Vim2col=True)
    dK_dn = _dK(dZ.numpy(), K, V_tilde_p)
    dV_dn = _dV(dZ.numpy(), 
                X_TRAIN, 
                K, 
                pad=pad, 
                stride=stride, 
                dilate=dilate
            )
    if bias:
        db_dn = _db(dZ.numpy())
    
    # compare
    assert np.allclose(Z_dn, Z.detach().numpy(), rtol=1e-6, atol=1e-06)
    assert np.allclose(dK_dn, dK_torch.numpy(), rtol=1e-6, atol=1e-06)
    assert np.allclose(dV_dn, dV_torch.numpy(), rtol=1e-6, atol=1e-06)
    if bias:
        assert np.allclose(db_dn, db_torch.numpy(), rtol=1e-6, atol=1e-06)