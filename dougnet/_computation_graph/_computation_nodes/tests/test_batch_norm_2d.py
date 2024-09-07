from datetime import datetime
import pytest
import numpy as np
import torch
import torch.nn as nn
import dougnet as dn
from dougnet.training import DataLoader
from dougnet.data import LoadCIFAR10


# DEFINE HELPER FUNCS
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
    
    return X.astype(dtype), Y_ohe.astype(dtype)

def _BN2d_forward_helper(eps, x_train, x_train_torch, gamma, beta, gamma_torch, beta_torch, parallel):
    # compute dougnet BN2d
    model = dn.ComputationGraph()
    X = dn.InputNode()
    Z_BN = dn.BN2d(X, x_train.shape[1], eps=eps, parallel=parallel)
    Z_BN.beta.output = beta
    Z_BN.gamma.output = gamma
    X.output = x_train
    Z_BN_dn = Z_BN.forward()

    # compute torch BN2d
    BN_torch = torch.nn.BatchNorm2d(x_train.shape[1], eps=eps)
    BN_torch.bias = torch.nn.Parameter(beta_torch)
    BN_torch.weight = torch.nn.Parameter(gamma_torch)
    BN_torch.train()
    Z_BN_torch = BN_torch(x_train_torch)
    
    return Z_BN_dn, Z_BN_torch, Z_BN, BN_torch


# LOAD CIFAR10 DATA FOR TESTING AND DO BASIC DATA PREP
X_TRAIN, Y_TRAIN, X_VAL, Y_VAL = LoadCIFAR10()
X_TRAIN, Y_TRAIN = PrepareData(X_TRAIN, Y_TRAIN, 10, dtype=np.float32)
X_VAL, Y_VAL = PrepareData(X_VAL, Y_VAL, 10, dtype=np.float32)
X_TRAIN_TORCH = torch.tensor(X_TRAIN)
X_VAL_TORCH = torch.tensor(X_VAL)

# DE-OHE TARGETS FOR PYTORCH
Y_TRAIN_DEOHE = np.argmax(Y_TRAIN, axis=0).astype(np.int64)
Y_VAL_DEOHE = np.argmax(Y_VAL, axis=0).astype(np.int64)

# INSTANTIATE BATCHNOR "WEIGHT" AND "BIAS"
SEED = 1984
RANDOM_STATE = np.random.RandomState(SEED) 
GAMMA = RANDOM_STATE.normal(0, 1, size=(X_TRAIN.shape[1],)).astype(np.float32)
BETA = RANDOM_STATE.normal(0, 1, size=(X_TRAIN.shape[1],)).astype(np.float32)
GAMMA_TORCH = torch.tensor(GAMMA)
BETA_TORCH = torch.tensor(BETA)


@pytest.mark.parametrize("eps", [1e-5, 1e-2])
@pytest.mark.parametrize("parallel", [True, False])
def test_BN2d_forward_normal(eps, parallel):
    """test forward functionality BN2d with normally distributed data"""
    seed = 2024
    random_state = np.random.RandomState(seed) 
    mu, var = 10, 2

    # create normally distributed testing data
    N, C, H, W = 1_000, 64, 28, 28
    x_train = random_state.normal(mu, var, size=(N, C, H, W)).astype(np.float32)
    x_train_torch = torch.tensor(x_train)
        
    # instantiate gamma and beta
    gamma = random_state.normal(0, 1, size=(x_train.shape[1],)).astype(np.float32)
    beta = random_state.normal(0, 1, size=(x_train.shape[1],)).astype(np.float32)
    gamma_torch = torch.tensor(gamma)
    beta_torch = torch.tensor(beta)
    
    # run test
    Z_BN_dn, Z_BN_torch, Z_BN, BN_torch = _BN2d_forward_helper(eps, x_train, x_train_torch, gamma, beta, gamma_torch, beta_torch, parallel)
    assert np.allclose(Z_BN_dn, Z_BN_torch.detach().numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(Z_BN.running_mean, BN_torch.running_mean.numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(Z_BN.running_var, BN_torch.running_var.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("eps", [1e-5, 1e-2])
@pytest.mark.parametrize("parallel", [True, False])
def test_BN2d_forward_CIFAR10(eps, parallel):
    """test forward functionality of BN2d with CIFAR10 data"""
    Z_BN_dn, Z_BN_torch, Z_BN, BN_torch = _BN2d_forward_helper(eps, X_TRAIN, X_TRAIN_TORCH, GAMMA, BETA, GAMMA_TORCH, BETA_TORCH, parallel)
    assert np.allclose(Z_BN_dn, Z_BN_torch.detach().numpy(), rtol=1e-2, atol=1e-2)
    assert np.allclose(Z_BN.running_mean, BN_torch.running_mean.numpy(), rtol=1e-4, atol=1e-4)
    assert np.allclose(Z_BN.running_var, BN_torch.running_var.numpy(), rtol=1e-4, atol=1e-4)
    

@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("parallel", [True, False])
def test_BN2d_train_eval(eps, parallel):
    """
    test forward/backward functionality of BN2d over multiple epochs (i.e., training capability)
    by defining a simple linear model BN2d -> flatten -> linear -> softmax with CIFAR10 data.
    Additionally, test the eval (inference) functionality of BN2d using the trained model on the
    CIFAR10 validation set.
    """
    percent_same_val_preds_thresh = .99
    seed_weights = 1984
    seed_dataloader = 2
    epochs = 10
    eta =.01
    batch_size = 1_000

    # define dn model
    model = dn.ComputationGraph()
    X, Y = dn.InputNode(), dn.InputNode() 
    Z_BN = dn.BN2d(X, X_TRAIN.shape[1], eps=eps, parallel=parallel)
    A = dn.Flatten(dn.Transpose(Z_BN, (1, 2, 3, 0)), 3)
    Yhat = dn.Linear(A, 10, 32 * 32 * 3)
    L = dn.SoftmaxCrossEntropyLoss(Yhat.module_output, Y)

    # train dn model
    t0 = datetime.now()
    model.initialize_params(seed_weights)
    W = Yhat.weight.output.copy()
    dataloader = DataLoader(X_TRAIN, Y_TRAIN, batch_size, random_state=seed_dataloader)
    optim = dn.optim.Adam(model, eta=eta)
    for epoch in range(epochs):
        for X_B, Y_B in dataloader.load():
            X.output, Y.output = X_B, Y_B
            _ = L.forward()
            L.backward()
            optim.step()
    print("dougnet training time = ", (datetime.now() - t0).seconds + (datetime.now() - t0).microseconds * 1e-6)

    # perform inference on validation set with dn model
    t0 = datetime.now()
    model.eval()
    X.output = X_VAL
    yhat_val_dn = Yhat.module_output.forward()
    print("dougnet inference time = ", (datetime.now() - t0).seconds + (datetime.now() - t0).microseconds * 1e-6)

    # define pytorch model
    model_pytorch = nn.Sequential(
        nn.BatchNorm2d(X_TRAIN.shape[1], eps=eps),
        nn.Flatten(start_dim=1),
        nn.Linear(32 * 32 * 3, 10)
    )

    # initialize model weights with same weights as previous models
    with torch.no_grad():
        model_pytorch[2].weight = nn.Parameter(torch.tensor(W))
        model_pytorch[2].bias = nn.Parameter(torch.zeros(10))

    # train pytorch model
    t0 = datetime.now()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_pytorch.parameters(), lr=eta)
    dataloader = DataLoader(X_TRAIN, Y_TRAIN_DEOHE, batch_size, random_state=seed_dataloader)
    for epoch in range(epochs):
        for X_B, Y_B in dataloader.load():
            X_B_tensor = torch.tensor(X_B)
            Y_B_tensor = torch.tensor(Y_B.reshape(-1), dtype=torch.long)
            yhat = model_pytorch(X_B_tensor)
            loss = loss_fn(yhat, Y_B_tensor)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() 
    print("pytorch training time = ", (datetime.now() - t0).seconds + (datetime.now() - t0).microseconds * 1e-6)

    # perform inference on validation set with pytorch model
    t0 = datetime.now()
    model_pytorch.eval()
    yhat_val_torch = model_pytorch(X_VAL_TORCH)
    print("pytorch inference time = ", (datetime.now() - t0).seconds + (datetime.now() - t0).microseconds * 1e-6)

    # check gamma and beta
    assert np.allclose(model_pytorch[0].weight.data.numpy(), Z_BN.gamma.output.reshape(-1), rtol=1e-4, atol=1e-4)
    assert np.allclose(model_pytorch[0].bias.data.numpy(), Z_BN.beta.output.reshape(-1), rtol=1e-4, atol=1e-4)

    # check running means and running vars
    assert np.allclose(model_pytorch[0].running_mean.numpy(), Z_BN.running_mean.reshape(-1), rtol=1e-4, atol=1e-4)
    assert np.allclose(model_pytorch[0].running_var.numpy(), Z_BN.running_var.reshape(-1), rtol=1e-4, atol=1e-4)

    # check linear params
    assert np.allclose(model_pytorch[2].weight.data.numpy(), Yhat.weight.output, rtol=1e-4, atol=1e-4)
    assert np.allclose(model_pytorch[2].bias.data.numpy(), Yhat.bias.output.reshape(-1), rtol=1e-4, atol=1e-4)

    # check inference preds 
    dn_preds = np.argmax(yhat_val_dn, axis=0)
    torch_preds = np.argmax(yhat_val_torch.detach().numpy().T, axis=0)
    assert np.sum(torch_preds == dn_preds) / yhat_val_dn.shape[1] > percent_same_val_preds_thresh