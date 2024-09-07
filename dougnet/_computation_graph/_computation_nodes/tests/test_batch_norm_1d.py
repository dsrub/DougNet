import pytest
import dougnet as dn
from dougnet.training import DataLoader, ProgressHelper
from dougnet.data import LoadMNIST
from dougnet.metrics import accuracy

import numpy as np
import torch
import torch.nn as nn

ETA = .01
SEED_WEIGHTS = 1984
SEED_DATALOADER = 2
BATCH_SIZE = 100
EPOCHS = 10

def PrepareData(X, y, n_classes, dtype=np.float32, seed=42):
    # one hot encode Ys
    Y_ohe = np.zeros((y.size, n_classes))
    Y_ohe[np.arange(y.size), y] = 1
    
    # standardize
    X = ((X / 255.) - .5) * 2

    #transpose data 
    X = X.T
    Y_ohe = Y_ohe.T

    ## randomly shuffle columns (examples)
    random_perm_of_cols = np.random.RandomState(seed=seed).permutation(X.shape[1])
    X = X[:, random_perm_of_cols]
    Y_ohe = Y_ohe[:, random_perm_of_cols]
    
    return X.astype(dtype), Y_ohe.astype(dtype)

# load data and do basic prep for tests
X_train, Y_train, X_val, Y_val = LoadMNIST()
X_train, Y_train = PrepareData(X_train, Y_train, 10, dtype=np.float32)
X_val, Y_val = PrepareData(X_val, Y_val, 10, dtype=np.float32)

# de-OHE ys for pytorch
Y_train_deohe = np.argmax(Y_train, axis=0).astype(np.int64)
Y_val_deohe = np.argmax(Y_val, axis=0).astype(np.int64)


@pytest.mark.parametrize("eps", [1e-5, 1])
@pytest.mark.parametrize("parallel", [True, False])
def test_BN1d_1epoch(eps, parallel):
    # define 1-hidden layer dougnet model
    model = dn.ComputationGraph()
    X = dn.InputNode()    
    Y = dn.InputNode()
        
    Z1 = dn.Linear(X, 100, 28 * 28)
    Z_BN = dn.BN1d(Z1.module_output, 100, eps=eps, parallel=parallel)
    A1 = dn.Tanh(Z_BN)
    
    Yhat = dn.Linear(A1, 10, 100)
    L = dn.SoftmaxCrossEntropyLoss(Yhat.module_output, Y)

    # initialize weights
    model.initialize_params(SEED_WEIGHTS)
    optim = dn.optim.SGD(model, eta=ETA)

    # copy weights to initialize pytorch weights
    W1 = Z1.weight.output.copy()
    W2 = Yhat.weight.output.copy()

    # train dougnet model for 1 epoch on entire dataset
    X.output, Y.output = X_train, Y_train
    _ = L.forward()
    L.backward()
    optim.step()

    # perform inference on validation set
    model.eval()
    X.output = X_val
    yhat_val_dn = Yhat.module_output.forward()
    
    # define pytorch model
    model_pytorch = nn.Sequential(
        nn.Linear(28 * 28, 100),
        nn.BatchNorm1d(100, eps=eps),
        nn.Tanh(),
        nn.Linear(100, 10)
    )

    # initialize model weights with same weights as previous models
    with torch.no_grad():
        model_pytorch[0].weight = nn.Parameter(torch.tensor(W1))
        model_pytorch[0].bias = nn.Parameter(torch.zeros(100))
        
        model_pytorch[3].weight = nn.Parameter(torch.tensor(W2))
        model_pytorch[3].bias = nn.Parameter(torch.zeros(10))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_pytorch.parameters(), lr=ETA)

    # train model for 1 epoch on entire dataset
    X_tensor = torch.tensor(X_train.T)
    Y_tensor = torch.tensor(Y_train_deohe.reshape(-1), dtype=torch.long)
    yhat = model_pytorch(X_tensor)
    loss = loss_fn(yhat, Y_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()  

    # perform inference on validation set
    model_pytorch.eval()
    yhat_val_torch = model_pytorch(torch.tensor(X_val.T))
    
    # check linear parameters of model
    assert np.allclose(model_pytorch[0].weight.data.numpy(), Z1.weight.output, rtol=1e-6, atol=1e-06)
    assert np.allclose(model_pytorch[0].bias.data.numpy(), Z1.bias.output.reshape(-1), rtol=1e-6, atol=1e-06)
    assert np.allclose(model_pytorch[3].weight.data.numpy(), Yhat.weight.output, rtol=1e-6, atol=1e-06)
    assert np.allclose(model_pytorch[3].bias.data.numpy(), Yhat.bias.output.reshape(-1), rtol=1e-6, atol=1e-06)

    # check gamma and beta
    assert np.allclose(model_pytorch[1].weight.data.numpy(), Z_BN.gamma.output, rtol=1e-6, atol=1e-06)
    assert np.allclose(model_pytorch[1].bias.data.numpy(), Z_BN.beta.output, rtol=1e-6, atol=1e-06)

    # check running stats
    assert np.allclose(model_pytorch[1].running_mean.numpy(), Z_BN.running_mean, rtol=1e-5, atol=1e-05)
    assert np.allclose(model_pytorch[1].running_var.numpy(), Z_BN.running_var, rtol=1e-4, atol=1e-4)

    # check inference
    assert np.allclose(yhat_val_torch.detach().numpy().T, yhat_val_dn, rtol=1e-4, atol=1e-04)
    
    # check dtypes
    assert Z_BN.gamma.output.dtype == np.float32
    assert Z_BN.beta.output.dtype == np.float32
    assert Z_BN.running_mean.dtype == np.float32
    assert Z_BN.running_var.dtype == np.float32
    
    
@pytest.mark.parametrize("eps", [1e-5, 1])
@pytest.mark.parametrize("parallel", [True, False])
def test_BN1d_multiple_epochs(eps, parallel):
    # define 1-hidden layer model
    model = dn.ComputationGraph()

    X = dn.InputNode()
    Y = dn.InputNode()

    Z1 = dn.Linear(X, 100, 28 * 28)
    Z_BN = dn.BN1d(Z1.module_output, 100, eps=eps, parallel=parallel)
    A1 = dn.Tanh(Z_BN)

    Yhat = dn.Linear(A1, 10, 100)
    L = dn.SoftmaxCrossEntropyLoss(Yhat.module_output, Y)

    # initialize weights
    model.initialize_params(SEED_WEIGHTS)
    optim = dn.optim.SGD(model, eta=ETA)

    # copy weights to initialize pytorch weights
    W1 = Z1.weight.output.copy()
    W2 = Yhat.weight.output.copy()

    dataloader = DataLoader(X_train, Y_train, BATCH_SIZE, random_state=SEED_DATALOADER)
    progress = ProgressHelper(EPOCHS, X, Y, Yhat.module_output, L, progress_metric=accuracy, verbose=False)
    for epoch in range(EPOCHS):

        # perform mini batch updates to parameters
        for X_B, Y_B in dataloader.load():
            X.output, Y.output = X_B, Y_B
            
            # run forward and backward methods
            _ = L.forward()
            L.backward()
            optim.step()
        
        progress.update(X_train, Y_train, X_val, Y_val)

    # perform inference on validation set
    model.eval()
    X.output = X_val
    yhat_val_dn = Yhat.module_output.forward()
    
    # define pytorch model
    model_pytorch = nn.Sequential(
        nn.Linear(28 * 28, 100),
        nn.BatchNorm1d(100, eps=eps),
        nn.Tanh(),
        nn.Linear(100, 10)
    )

    # initialize model weights with same weights as previous models
    with torch.no_grad():
        model_pytorch[0].weight = nn.Parameter(torch.tensor(W1))
        model_pytorch[0].bias = nn.Parameter(torch.zeros(100))
        
        model_pytorch[3].weight = nn.Parameter(torch.tensor(W2))
        model_pytorch[3].bias = nn.Parameter(torch.zeros(10))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_pytorch.parameters(), lr=ETA)

    # train
    dataloader = DataLoader(X_train, Y_train_deohe, BATCH_SIZE, random_state=SEED_DATALOADER)
    for epoch in range(EPOCHS):
        for X_B, Y_B in dataloader.load():
            
            X_B_tensor = torch.tensor(X_B.T)
            Y_B_tensor = torch.tensor(Y_B.reshape(-1), dtype=torch.long)
                    
            yhat = model_pytorch(X_B_tensor)
            loss = loss_fn(yhat, Y_B_tensor)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
    # perform inference on validation set
    model_pytorch.eval()
    yhat_val_torch = model_pytorch(torch.tensor(X_val.T))
    
    # check linear parameters of model
    assert np.allclose(model_pytorch[0].weight.data.numpy(), Z1.weight.output, rtol=1e-5, atol=1e-5)
    assert np.allclose(model_pytorch[0].bias.data.numpy(), Z1.bias.output.reshape(-1), rtol=1e-5, atol=1e-5)
    assert np.allclose(model_pytorch[3].weight.data.numpy(), Yhat.weight.output, rtol=1e-5, atol=1e-5)
    assert np.allclose(model_pytorch[3].bias.data.numpy(), Yhat.bias.output.reshape(-1), rtol=1e-5, atol=1e-5)

    # check gamma and beta
    assert np.allclose(model_pytorch[1].weight.data.numpy(), Z_BN.gamma.output.reshape(-1), rtol=1e-5, atol=1e-05)
    assert np.allclose(model_pytorch[1].bias.data.numpy(), Z_BN.beta.output.reshape(-1), rtol=1e-5, atol=1e-05)

    # check running stats
    assert np.allclose(model_pytorch[1].running_mean.numpy(), Z_BN.running_mean.reshape(-1), rtol=1e-5, atol=1e-5)
    assert np.allclose(model_pytorch[1].running_var.numpy(), Z_BN.running_var.reshape(-1), rtol=1e-5, atol=1e-5)

    # check inference
    assert np.allclose(yhat_val_torch.detach().numpy().T, yhat_val_dn, rtol=1e-5, atol=1e-5)
    
    # check dtypes
    assert Z_BN.gamma.output.dtype == np.float32
    assert Z_BN.beta.output.dtype == np.float32
    assert Z_BN.running_mean.dtype == np.float32
    assert Z_BN.running_var.dtype == np.float32