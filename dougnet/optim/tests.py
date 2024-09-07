import pytest
import dougnet as dn
import numpy as np
import torch
import torch.nn as nn
from dougnet.data import LoadMNIST
from dougnet.training import DataLoader


# load mnist data for testing
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

X_TRAIN, Y_TRAIN, _, _ = LoadMNIST()
X_TRAIN, Y_TRAIN = PrepareData(X_TRAIN, Y_TRAIN, 10, dtype=np.float32)
Y_TRAIN_NO_OHE = np.argmax(Y_TRAIN, axis=0).astype(np.int64)

EPOCHS = 10
BATCH_SIZE = 100
SEED_WEIGHTS = 1984
SEED_DATALOADER = 2


def train_models(lmbda, 
                 dougnet_optim, 
                 dougnet_optim_kwargs, 
                 pytorch_optim, 
                 pytorch_optim_kwargs):
    """
    Helper function to train linear models on mnist with both dougnet and pytorch.
    """
    # define dougnet model
    model_dn = dn.ComputationGraph()

    X = dn.InputNode()
    Y = dn.InputNode()
    Z = dn.Linear(X, 10, 28 * 28)

    # loss node (weight decay bias too since by default pytorch decays the bias)
    L_data = dn.SoftmaxCrossEntropyLoss(Z.node, Y)
    L = L_data + dn.L2RegLoss(Z.weight, Z.bias, lmbda=lmbda)

    # initialize weights
    model_dn.initialize_params(SEED_WEIGHTS)
    weight = model_dn.parameters[0].output.copy()

    # train
    optim = dougnet_optim(model_dn, **dougnet_optim_kwargs)
    dataloader = DataLoader(X_TRAIN, Y_TRAIN, BATCH_SIZE, random_state=SEED_DATALOADER)
    for _ in range(EPOCHS):

        # perform mini batch updates to parameters
        for X_B, Y_B in dataloader.load():
            X.output, Y.output = X_B, Y_B

            # run forward and backward methods
            _ = L.forward()
            L.backward()

            # update parameters 
            optim.step()
        
    # define pytorch model
    model_pytorch = nn.Linear(28 * 28, 10)
    loss_fn = nn.CrossEntropyLoss()

    # initialize weights with same initial values as before
    with torch.no_grad():
        model_pytorch.weight = nn.Parameter(torch.tensor(weight))
        model_pytorch.bias = nn.Parameter(torch.zeros(10))

    # train
    dataloader = DataLoader(X_TRAIN, Y_TRAIN_NO_OHE, BATCH_SIZE, random_state=SEED_DATALOADER)
    optim = pytorch_optim(model_pytorch.parameters(), **pytorch_optim_kwargs)
    for _ in range(EPOCHS):
        for X_B, Y_B in dataloader.load():

            X_B_tensor = torch.tensor(X_B.T)
            Y_B_tensor = torch.tensor(Y_B.reshape(-1), dtype=torch.long)

            yhat = model_pytorch(X_B_tensor)
            loss = loss_fn(yhat, Y_B_tensor)
            loss.backward()

            optim.step()
            optim.zero_grad()
            
    return model_dn, model_pytorch

@pytest.mark.parametrize("eta", [.01])
@pytest.mark.parametrize("mom", [0, .1, .5, .9])
@pytest.mark.parametrize("lmbda", [0, .01, .1, .9])
def test_sgd(eta, mom, lmbda):
    """
    train a linear model using sgd on mnist for both dougnet and pytorch and compare
    """
    # train both models with specified optimizers
    model_dn, model_pytorch = train_models(lmbda, 
                                           dn.optim.SGD, 
                                           {"eta":eta, "momentum":mom},
                                           torch.optim.SGD, 
                                           {"lr":eta, "momentum":mom, "weight_decay":lmbda}
                                           )
    
    # check weight and bias values
    assert np.allclose(model_dn.parameters[0].output, 
                       model_pytorch.weight.data.numpy(), 
                       rtol=1e-4, 
                       atol=1e-4)
    assert np.allclose(model_dn.parameters[1].output.reshape(-1), 
                       model_pytorch.bias.data.numpy(), 
                       rtol=1e-4, 
                       atol=1e-4)
    
@pytest.mark.parametrize("eta", [.001])
@pytest.mark.parametrize("beta1", [.9, .8])
@pytest.mark.parametrize("beta2", [.999, .8])
@pytest.mark.parametrize("eps", [1e-8])
@pytest.mark.parametrize("lmbda", [0, .01, .9])
def test_adam(eta, beta1, beta2, eps, lmbda):
    """
    train a linear model using adam on mnist for both dougnet and pytorch and compare
    """    
    # train both models with specified optimizers
    model_dn, model_pytorch = train_models(lmbda, 
                                           dn.optim.Adam, 
                                           {"eta":eta, "betas":(beta1, beta2), "eps":eps},
                                           torch.optim.Adam, 
                                           {"lr":eta, 
                                            "betas":(beta1, beta2), 
                                            "eps":eps, 
                                            "weight_decay":lmbda}
                                           )
    
    # check weight and bias values
    assert np.allclose(model_dn.parameters[0].output, 
                       model_pytorch.weight.data.numpy(), 
                       rtol=1e-4, 
                       atol=1e-4)
    assert np.allclose(model_dn.parameters[1].output.reshape(-1), 
                       model_pytorch.bias.data.numpy(), 
                       rtol=1e-4, 
                       atol=1e-4)