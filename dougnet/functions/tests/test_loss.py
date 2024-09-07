import pytest
import numpy as np
import torch
import torch.nn.functional as F
from dougnet.functions import softmax_cross_entropy_loss


@pytest.mark.parametrize("n_classes", [10, 2], ids=["10", "2"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
def test_cross_entropy_loss(n_classes, dtype):
    
    n_classes = 10
    n_examples = 1000

    # compute loss with pytorch
    input = torch.randn(n_examples, n_classes)
    target = torch.randint(n_classes, (n_examples,), dtype=torch.int64)
    loss_torch = F.cross_entropy(input, target, reduction='mean')

    # compute loss with dougnet
    Z = input.numpy().T.astype(dtype)
    y = target.numpy()

    # one hot encode target 
    Y_ohe = np.zeros((y.size, n_classes), dtype=dtype)
    Y_ohe[np.arange(y.size),y] = 1
    Y_ohe = Y_ohe.T

    loss_dougnet = softmax_cross_entropy_loss(Z, Y_ohe)

    # check if correct
    assert np.allclose(loss_torch.numpy(), loss_dougnet)
    
    # check dtype
    assert loss_dougnet.dtype == dtype