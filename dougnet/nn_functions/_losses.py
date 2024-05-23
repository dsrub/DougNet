import numpy as np


def softmax_cross_entropy_loss(Z, Y):
    """ Computes the loss with a Softmax output layer and Cross entropy with
    trick to avoid overflow.

    Parameters
    -----------
    Z : ndarray, shape = [n_classes, n_examples]
        Net Input matrix of final layer.
    Y : ndarray, shape = [n_classes, n_examples]
        One-hot encoded target matrix corresponding to Z. 

    Returns:
    ----------
    Cross entropy loss
    """
    Z = Z - np.max(Z, axis=0)
    loss = (-np.sum(Z * Y) + np.sum(np.log(np.sum(np.exp(Z), axis=0)))) / Y.shape[1]
    
    # python forces the division to be float64
    loss = loss.astype(Z.dtype)

    return loss


def l2regloss(*Ws, lmbda=.1):
    """ Compute the loss associated with L2 regularization of the weight matrices.

    Parameters
    -----------
    Ws : List or tuple of weight ndarrays.
        List of weights for the nerual net.

    Returns:
    ----------
    L2 loss
    """
    return lmbda * sum(np.sum(W ** 2) for W in Ws) / 2


def l2loss(Z, Y):
    """ Computes the loss with an identity function output layer and squared loss.

    Parameters
    -----------
    Z : ndarray, shape = [n_targets, n_examples]
        Net Input matrix of final layer.
    Y : ndarray, shape = [n_targets, n_examples]
        Target matrix corresponding to Z. 


    Returns:
    ----------
    squared loss 
    """
    loss = .5 * np.sum((Z - Y) ** 2) / Y.shape[1]
    loss = loss.astype(Z.dtype)
    
    return loss