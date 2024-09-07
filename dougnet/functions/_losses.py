import numpy as np


def softmax_cross_entropy_loss(Z, Y):
    """
    Compute the mean softmax cross-entropy loss between the predicted logits, Z 
    [N_classes x N_examples], and the actual one-hot-encoded targets, Y 
    [N_classes x N_examples].  Uses log-sum-exp trick  to avoid overflow.
    """
    Z = Z - np.max(Z, axis=0)
    loss = -np.sum((Z * Y) / Y.shape[1]) + np.sum(np.log(np.sum(np.exp(Z), axis=0)) / Y.shape[1]) 
    loss = loss.astype(Z.dtype)
    return loss

def l2regloss(*Ws, lmbda=.1):
    """Compute the L2 loss on the provided weight tensors."""
    return lmbda * sum(np.sum(W ** 2) for W in Ws) / 2

def l2loss(Z, Y):
    """
    Computes the mean L2 loss between Z [N_classes x N_examples] and Y 
    [N_classes x N_examples].
    """
    loss = .5 * np.sum((Z - Y) ** 2 / Y.shape[1]).astype(Z.dtype)
    return loss