import numpy as np

class SoftmaxCrossEntropyLoss:
    @staticmethod
    def func(Z, Y):
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
        return -np.sum(Z * Y) + np.sum(np.log(np.sum(np.exp(Z), axis=0)))
    
    
class L2RegLoss:
    @staticmethod
    def func(Ws):
        """ Compute the loss associated with L2 regularization of the weight matrices.

        Parameters
        -----------
        Ws : List of 2-D ndarrays.
            List of weight matrices for the nerual net.

        Returns:
        ----------
        L2 loss
        """
        return sum(np.sum(W ** 2) for W in Ws)

    def deriv(Ws, k):
        """ Compute d (L2RegLoss) / dW[k]

        Parameters
        -----------
        Ws : List of 2-D ndarrays.
            List of weight matrices for the nerual net.
        k : int
            The index of the weight matrix 

        Returns:
        ----------
        gradient of L2RegLoss wrt. W[k] (ndarray with same dims as W[k])
        """
        return 2 * Ws[k]


class L2Loss:
    @staticmethod
    def func(Z, Y):
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
        return .5 * np.sum((Z - Y)**2)/Y.shape[1]