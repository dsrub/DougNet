import numpy as np
from datetime import datetime
from dougnet.training_utils import *

class MultLayerPerceptron:
    """ Multi-layer perceptron.

    Parameters
    ------------
    num_units : List of ints 
        number of units in each layer (including the input and output layers)
    activation_funcs : List of dougnet activations
        Activation functions for the hidden layers and output layer.
    loss : dougnet loss class
        Loss to be used for training (note that a loss class includes the 
        activation in the final layer.  This means that that the final activation 
        function in activation_funcs must match the activation function associated 
        with the supplied loss).
    reg : regularization dougnet loss class (default: None, indicating no 
          regularization)
        Loss associated with regularization.  As of now, only L2 regularization is 
        implemented
    B : int (default: 100)
        mini-batch size
    eta : float (default: 0.01)
        Learning rate.
    n_epochs : int (default: 100)
        Number of eopchs. 
    lmbda : float (default: 0.1)
        Lambda value for L2-regularization.lmbda = 0 corresponds to no 
        regularization
    seed : int (default: None)
        seed for shuffling data during SGD and parameter initialization. 
        If none, no seed is used.
    progress_metric : dougnet metric function (default: None, indicating to not 
                      track progress on a validation set) 
        If set, records the training and validation loss and provided metric score 
        for each epoch in loss_train_, loss_val_, score_train_ and score_val_.  As 
        of now, only Accuracy, RMSE and R^2 metrics are implemented.
    verbose : bool (default: True)
        if set, print progress during training.

    Attributes
    -----------
    loss_train_ : List
        Training loss after each epoch
    loss_val_ : List
        Validation loss after each epoch
    score_train_ : List
        Training metric score after each epoch
    score_val_ : List
        Validation metric score after each epoch
        
    Methods
    -----------
    fit(Xtrain, Ytrain, Xval, Yval)
        Fit the MLP with mini-batch SGD and return fitted model.
    predict(Xpred)
        Predict and return targets from supplied design matrix.

    Author: Douglas Rubin

    """
    def __init__(self, num_units, activations, loss, reg=None, B=100, eta = .01, 
        n_epochs=100, lmbda=.1, seed=None, progress_metric=None, verbose=True):
    
        self.g = [None] + [activation.func for activation in activations]
        self.gprime = [None] + [activation.deriv for activation in \
                                activations[:-1]] + [None]
        if not reg: 
            self.reg_loss = lambda W: 0
            self.reg_loss_deriv = lambda W, k: 0
        else: 
            self.reg_loss = reg.func
            self.reg_loss_deriv = reg.deriv
        self.total_loss = lambda ZZ, YY: loss.func(ZZ, YY) \
                                         + lmbda * self.reg_loss(self.W[1:])
        self.L = len(activations)
        self.num_units = num_units
        self.B = B
        self.eta = eta
        self.n_epochs = n_epochs
        self.lmbda = lmbda
        self.random = np.random.RandomState(seed)
        self.progress_metric = progress_metric
        self.verbose = verbose
        if progress_metric:
            self.epoch_str_len = len(str(self.n_epochs))
            self.loss_train_ = []
            self.loss_val_ = []
            self.score_train_ = []
            self.score_val_ = []
            

    def _Forward(self, X):
        """Compute the forward pass to populate all net inputs and all activations.  
        Returns matrix of target predictions for training batch."""
        
        self.A[0] = X
        for k in range(1, self.L + 1):
            self.Z[k] = self.W[k] @ self.A[k - 1] + self.b[k]
            self.A[k] = self.g[k](self.Z[k])
        
        Y_hat = self.A[-1]  
        return Y_hat

    
    def _Backward(self, Y, Y_hat):
        """Compute backwatd pass to compute gradients for all model parameters"""
        
        Delta = Y_hat - Y
        self.gradW[-1] = Delta @ self.A[-2].T \
                        + self.lmbda * self.reg_loss_deriv(self.W, -1)
        self.gradb[-1] = Delta.sum(axis=1).reshape(Delta.shape[0], 1)
        for k in range(self.L)[:0:-1]:
            Delta = self.gprime[k](self.Z[k]) * (self.W[k + 1].T @ Delta)
            self.gradW[k] = Delta @ self.A[k - 1].T \
                            + self.lmbda * self.reg_loss_deriv(self.W, k)
            self.gradb[k] = Delta.sum(axis=1).reshape(Delta.shape[0], 1)


    def fit(self, Xtrain, Ytrain, Xval, Yval):
        """ Use mini-batch SGD to learn weights from training data.

        Parameters
        -----------
        Xtrain : array, shape = [n_features, n_examples]
            Training design matrix
        Ytrain : array, shape = [n_classes, n_examples]
            Training targets
        Xval : array, shape = [n_features, n_examples]
            Validation design matrix used to compute loss/score during training.
        Yval : array, shape = [n_classes, n_examples]
            Validation targets

        Returns:
        ----------
        self

        """
        self.start_time = datetime.now()
        self.W = [None]*(self.L + 1)
        self.b = [None]*(self.L + 1)

        self.Z = [None]*(self.L + 1)
        self.A = [None]*(self.L + 1)

        self.gradW = [None]*(self.L + 1)
        self.gradb = [None]*(self.L + 1)
    
        # initialize all model parameters
        for k in range(1, self.L + 1):
            self.W[k] = self.random.normal(0, 1, (self.num_units[k], 
                                           self.num_units[k - 1]))
            self.b[k] = np.zeros((1, self.num_units[k])).T
        
        # loop over epochs
        for epoch in range(self.n_epochs):
            
            # perform mini batch updates to parameters
            for X_B, Y_B in GetMiniBatches(self, Xtrain, Ytrain):
                Y_hat = self._Forward(X_B)
                self._Backward(Y_B, Y_hat)
                
                # update parameters 
                for k in range(1, self.L + 1):
                    self.W[k] -= (self.eta/self.B) * self.gradW[k]
                    self.b[k] -= (self.eta/self.B) * self.gradb[k]

            if self.progress_metric or self.verbose:
                ProgressHelper(self, Xtrain, Ytrain, Xval, Yval, epoch)

        return self
    
    
    def predict(self, Xpred, return_net_input=False):
        """Predict target 

        Parameters
        -----------
        Xpred : array, shape = [n_features, n_examples]
            Design matrix corresponding to desired predictions.
        return_net_input : bool (default: False)
            if set, returns both the Y_hat matrix as well as Z, the net input 
            matrix of the final layer. 

        Returns:
        ----------
        Y_hat : array, shape = [n_classes, n_examples]
            Prediction targets for each example
        Z : array, shape = [n_classes, n_examples]
            The net input for each example.  Only returned if return_net_input 
            set.
        
        """
        Y_hat = self._Forward(Xpred)
        Z = self.Z[-1]
        
        if return_net_input:
            return Y_hat, Z
        else:
            return Y_hat