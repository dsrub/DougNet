import numpy as np
from dougnet.training import *
from dougnet.functions import (tanh, 
                               sigmoid, 
                               relu,
                               l2regloss)

# register derivatives for activations
activation_prime = {}
activation_prime[tanh] = lambda z: 1 - tanh(z) ** 2
activation_prime[sigmoid] = lambda z: sigmoid(z) * (1 - sigmoid(z))
activation_prime[relu] = lambda z: (z > 0).astype(int).astype(z.dtype)

# register derivatives for regularization functions
reg_prime = {}
reg_prime[None] = lambda W: 0
reg_prime[l2regloss] = lambda W: W


class ProgressHelperMLP:
    """Helper class to record and print progress during training of MLP."""
    def __init__(self, mlp):
        self.mlp = mlp
        self._i = 0
        if self.mlp.verbose:
            self.pbar = tqdm(total=self.mlp.n_epochs, 
                             desc="epoch", 
                             unit="epoch", 
                             bar_format='{l_bar}{bar:15}{r_bar}{bar:-10b}')
        
    def _compute_progress(self, X_data, Y_data):
        """Compute train/validation loss and train/validation metric score"""
        # compute loss and score
        self.mlp.predict(X_data)
        loss = self.mlp.loss(self.mlp.Z[-1], Y_data)
        score = self.mlp.progress_metric(self.mlp.A[-1], Y_data)
        return score, loss
    
    def update(self, Xtrain, Ytrain, Xval, Yval):
        self._i +=1
        
        train_score, train_loss = self._compute_progress(Xtrain, Ytrain)
        val_score, val_loss = self._compute_progress(Xval, Yval)  
        
        # update progress lists
        self.mlp.score_train_.append(train_score)
        self.mlp.loss_train_.append(train_loss)
        self.mlp.score_val_.append(val_score)
        self.mlp.loss_val_.append(val_loss)
        
        # print progress to screen
        if self.mlp.verbose:
            self.pbar.set_postfix(loss=str(round(train_loss, 2)) + "/" + str(round(val_loss, 2)), 
                                  score=str(round(train_score, 2)) + "/" + str(round(val_score, 2)))
            self.pbar.update(1)
            if self._i == self.mlp.n_epochs:
                self.pbar.close()


class MultiLayerPerceptron:
    """ 
    A multi-layer perceptron for classification (softmax cross entropy loss) or 
    regression (l2 loss) trained with sgd.

    Parameters
    ------------
    num_units : List of ints 
        Number of units in each hidden layer and output layer.
    activation_funcs : List of dougnet activations
        Activation functions for the hidden layers and output layer.
    reg_loss : dougnet regularization loss function (default: None)
        Loss associated with regularization.  If None, no regularization is used.
    lmbda : float (default: 0.1)
        Lambda value for L2-regularization. Has no effect if reg_loss = None.
    eta : float (default: 0.01)
        Learning rate.
    batch_size : int (default: 100)
        Mini-batch size.
    n_epochs : int (default: 100)
        Number of eopchs. 
    weight_seed : int (default: None)
        Seed for initializing weights.  If none, no seed is used.
    data_seed : int (default: None)
        Seed for shuffling mini-batches.  If none, no seed is used.
    progress_metric : dougnet metric function (default: None) 
        Metric (e.g., accuracy) used to track progress.
    loss : dougnet loss function (default: None) 
        Loss (e.g., softmax cross entropy) used to track progress.  progress_metric 
        and loss should either both be None or both be not None.  Additionally, if 
        they are both not None, then a validation dataset should also be passed to the 
        fit() method to track the validation proress (in addition to the training 
        dataset).  Progress is saved in the attributes: loss_train_, loss_val_, 
        score_train_ and score_val_.  
    verbose : bool (default: False) 
        Print progress during training.
        
    Attributes
    -----------
    loss_train_ : List
        Training loss after each epoch.
    loss_val_ : List
        Validation loss after each epoch.
    score_train_ : List
        Training metric score after each epoch.
    score_val_ : List
        Validation metric score after each epoch.
        
    Methods
    -------
    fit :
        Fit the MLP with mini-batch SGD.
    predict :
        Return predictions from supplied design matrix.

    Notes
    -------
    Note that a loss function for training is not actually required since both softmax 
    cross entropy loss for classificaion and l2 loss for regression result in the same 
    update equations.
    """
    def __init__(self, 
                 num_units, 
                 activations, 
                 reg_loss=None, 
                 lmbda=.1, 
                 eta=.01, 
                 batch_size=100, 
                 n_epochs=100, 
                 weight_seed=None,
                 data_seed=None, 
                 progress_metric=None, 
                 loss=None,
                 verbose=False
                 ):
        self.num_units = num_units
        self.loss = loss
        self.reg_loss = reg_loss
        self.batch_size = batch_size
        self.eta = eta
        self.n_epochs = n_epochs
        self.lmbda = lmbda
        self.weight_seed = weight_seed
        self.data_seed = data_seed
        self.progress_metric = progress_metric
        self.verbose = verbose
        self.L = len(activations)
        
        # define list of activation functions and derivatives
        self.g = [None] + [activation for activation in activations]
        self.gprime = [None] + [activation_prime[activation] for activation in activations[:-1]] + [None] 
        self.reg_loss_prime = reg_prime[self.reg_loss]
        if progress_metric:
            self.loss_train_ = []
            self.loss_val_ = []
            self.score_train_ = []
            self.score_val_ = []
            
        msg = "progress_metric and loss should both be None, or both not be None"
        cond1 = (progress_metric is None) and (loss is None)
        cond2 = (progress_metric is not None) and (loss is not None)
        assert cond1 or cond2, msg

    def _forward(self, X):
        """
        Compute the forward pass to populate all net inputs and all activations.  
        Returns matrix of target predictions for training batch.
        """
        self.A[0] = X
        for k in range(1, self.L + 1):
            self.Z[k] = self.W[k] @ self.A[k - 1] + self.b[k]
            self.A[k] = self.g[k](self.Z[k])
        
        Y_hat = self.A[-1]  
        return Y_hat
    
    def _backward(self, Y, Y_hat):
        """
        Compute backward pass to compute gradients for all model parameters.
        """
        Delta = Y_hat - Y
        self.gradW[-1] = Delta @ self.A[-2].T / Y.shape[1] + self.lmbda * self.reg_loss_prime(self.W[-1])
        self.gradb[-1] = (Delta/ Y.shape[1]).sum(axis=1).reshape(Delta.shape[0], 1) 
        for k in range(self.L)[:0:-1]:
            Delta = self.gprime[k](self.Z[k]) * (self.W[k + 1].T @ Delta)
            self.gradW[k] = Delta @ self.A[k - 1].T / Y.shape[1] + self.lmbda * self.reg_loss_prime(self.W[k])
            self.gradb[k] = (Delta/ Y.shape[1]).sum(axis=1).reshape(Delta.shape[0], 1) 

    def fit(self, Xtrain, Ytrain, Xval=None, Yval=None):
        """ 
        Use mini-batch SGD to learn weights from training data.

        Parameters
        -----------
        Xtrain : np.ndarray [n_features x n_examples_train]
            Training design matrix.
        Ytrain : np.ndarray [n_classes x n_examples_train]
            Training targets.
        Xval : np.ndarray [n_features x n_examples_val]
            Validation design matrix used to compute loss/score progress during training.  
        Yval : np.ndarray [n_calsses x n_examples_val]
            Validation targets.
        """
        
        msg = "If progress_metric is not None, then validation data should be supplied. \
            Otherwise no validation data should be supplied."
        cond1 = (self.progress_metric is None) and (Xval is None) and (Yval is None)
        cond2 = (self.progress_metric is not None) and (Xval is not None) and (Yval is not None)
        assert cond1 or cond2, msg
        
        self.W = [None] * (self.L + 1)
        self.b = [None] * (self.L + 1)

        self.Z = [None] * (self.L + 1)
        self.A = [None] * (self.L + 1)

        self.gradW = [None]*(self.L + 1)
        self.gradb = [None]*(self.L + 1)
        
        # add input number of features to num_units
        num_units = [Xtrain.shape[0]] + self.num_units
    
        # initialize all model parameters
        random_state = np.random.RandomState(self.weight_seed)
        for k in range(1, self.L + 1):
            self.W[k] = random_state.normal(0, 1, (num_units[k], num_units[k - 1])
                                            ).astype(Xtrain.dtype)
            self.b[k] = np.zeros((1, num_units[k])).astype(Xtrain.dtype).T
        
        # loop over epochs
        progress = ProgressHelperMLP(self)
        dataloader = DataLoader(Xtrain, Ytrain, self.batch_size, random_state=self.data_seed)
        for epoch in range(self.n_epochs):
            for X_B, Y_B in dataloader.load():
                Y_hat = self._forward(X_B)
                self._backward(Y_B, Y_hat)
                
                # update parameters 
                for k in range(1, self.L + 1):
                    self.W[k] -= self.eta * self.gradW[k]
                    self.b[k] -= self.eta * self.gradb[k]
            
            if self.progress_metric:
                progress.update(Xtrain, Ytrain, Xval, Yval)
    
    def predict(self, Xpred):
        """
        Predict target.

        Parameters
        -----------
        Xpred : np.ndarray [n_features x n_examples_prediction]
            Design matrix corresponding to desired predictions.
            
        Returns:
        ----------
        Y_hat : np.ndarray [n_classes x n_examples_prediction]
            Predictions.
        """
        return self._forward(Xpred)