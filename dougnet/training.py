import numpy as np
import sys
from datetime import datetime


def LoadMiniBatches(Xtrain, Ytrain, batch_size, seed=None):
    """Create mini-batches"""
        
    # make sure Ytrain is 2d
    if Ytrain.ndim == 1:
        Ytrain = Ytrain.reshape(1, -1)
    
    # intialize an rng if random_state is a seed
    if type(random_state) == int:
        random_state = np.random.RandomState(random_state)
            
    # randomly shuffle dataset
    random_perm_of_cols = np.arange(Xtrain.shape[1])
    random_state.shuffle(random_perm_of_cols)
    Xtrain = Xtrain[:, random_perm_of_cols]
    Ytrain = Ytrain[:, random_perm_of_cols]
    
    # iterate through mini batches
    for i in range(0, Xtrain.shape[1], batch_size):
        X_B = Xtrain[:, i:min(i + batch_size, Xtrain.shape[1])]
        Y_B = Ytrain[:, i:min(i + batch_size, Xtrain.shape[1])]
        
        yield X_B, Y_B
        

class ProgressHelper:
    def __init__(self, 
                 n_epochs, 
                 x_node, 
                 y_node, 
                 yhat_node, 
                 l_node, 
                 progress_metric,
                 verbose=True):
        self.n_epochs = n_epochs
        self.x_node = x_node
        self.y_node = y_node 
        self.yhat_node = yhat_node
        self.l_node = l_node 
        self.progress_metric = progress_metric
        self.verbose = verbose
        self.score_train_ = []
        self.loss_train_ = []
        self.score_val_ = []
        self.loss_val_ = []
        self.start_time = datetime.now()
        
    def _compute_progress(self, X_data, Y_data):
        """
        Helper function to compute train and validation loss and train and
        validation metric score.
        """
        # compute loss and score
        self.x_node.output, self.y_node.output = X_data, Y_data
        loss = self.l_node.forward()
        Yhat = self.yhat_node.output
        score = self.progress_metric(Yhat, Y_data)

        return score, loss
    
    def _verbose(self, train_score, train_loss, val_score, val_loss, epoch):
        """Print Progress to screen (code modified from Sebastian Raschka)"""
        # compute elapsed time
        elapsed_timedelta = datetime.now() - self.start_time
        elapsed_time = elapsed_timedelta.seconds + elapsed_timedelta.microseconds / 1e6

        # print to screen
        sys.stderr.write('\r%0*d/%d | Train/Val. Loss: %.2f/%.2f ' 
                        '| Train/Val. Score: %.2f%%/%.2f%% ' 
                        '| Elapsed Time: %.2f seconds' %
                        (len(str(self.n_epochs)), epoch + 1, self.n_epochs, train_loss, 
                        val_loss, train_score * 100, val_score * 100, elapsed_time)
                        )
        sys.stderr.flush()
    
    def update(self, epoch, Xtrain, Ytrain, Xval, Yval):
        train_score, train_loss = self._compute_progress(Xtrain, Ytrain)
        val_score, val_loss = self._compute_progress(Xval, Yval)
        
        # record progress
        self.score_train_.append(train_score)
        self.loss_train_.append(train_loss)
        self.score_val_.append(val_score)
        self.loss_val_.append(val_loss)
            
        if self.verbose:
            self._verbose(train_score, train_loss, val_score, val_loss, epoch)