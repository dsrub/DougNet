import numpy as np
from tqdm import tqdm

        
class DataLoader:
    """
    Mini-batch data loader class.
    
    Parameters
    ------------
    Xtrain : np.ndarray
        Training features (can be 2-d or 4-d for image data).
    Ytrain : np.ndarray
        Training targets (can be 1-d or 2-d).
    batch_size : int
        Mini-batch size.
    random_state: None, int or np.random.RandomState, (default=None)
        If None, instantiate an rng, if int, instantiate and rng with
        a seed, else use the provided rng.
    """
    def __init__(self, Xtrain, Ytrain, batch_size, random_state=None):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.batch_size = batch_size
        self.random_state = random_state
        if (type(self.random_state) == int) or self.random_state is None:
            self.random_state = np.random.RandomState(self.random_state)
        
        # make sure Ytrain is 2d
        if self.Ytrain.ndim == 1:
            self.Ytrain = self.Ytrain.reshape(1, -1)    
        
    def load(self):
        """load a mini-batch"""
        if self.Xtrain.ndim == 2:
            # randomly shuffle dataset
            random_perm_of_cols = np.arange(self.Xtrain.shape[1])
            self.random_state.shuffle(random_perm_of_cols)
            self.Xtrain = self.Xtrain[:, random_perm_of_cols]
            self.Ytrain = self.Ytrain[:, random_perm_of_cols]
        
            # iterate through mini batches
            for i in range(0, self.Xtrain.shape[1], self.batch_size):
                X_B = self.Xtrain[:, i:min(i + self.batch_size, self.Xtrain.shape[1])]
                Y_B = self.Ytrain[:, i:min(i + self.batch_size, self.Xtrain.shape[1])]
                yield X_B, Y_B 
        
        elif self.Xtrain.ndim == 4:
            # randomly shuffle dataset
            random_perm_of_cols = np.arange(self.Xtrain.shape[0])
            self.random_state.shuffle(random_perm_of_cols)
            self.Xtrain = self.Xtrain[random_perm_of_cols, :, :, :]
            self.Ytrain = self.Ytrain[:, random_perm_of_cols]
        
            # iterate through mini batches
            for i in range(0, self.Xtrain.shape[0], self.batch_size):
                X_B = self.Xtrain[i:min(i + self.batch_size, self.Xtrain.shape[0]), :, :, :]
                Y_B = self.Ytrain[:, i:min(i + self.batch_size, self.Xtrain.shape[0])]
                yield X_B, Y_B

        
class ProgressHelper:
    """
    Track progress during training.
    
    Parameters
    ------------
    n_epochs : int
        Number of epochs.
    x_node : InputNode
        Features node.
    y_node : InputNode
        Targets node.
    yhat_node: ComputationNode
        Prediction node.
    l_node: ComputationNode
        Loss node.
    progress_metric: dougnet metric function
        Track this metric during training (e.g. accuracy).
    verbose: bool (default=True)
        Print progress during training.
    """
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
        self._eval_time = 0
        self._i = 0
        self.score_train_ = []
        self.loss_train_ = []
        self.score_val_ = []
        self.loss_val_ = []
        if verbose:
            self.pbar = tqdm(total=n_epochs, 
                             desc="epoch", 
                             unit="epoch", 
                             bar_format='{l_bar}{bar:15}{r_bar}{bar:-10b}'
                             )
            
    def _compute_progress(self, X_data, Y_data):
        """Helper function to compute train/val loss and train/val metric."""
        # compute loss and score
        self.x_node.output, self.y_node.output = X_data, Y_data
        loss = self.l_node.forward()
        Yhat = self.yhat_node.output
        score = self.progress_metric(Yhat, Y_data)
        return score, loss
    
    def update(self, Xtrain, Ytrain, Xval, Yval):
        """Method to update progress.  Usually called after each epoch."""
        self._i +=1
        
        # make sure in eval mode before evaluating performance
        old_mode = self.x_node.graph.eval_mode
        self.x_node.graph.eval()
        
        train_score, train_loss = self._compute_progress(Xtrain, Ytrain)
        val_score, val_loss = self._compute_progress(Xval, Yval)
        
        # set back to old mode
        self.x_node.graph.eval_mode = old_mode
        
        # record progress
        self.score_train_.append(train_score)
        self.loss_train_.append(train_loss)
        self.score_val_.append(val_score)
        self.loss_val_.append(val_loss)
                    
        if self.verbose:
            # print progress to screen
            self.pbar.set_postfix(loss=str(round(train_loss, 2)) + "/" + str(round(val_loss, 2)), 
                                  score=str(round(train_score, 2)) + "/" + str(round(val_score, 2))
                                  )
            self.pbar.update(1)
            if self._i == self.n_epochs:
                self.pbar.close()