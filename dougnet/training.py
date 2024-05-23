import numpy as np
import sys
from datetime import datetime


def LoadMiniBatches(Xtrain, Ytrain, batch_size, seed=None):
    """Create mini-batches"""
        
    # make sure Ytrain is 2d
    if Ytrain.ndim == 1:
        Ytrain = Ytrain.reshape(1, -1)
    
    # randomly shuffle dataset
    random_state = np.random.RandomState(seed)
    random_perm_of_cols = np.arange(Xtrain.shape[1])
    random_state.shuffle(random_perm_of_cols)
    Xtrain = Xtrain[:, random_perm_of_cols]
    Ytrain = Ytrain[:, random_perm_of_cols]

    for i in range(0, Xtrain.shape[1], batch_size):

        # grab mini batch
        X_B = Xtrain[:, i:min(i + batch_size, Xtrain.shape[1])]
        Y_B = Ytrain[:, i:min(i + batch_size, Xtrain.shape[1])]
        
        yield X_B, Y_B
        
        

def _compute_progress(X_data, 
                      Y_data,  
                      x_node, 
                      y_node, 
                      yhat_node, 
                      l_node, 
                      progress_metric):
    """Compute train and validation loss and train and validation metric score"""
    
    # compute loss and score
    x_node.output, y_node.output = X_data, Y_data
    loss = l_node.forward()
    Yhat = yhat_node.output
    score = progress_metric(Yhat, Y_data)

    return score, loss


def _verbose(train_score, train_loss, val_score, val_loss, epoch, n_epochs, start_time):
    """Print Progress to screen (code modified from Sebastian Raschka)"""
    
    elapsed_timedelta = datetime.now() - start_time
    elapsed_time = elapsed_timedelta.seconds + elapsed_timedelta.microseconds / 1e6

    sys.stderr.write('\r%0*d/%d | Train/Val. Loss: %.2f/%.2f ' 
                     '| Train/Val. Score: %.2f%%/%.2f%% ' 
                     '| Elapsed Time: %.2f seconds' %
                     (len(str(n_epochs)), epoch + 1, n_epochs, train_loss, 
                      val_loss, train_score * 100, val_score * 100, elapsed_time)
                     )
    sys.stderr.flush()



class ProgressHelper:
    def __init__(self, 
                 n_epochs, 
                 x_node, 
                 y_node, 
                 yhat_node, 
                 l_node, 
                 progress_metric,
                 record_progress=True, 
                 verbose=True 
                ):
        self.record_progress = record_progress
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.x_node = x_node
        self.y_node = y_node 
        self.yhat_node = yhat_node
        self.l_node = l_node 
        self.progress_metric = progress_metric
        if record_progress:
            self.score_train_ = []
            self.loss_train_ = []
            self.score_val_ = []
            self.loss_val_ = []
        self.start_time = datetime.now()
    
    def update(self, epoch, Xtrain, Ytrain, Xval, Yval):
        train_score, train_loss = _compute_progress(Xtrain, 
                                                    Ytrain,  
                                                    self.x_node, 
                                                    self.y_node, 
                                                    self.yhat_node, 
                                                    self.l_node, 
                                                    self.progress_metric)
        val_score, val_loss = _compute_progress(Xval, 
                                                Yval,  
                                                self.x_node, 
                                                self.y_node, 
                                                self.yhat_node, 
                                                self.l_node, 
                                                self.progress_metric)
        
        if self.record_progress:
            self.score_train_.append(train_score)
            self.loss_train_.append(train_loss)
            self.score_val_.append(val_score)
            self.loss_val_.append(val_loss)
            
        if self.verbose:
            _verbose(train_score, train_loss, val_score, val_loss, epoch, self.n_epochs, self.start_time)



# def ComputeProgress(model_inst, Xtrain, Ytrain, Xval, Yval):
#     """Compute train and validation loss and train and validation metric score"""
    
#     #compute training loss and score
#     Y_hat_train, Z_train = model_inst.predict(Xtrain, return_net_input=True) 
#     train_score = model_inst.progress_metric(Y_hat_train, Ytrain)
#     train_loss = model_inst.total_loss(Z_train, Ytrain)

#     #compute validation loss and score
#     Y_hat_val, Z_val = model_inst.predict(Xval, return_net_input=True) 
#     val_score = model_inst.progress_metric(Y_hat_val, Yval)
#     val_loss = model_inst.total_loss(Z_val, Yval)

#     return train_score, train_loss, val_score, val_loss


# def Verbose(model_inst, train_score, train_loss, val_score, val_loss, epoch):
#     """Print Progress to screen (code modified from Sebastian Raschka)"""
    
#     elapsed_timedelta = datetime.now() - model_inst.start_time
#     elapsed_time = elapsed_timedelta.seconds + elapsed_timedelta.microseconds / 1e6

#     sys.stderr.write('\r%0*d/%d | Train/Val. Loss: %.2f/%.2f ' 
#                      '| Train/Val. Score: %.2f%%/%.2f%% ' 
#                      '| Elapsed Time: %.2f seconds' %
#                      (model_inst.epoch_str_len, epoch + 1, model_inst.n_epochs, train_loss, 
#                       val_loss, train_score * 100, val_score * 100, elapsed_time)
#                      )
#     sys.stderr.flush()


# def RecordProgress(model_inst, train_score, train_loss, val_score, val_loss):
#     """Record training progress"""

#     model_inst.score_train_.append(train_score)
#     model_inst.loss_train_.append(train_loss)
#     model_inst.score_val_.append(val_score)
#     model_inst.loss_val_.append(val_loss)


# def ProgressHelper(model_inst, Xtrain, Ytrain, Xval, Yval, epoch):
#     train_score, train_loss, val_score, val_loss = ComputeProgress(model_inst, \
#         Xtrain, Ytrain, Xval, Yval)

#     if model_inst.progress_metric:
#         RecordProgress(model_inst, train_score, train_loss, val_score, val_loss)
#     if model_inst.verbose:
#         Verbose(model_inst, train_score, train_loss, val_score, val_loss, epoch)