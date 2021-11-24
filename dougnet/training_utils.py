import numpy as np
import sys
from datetime import datetime


def GetMiniBatches(model_inst, Xtrain, Ytrain):
    """Create mini-batches"""
    
    # randomly shuffle dataset
    random_perm_of_cols = np.arange(Xtrain.shape[1])
    model_inst.random.shuffle(random_perm_of_cols)
    Xtrain = Xtrain[:, random_perm_of_cols]
    Ytrain = Ytrain[:, random_perm_of_cols]

    for i in range(0, Xtrain.shape[1], model_inst.B):

        # grab mini batch
        X_B = Xtrain[:, i:min(i + model_inst.B, Xtrain.shape[1])]
        Y_B = Ytrain[:, i:min(i + model_inst.B, Xtrain.shape[1])]
        
        yield X_B, Y_B



def ComputeProgress(model_inst, Xtrain, Ytrain, Xval, Yval):
    """Compute train and validation loss and train and validation metric score"""
    
    #compute training loss and score
    Y_hat_train, Z_train = model_inst.predict(Xtrain, return_net_input=True) 
    train_score = model_inst.progress_metric(Y_hat_train, Ytrain)
    train_loss = model_inst.total_loss(Z_train, Ytrain)

    #compute validation loss and score
    Y_hat_val, Z_val = model_inst.predict(Xval, return_net_input=True) 
    val_score = model_inst.progress_metric(Y_hat_val, Yval)
    val_loss = model_inst.total_loss(Z_val, Yval)

    return train_score, train_loss, val_score, val_loss


def Verbose(model_inst, train_score, train_loss, val_score, val_loss, epoch):
    """Print Progress to screen (code modified from Sebastian Raschka)"""
    
    elapsed_timedelta = datetime.now() - model_inst.start_time
    elapsed_time = elapsed_timedelta.seconds + elapsed_timedelta.microseconds / 1e6

    sys.stderr.write('\r%0*d/%d | Train/Val. Loss: %.2f/%.2f ' 
                     '| Train/Val. Score: %.2f%%/%.2f%% ' 
                     '| Elapsed Time: %.2f seconds' %
                     (model_inst.epoch_str_len, epoch + 1, model_inst.n_epochs, train_loss, 
                      val_loss, train_score * 100, val_score * 100, elapsed_time)
                     )
    sys.stderr.flush()


def RecordProgress(model_inst, train_score, train_loss, val_score, val_loss):
    """Record training progress"""

    model_inst.score_train_.append(train_score)
    model_inst.loss_train_.append(train_loss)
    model_inst.score_val_.append(val_score)
    model_inst.loss_val_.append(val_loss)


def ProgressHelper(model_inst, Xtrain, Ytrain, Xval, Yval, epoch):
    train_score, train_loss, val_score, val_loss = ComputeProgress(model_inst, \
        Xtrain, Ytrain, Xval, Yval)

    if model_inst.progress_metric:
        RecordProgress(model_inst, train_score, train_loss, val_score, val_loss)
    if model_inst.verbose:
        Verbose(model_inst, train_score, train_loss, val_score, val_loss, epoch)