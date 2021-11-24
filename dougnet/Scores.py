import numpy as np

def Accuracy(Y_hat, Y):
	"""Return accuracy given one-hot encoded predictions, Y_hat, and the corresponding
	one-hot encoded ground truth labels, Y."""
	return np.sum(np.argmax(Y_hat, axis=0) == np.argmax((Y), axis=0))/Y.shape[1]

def RMSE(Y_hat, Y):
	"""Return the RMSE given the predictions, Y_hat, and the corresponding ground truth
	labels, Y."""
	return np.sqrt(np.sum((Y_hat - Y)**2) / Y.shape[1])

def R2(Y_hat, Y):
	"""Return the R^2 given the predictions, Y_hat, and the corresponding ground truth
	labels, Y."""
	return 1 - (np.sum((Y_hat - Y)**2))/(np.sum((Y_hat - np.mean(Y))**2))