import numpy as np

def relu(z):
    """Relu activation function."""
    return np.maximum(0, z)
    
def sigmoid(z): 
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def tanh(z):
    """Tanh activation function."""
    return np.tanh(z)

def identity(z):
    """Identity activation function."""
    return z
    
def softmax(z): 
    """Softmax activation function with trick avoid overflow."""  
    z = z - z.max(axis=0)
    z = np.exp(z)
    return z / z.sum(axis=0)
