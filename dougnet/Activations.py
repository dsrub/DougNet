# import numpy as np

# class Relu:
#     @staticmethod
#     def func(z):
#         """Relu activation function."""
#         return np.maximum(0, z)
    
#     @staticmethod
#     def deriv(z):
#         """Derivative of Relu."""
#         return (z >= 0).astype(int)
    
    
# class Sigmoid:
#     @staticmethod
#     def func(z): 
#         """Sigmoid activation function."""
#         return 1./(1. + np.exp(-z))
    
#     @staticmethod
#     def deriv(z):
#         """Derivative of Sigmoid."""
#         return Sigmoid.func(z) * (1 - Sigmoid.func(z))
    
    
# class Tanh:
#     @staticmethod
#     def func(z): 
#         """Tanh activation function."""
#         return np.tanh(z)
    
#     @staticmethod
#     def deriv(z):
#         """Derivative of Tanh."""
#         return 1 - np.tanh(z)**2
    
    
# class SoftMax:   
#     @staticmethod
#     def func(z):
#         """Softmax activation function with trick avoid overflow."""
#         z = z - z.max(axis=0)
#         z = np.exp(z)
#         return z / z.sum(axis=0)

# class Identity:   
#     @staticmethod
#     def func(z):
#         """Identity activation function."""
#         return z 


import numpy as np


def relu(z):
    """Relu activation function."""
    return np.maximum(0, z)
register_vjp(relu, lambda g, z: g * (z >= 0).astype(int).astype(z.dtype))
    
def sigmoid(z): 
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))
register_vjp(sigmoid, lambda g, z: g * sigmoid(z) * (1 - sigmoid(z)))

def tanh(z):
    """Tanh activation function."""
    return np.tanh(z)
register_vjp(tanh, lambda g, z: g * (1 - np.tanh(z) ** 2))
    
def softmax(z): 
    """Softmax activation function with trick avoid overflow."""  
    z = z - z.max(axis=0)
    z = np.exp(z)
    return z / z.sum(axis=0)
