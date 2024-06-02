from math import sqrt
import numpy as np
from dougnet._computation_graph._graph_base import ParameterNode

initializer_dict = {}

def _zeros(shape, rng, dtype):
    """zero initialization"""
    return np.zeros(shape, dtype=dtype)

def _ones(shape, rng, dtype):
    """zero initialization"""
    return np.ones(shape, dtype=dtype)

def _normal(shape, rng, dtype, mu=0, std=1):
    """normal initialization"""
    return rng.normal(mu, std, shape).astype(dtype)

def _xavier(shape, rng, dtype, gain=1.0, fan_in=1, fan_out=1):
    """xavier initialization with a uniform distribution"""
    bound = gain * sqrt(2 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, shape).astype(dtype)

def _kaiming(shape, rng, dtype, gain=1.0, fan=1):
    """kaiming initialization with a uniform distribution"""
    bound = gain * sqrt(3 / fan)
    return rng.uniform(-bound, bound, shape).astype(dtype)

initializer_dict["zeros"] = _zeros
initializer_dict["ones"] = _ones
initializer_dict["normal"] = _normal
initializer_dict["xavier"] = _xavier
initializer_dict["kaiming"] = _kaiming


class WeightNode(ParameterNode):
    """
    A ParameterNode that stores a weight matrix for the neural net.
    """
    def __init__(self, *shape, dtype=np.float32, initializer="normal", **init_kwargs):
        super().__init__()
        self.shape = shape
        self.dtype = dtype
        self.init_func = initializer_dict[initializer]
        self.init_kwargs = init_kwargs
    
    def initialize(self, random_state=None):
        """Initialize the weight matrix

        Parameters
        ------------
        random_state : int or numpy random state object
            The random state with which to initialize the weights.
        """
        if (type(random_state) == int) or random_state is None:
            random_state = np.random.RandomState(random_state)
        self.output = self.init_func(self.shape, random_state, self.dtype, **self.init_kwargs)
 
    
class BiasNode(ParameterNode):
    """
    A ParameterNode that stores a bias vector for the neural net.
    """
    def __init__(self, size, dtype=np.float32, initializer="zeros", **init_kwargs):
        super().__init__()
        self.shape = (size, 1)
        self.dtype = dtype
        self.init_func = initializer_dict[initializer]
        self.init_kwargs = init_kwargs
    
    def initialize(self, random_state=None):
        """Initialize the weight matrix

        Parameters
        ------------
        random_state : int or numpy random state object
            The random state with which to initialize the weights.
        """
        if (type(random_state) == int) or random_state is None:
            random_state = np.random.RandomState(random_state)
        self.output = self.init_func(self.shape, random_state, self.dtype, **self.init_kwargs).reshape(-1, 1)
        
        

