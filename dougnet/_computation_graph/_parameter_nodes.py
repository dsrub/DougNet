from math import sqrt
import numpy as np
from dougnet._computation_graph._node_base import ParameterNode

initializer_dict = {}


# DEFINE INITIALIZER FUNCTIONS
def _zeros(shape, rng, dtype):
    """zero initialization"""
    return np.zeros(shape, dtype=dtype)

def _ones(shape, rng, dtype):
    """zero initialization"""
    return np.ones(shape, dtype=dtype)

def _normal(shape, rng, dtype, mu=0, std=1):
    """normal initialization"""
    return rng.normal(mu, std, shape).astype(dtype)

def _xavier(shape, rng, dtype, gain=sqrt(2), fan_in=None, fan_out=None):
    """xavier initialization with a uniform distribution"""
    if (fan_in is None) or (fan_out is None):
        msg = "default values for fan_in and fan_out only specified for rank-2 and rank-4 tensors"
        assert len(shape) in [2, 4], msg
    
    if fan_in is None:
        fan_in = shape[1] if len(shape) == 2 else shape[1] * shape[2] * shape[3]
    
    if fan_out is None:
        fan_out = shape[0] if len(shape) == 2 else shape[0] * shape[2] * shape[3]
    
    bound = gain * sqrt(2 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, shape).astype(dtype)

def _kaiming(shape, rng, dtype, gain=sqrt(2), fan_in=None):
    """kaiming initialization with a uniform distribution"""
    if fan_in is None:
        msg = "default value for fan_in only specified for rank-2 and rank-4 tensors"
        assert len(shape) in [2, 4], msg
    
    if fan_in is None:
        fan_in = shape[1] if len(shape) == 2 else shape[1] * shape[2] * shape[3]
    
    bound = gain * sqrt(3 / fan_in)
    return rng.uniform(-bound, bound, shape).astype(dtype)

initializer_dict["zeros"] = _zeros
initializer_dict["ones"] = _ones
initializer_dict["normal"] = _normal
initializer_dict["xavier"] = _xavier
initializer_dict["kaiming"] = _kaiming


class WeightNode(ParameterNode):
    """A ParameterNode that stores a weight tensor for the neural net."""
    def __init__(self, *shape, dtype=np.float32, initializer="normal", **init_kwargs):
        super().__init__()
        self.shape = shape
        self.dtype = dtype
        self.init_func = initializer_dict[initializer]
        self.init_kwargs = init_kwargs
    
    def initialize(self, random_state=None):
        """
        Initialize the weight tensor

        Parameters
        ------------
        random_state : int or numpy random state object
            The random state with which to initialize the weights.
        """
        if (type(random_state) == int) or random_state is None:
            random_state = np.random.RandomState(random_state)
        self.output = self.init_func(self.shape, random_state, self.dtype, **self.init_kwargs)
 
    
class BiasNode(ParameterNode):
    """A ParameterNode that stores a bias vector for the neural net."""
    def __init__(self, size, dtype=np.float32, initializer="zeros", **init_kwargs):
        super().__init__()
        self.shape = (size, 1)
        self.dtype = dtype
        self.init_func = initializer_dict[initializer]
        self.init_kwargs = init_kwargs
    
    def initialize(self, random_state=None):
        """
        Initialize the bias tensor.

        Parameters
        ------------
        random_state : int or numpy random state object
            The random state with which to initialize the weights.
        """
        if (type(random_state) == int) or random_state is None:
            random_state = np.random.RandomState(random_state)
        self.output = self.init_func(self.shape, random_state, self.dtype, **self.init_kwargs).reshape(-1, 1)
        
        

