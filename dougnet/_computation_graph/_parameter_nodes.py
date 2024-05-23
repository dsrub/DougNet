import numpy as np
from dougnet._computation_graph._graph_base import ParameterNode


class WeightNode(ParameterNode):
    """A ParameterNode that stores a weight matrix for the neural net.

    Parameters
    ------------
    num_units : int
        The number of neuron units in this layer.
    prev_num_units : int
        The number of neuron units in the previous layer.
    """
    def __init__(self, *shape, dtype=np.float32):
        super().__init__()
        self.shape = shape
        self.dtype = dtype
    
    def initialize(self, random_state=None):
        """Initialize the weight matrix

        Parameters
        ------------
        random_state : int or numpy random state object
            The random state with which to initialize the weights.
        """
        if type(random_state) == int:
            random_state = np.random.RandomState(random_state)
        self.output = random_state.normal(0, 1, self.shape).astype(self.dtype)
       
    
class BiasNode(ParameterNode):
    """A ParameterNode that stores a bias vector for the neural net.

    Parameters
    ------------
    num_units : int
        The number of neuron units in this layer.
    """
    def __init__(self, num_units, dtype=np.float32):
        super().__init__()
        self.num_units = num_units
        self.dtype = dtype
    
    def initialize(self, random_state=None):
        """Initialize the bias vector
        """
        self.output = np.zeros((1, self.num_units), dtype=self.dtype).T
