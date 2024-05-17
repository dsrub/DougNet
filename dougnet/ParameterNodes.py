import numpy as np
from dougnet.graph import ParameterNode

class WeightNode(ParameterNode):
    """A ParameterNode that stores a weight matrix for the neural net.

    Parameters
    ------------
    num_units : int
        The number of neuron units in this layer.
    prev_num_units : int
        The number of neuron units in the previous layer.
    """
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    
    def initialize(self, random_state=None):
        """Initialize the weight matrix

        Parameters
        ------------
        random_state : numpy random state object
            The random state with which to initialize the weights. (default is None and 
            corresponds to a randomly chosen seed.)
        """
        if not random_state:
            self.output = np.random.normal(0, 1, self.shape)
        else:
            self.output = random_state.normal(0, 1, self.shape)
       
    
class BiasNode(ParameterNode):
    """A ParameterNode that stores a bias vector for the neural net.

    Parameters
    ------------
    num_units : int
        The number of neuron units in this layer.
    """
    def __init__(self, num_units):
        super().__init__()
        self.num_units = num_units
    
    def initialize(self, random_state=None):
        """Initialize the bias vector
        """
        self.output = np.zeros((1, self.num_units)).T
