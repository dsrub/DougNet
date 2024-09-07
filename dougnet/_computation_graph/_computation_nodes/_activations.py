import numpy as np
from dougnet._computation_graph._node_base import ComputationNode, output_container
from dougnet.functions._activations import *


class Sigmoid(ComputationNode):
    """An element-wise sigmoid computation node."""
    def __init__(self, z):
        super().__init__([z])
        self.func = lambda zz: output_container(sigmoid(zz.output))
        self.vjps[z] = lambda gg, cache, zz: gg * sigmoid(zz.output) * (1 - sigmoid(zz.output))

class Relu(ComputationNode):
    """An element-wise relu computation node."""
    def __init__(self, z,):
        super().__init__([z])
        self.func = lambda zz: output_container(relu(zz.output))
        self.vjps[z] = lambda gg, cache, zz: gg * (zz.output > 0).astype(int).astype(zz.output.dtype)

class Tanh(ComputationNode):
    """An element-wise tanh computation node."""
    def __init__(self, z):
        super().__init__([z])
        self.func = lambda zz: output_container(tanh(zz.output))
        self.vjps[z] = lambda gg, cache, zz: gg * (1 - tanh(zz.output) ** 2)

class Softmax(ComputationNode):
    """A softmax computation node."""  
    def __init__(self, z):
        super().__init__([z])
        self.func = lambda zz: output_container(softmax(zz.output))
        self.vjps[z] = None