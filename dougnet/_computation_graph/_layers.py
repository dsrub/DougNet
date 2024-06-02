import numpy as np
import dougnet as dn
from dougnet._computation_graph._parameter_nodes import WeightNode, BiasNode


class Layer:
    def __init__(self):
        self.node = self._forward()
        
    def _forward(self):
        raise NotImplementedError()
        
    def forward(self):
        return self.node.forward()
        
    def backward(self):
        self.node.backward()
        
    
class Linear(Layer):
    """A computation computing Z = WX + b.  The parameters W, b are created in 
    this class.

    Parameters
    ------------
    Z : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation.
    out_features : int
        number of output features
    in_features : int
        number of input features
    dtype : numpy data type (default=np.float32)
        datatype of the created weights
    """
    def __init__(self, 
                 Z, 
                 out_features, 
                 in_features, 
                 dtype=np.float32, 
                 weight_init="normal", 
                 bias_init="zeros", 
                 weight_init_kwargs={},
                 bias_init_kwargs={}):
        self.Z = Z
        self.out_features = out_features
        self.in_features = in_features
        self.dtype = dtype
        self.weight = WeightNode(self.out_features, 
                                 self.in_features, 
                                 dtype=self.dtype, 
                                 initializer=weight_init, 
                                 **weight_init_kwargs)
        self.bias = BiasNode(self.out_features, 
                             dtype=self.dtype,
                             initializer=bias_init, 
                            **bias_init_kwargs)
        super().__init__()
        
    def _forward(self):
        return dn.MatMult(self.weight, self.Z) + self.bias        