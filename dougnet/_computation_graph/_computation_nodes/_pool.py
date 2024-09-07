from dougnet._computation_graph._node_base import ComputationNode, output_container
from dougnet.functions._pool import (mp2d, 
                                     _dZ_mp2d, 
                                     gap2d, 
                                     _dZ_gap2d)


class MP2d(ComputationNode):
    """A max pool computation node."""
    def __init__(self, Z, H_K=3, W_K=3, stride=1):
        super().__init__([Z])
        self.H_K = H_K
        self.W_K = W_K
        self.stride = stride
        self.func = lambda ZZ: output_container(*mp2d(ZZ.output, 
                                                      H_K=H_K, 
                                                      W_K=W_K, 
                                                      stride=stride, 
                                                      return_inds=True))
        self.vjps[Z] = lambda gg, cache, ZZ: _dZ_mp2d(gg, 
                                                      ZZ.output,
                                                      cache[0], 
                                                      cache[1], 
                                                      H_K, 
                                                      W_K, 
                                                      stride=stride)
        
class GAP2d(ComputationNode):
    """A global average pool computation node."""
    def __init__(self, Z):
        super().__init__([Z])
        self.func = lambda ZZ: output_container(gap2d(ZZ.output))
        self.vjps[Z] = lambda gg, cache, ZZ: _dZ_gap2d(gg, ZZ.output)