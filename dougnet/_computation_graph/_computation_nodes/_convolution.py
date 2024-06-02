import numpy as np
from dougnet._computation_graph._graph_base import ComputationNode, output_container
from dougnet.nn_functions._convnet_funcs._convolution import (conv2d,
                                                              _db as _db_conv2d, 
                                                              _dK as _dK_conv2d, 
                                                              _dV as _dV_conv2d)
from dougnet.nn_functions._convnet_funcs._pool import (mp2d,
                                                       _dZ_mp2d,
                                                       gap2d,
                                                       _dZ_gap2d
                                                       )

class Conv2d(ComputationNode):   
    
    #def __init__(self, V, out_channels, kernel_size, pad=0, stride=1, dilate=1, dtype=np.float32):
    #    self.kernel_size = kernel_size
    #    self.out_channels = out_channels
    #    self.pad = pad
    #    self.stride = stride
    #    self.dilate = dilate

    #    # instantiate kernel tensor and bias vector
    #    N, C_in, _, _ = V.shape
    #    K_H = K_W = kernel_size
    #    if type(kernel_size) == tuple:
    #        K_H, K_W = kernel_size
    #    self.K = WeightNode(out_channels, C_in, K_H, K_W)
    #    self.b = BiasNode(out_channels)
    #    super().__init__([V, self.K, self.b])

     
    def __init__(self, V, K, b, pad=0, stride=1, dilate=1):
        super().__init__([V, K, b])
        self.pad = pad
        self.stride = stride
        self.dilate = dilate
        self.func = lambda VV, KK, bb: output_container(*conv2d(VV.output, 
                                                      KK.output, 
                                                      bb.output, 
                                                      pad=pad, 
                                                      stride=stride, 
                                                      dilate=dilate,
                                                      return_Vim2col=True
                                                      )
                                              )
        self.vjps[V] = lambda gg, cache, VV, KK, bb: _dV_conv2d(gg, 
                                                                VV.output, 
                                                                KK.output, 
                                                                pad=pad, 
                                                                stride=stride, 
                                                                dilate=dilate
                                                                )
        self.vjps[K] = lambda gg, cache, VV, KK, bb: _dK_conv2d(gg, 
                                                                KK.output, 
                                                                cache
                                                                )
        self.vjps[b] = lambda gg, cache, VV, KK, bb: _db_conv2d(gg)
        
class MP2d(ComputationNode):    
    def __init__(self, Z, H_K=3, W_K=3, stride=1):
        super().__init__([Z])
        self.H_K = H_K
        self.W_K = W_K
        self.stride = stride
        self.func = lambda ZZ: output_container(*MP2d(ZZ.output, 
                                            H_K=H_K, 
                                            W_K=W_K, 
                                            stride=stride,
                                            return_inds=True
                                            )
                                      )        
        self.vjps[Z] = lambda gg, cache, ZZ: _dZ_MP2d(gg, 
                                                      ZZ.output, 
                                                      cache[0],
                                                      cache[1],
                                                      H_K,
                                                      W_K,
                                                      stride=stride
                                                      )
        
class GAP2d(ComputationNode):    
    def __init__(self, Z):
        super().__init__([Z])
        self.func = lambda ZZ: output_container(GAP2d(ZZ.output))        
        self.vjps[Z] = lambda gg, cache, ZZ: _dZ_GAP2d(gg, ZZ.output)
        
class BN2d(ComputationNode):    
    pass