import numpy as np
from dougnet._computation_graph._node_base import ComputationNode, output_container
from dougnet._computation_graph._parameter_nodes import WeightNode
from dougnet.functions._convolution import (conv2d,
                                            _db as _db_conv2d, 
                                            _dK as _dK_conv2d, 
                                            _dV as _dV_conv2d)
conv2d_docstring = """
A multi-channel convolution computation node.

Parameters
------------
V : Node
    The input node to be convolved.  The V.output should be a rank-4 tensor of shape
    [N x C x H x W].
in_channels : int
    The number of input channels, C.
out_channels : int
    The number of output channels.  
kernel_size : int or tuple, default=3
    The height and width of the kernel.  If a tuple is provided, this corresponds 
    to (H_K, W_K).
pad : int, default=0
    How much to pad the edges of each image.  Zero padding is used.
stride : int, default=1
    The convolution stride.
dilate : int, default=1
    How much to dilate the kernel tensor (dilate=1 corresponds to no dilation).
dtype : np.dtype, default=np.float32
    The dtype to use for the bias vector and kernel tensor.
bias : bool, default=True
    Add a bias term to the multi-channel convolution.
bias_initializer : str, default="zeros"
    The type of initialization for the bias vector.
bias_init_kwargs : dict, default={}
    The initialization arguments for the bias vector.
kernel_initializer : str, default="xavier"
    The type of initialization for the kernel tensor.
kernel_init_kwargs : dict, default={}
    The initialization arguments for the kernel tensor.
"""
        
class Conv2d(ComputationNode):   
    def __init__(self, 
                 V, 
                 in_channels,
                 out_channels, 
                 kernel_size=3, 
                 pad=0, 
                 stride=1, 
                 dilate=1, 
                 dtype=np.float32, 
                 bias=True,
                 bias_initializer="zeros", 
                 bias_init_kwargs={}, 
                 kernel_initializer="xavier", 
                 kernel_init_kwargs={}):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pad = pad
        self.stride = stride
        self.dilate = dilate
        self.dtype = dtype
        self.bias = bias
        self.bias_initializer = bias_initializer
        self.bias_init_kwargs = bias_init_kwargs
        self.kernel_initializer = kernel_initializer
        self.kernel_init_kwargs = kernel_init_kwargs

        # instantiate kernel tensor
        H_K = W_K = kernel_size
        if type(kernel_size) == tuple:
            H_K, W_K = kernel_size
        self.K = WeightNode(self.out_channels, 
                            self.in_channels, 
                            H_K, 
                            W_K, 
                            dtype=dtype, 
                            initializer=kernel_initializer, 
                            **kernel_init_kwargs)
            
        # instantiate bias vector
        if self.bias:
            self.b = WeightNode(self.out_channels, 
                               dtype=dtype, 
                               initializer=bias_initializer, 
                               **bias_init_kwargs)
            
        if self.bias:
            # define compute method and VJPs for bias=True
            super().__init__([V, self.K, self.b])
            self.func = lambda VV, KK, bb: output_container(*conv2d(VV.output, 
                                                                    KK.output, 
                                                                    bb.output, 
                                                                    pad=pad, 
                                                                    stride=stride, 
                                                                    dilate=dilate,
                                                                    return_Vim2col=True))
            self.vjps[V] = lambda gg, cache, VV, KK, bb: _dV_conv2d(gg, 
                                                                    VV.output, 
                                                                    KK.output, 
                                                                    pad=pad, 
                                                                    stride=stride, 
                                                                    dilate=dilate)
            self.vjps[self.K] = lambda gg, cache, VV, KK, bb: _dK_conv2d(gg, KK.output, cache)
            self.vjps[self.b] = lambda gg, cache, VV, KK, bb: _db_conv2d(gg)
        else:
            # define compute method and VJPs for bias=False
            super().__init__([V, self.K])
            self.func = lambda VV, KK: output_container(*conv2d(VV.output, 
                                                                KK.output, 
                                                                None, 
                                                                pad=pad, 
                                                                stride=stride, 
                                                                dilate=dilate,
                                                                return_Vim2col=True))
            self.vjps[V] = lambda gg, cache, VV, KK: _dV_conv2d(gg, 
                                                                VV.output, 
                                                                KK.output, 
                                                                pad=pad, 
                                                                stride=stride, 
                                                                dilate=dilate)
            self.vjps[self.K] = lambda gg, cache, VV, KK: _dK_conv2d(gg, KK.output, cache)
Conv2d.__doc__ = conv2d_docstring