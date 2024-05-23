import numpy as np
from dougnet._computation_graph._graph_base import ComputationNode, OUTPUT
from dougnet.nn_functions._activations import *
from dougnet.nn_functions._losses import *
from dougnet._computation_graph._parameter_nodes import WeightNode, BiasNode
from dougnet.nn_functions._convnet_funcs._convolution import (conv2d,
                                                              _db as _db_conv2d, 
                                                              _dK as _dK_conv2d, 
                                                              _dV as _dV_conv2d)
from dougnet.nn_functions._convnet_funcs._pool import (mp2d,
                                                       _dZ_mp2d,
                                                       gap2d,
                                                       _dZ_gap2d
                                                       )


################# MATH NODES ################
class Sqr(ComputationNode):
    """An element-wise square computation node.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray or a float)
    """
    def __init__(self, x):
        super().__init__([x])
        self.func = lambda xx: OUTPUT(xx.output ** 2)
        self.vjps[x] = lambda gg, cache, xx: 2 * xx.output * gg

class MatMult(ComputationNode):
    """A matrix multiplication computation node.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode
        The parent node from which to make the computation (x.output must be an 
        ndarray matrix)
    y : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation (y.output must be an 
        ndarray matrix with dims. st. the matrix multiplication x.output times 
        y.output makes sense)
    """
    def __init__(self, x, y):
        super().__init__([x, y])
        self.func = lambda xx, yy: OUTPUT(np.dot(xx.output, yy.output))
        self.vjps[x] = lambda gg, cache, xx, yy: np.dot(gg, yy.output.T)
        self.vjps[y] = lambda gg, cache, xx, yy: np.dot(xx.output.T, gg)

class Sqrt(ComputationNode):
    """An element-wise sqaure root computation node.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray or a float).  Note that all elements of 
        x.output must be non-negative.
    """
    def __init__(self, x):
        super().__init__([x])
        self.func = lambda xx: OUTPUT(np.sqrt(xx.output)) 
        self.vjps[x] = lambda gg, cache, xx: gg / (2. * np.sqrt(xx.output))

class Cos(ComputationNode):
    """An element-wise cosine computation node.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray or a float)
    """
    def __init__(self, x):
        super().__init__([x])
        self.func = lambda xx: OUTPUT(np.cos(xx.output))
        self.vjps[x] = lambda gg, cache, xx: -np.sin(xx.output) * gg

class Exp(ComputationNode):
    """An element-wise exponentiation computation node.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray or a float)
    """
    def __init__(self, x):
        super().__init__([x])
        self.func = lambda xx: OUTPUT(np.exp(xx.output))
        self.vjps[x] = lambda gg, cache, xx: np.exp(xx.output) * gg
        
class Sum(ComputationNode):
    """A computation node summing all elements of the input tensor.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray)
    """
    def __init__(self, x):
        super().__init__([x])
        self.func = lambda xx: OUTPUT(np.sum(xx.output))
        self.vjps[x] = lambda gg, cache, xx: np.zeros_like(xx.output) + gg
       
       
############### MLP NODES ###############
class Linear(ComputationNode):
    """A computation computing Z = WX + b.  The parameters W, b are created in 
    this class.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray)
    out_features : int
        number of output features
    in_features : int
        number of input features
    dtype : numpy data type (default=np.float32)
        datatype of the created weights
    """
    def __init__(self, x, out_features, in_features, dtype=np.float32):
        # instantiate weights and bias
        self.W = WeightNode(out_features, in_features, dtype=dtype)
        self.b = BiasNode(out_features, dtype=dtype)
        super().__init__([x, self.W, self.b])
        
        self.func = lambda xx, WW, bb: OUTPUT(np.dot(WW.output, xx.output) + bb.output)
        self.vjps[self.W] = lambda gg, cache, xx, WW, bb: np.dot(gg, xx.output.T)
        self.vjps[x] = lambda gg, cache, xx, WW, bb: np.dot(WW.output.T, gg)
        self.vjps[self.b] = lambda gg, cache, xx, WW, bb: np.sum(gg, axis=1).reshape(gg.shape[0], 1)


############## ACTIVATION NODES #############
class Sigmoid(ComputationNode):
    """An element-wise sigmoid computation node.

    Parameters
    ------------
    z : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation (z.output can 
        be an arbitrary dim. ndarray or a float)
    """
    def __init__(self, z):
        super().__init__([z])
        self.func = lambda zz: OUTPUT(sigmoid(zz.output))
        self.vjps[z] = lambda gg, cache, zz: gg * sigmoid(zz.output) * (1 - sigmoid(zz.output))

class Relu(ComputationNode):
    """An element-wise relu computation node.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation (z.output can 
        be an arbitrary dim. ndarray or a float)
    """
    def __init__(self, z):
        super().__init__([z])
        self.func = lambda zz: OUTPUT(relu(zz.output))
        self.vjps[z] = lambda gg, cache, zz: gg * (zz.output >= 0).astype(int).astype(zz.output.dtype)

class Tanh(ComputationNode):
    """An element-wise tanh computation node.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray or a float)
    """
    def __init__(self, z):
        super().__init__([z])
        self.func = lambda zz: OUTPUT(tanh(zz.output))
        self.vjps[z] = lambda gg, cache, zz: gg * (1 - tanh(zz.output) ** 2)

class Softmax(ComputationNode):
    """A softmax computation node.

    Parameters
    ------------
    z : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation. Note that z.output must 
        either be an ndarray column vector or an ndarray m x n matrix, where the 
        softmax is performed column-wise in the matrix (i.e, each data point is stored
        as a column vector in z.output).
    """  
    def __init__(self, z):
        super().__init__([z])
        self.func = lambda zz: OUTPUT(softmax(zz.output))
        self.vjps[z] = None


############### LOSS NODES #################
class SoftmaxCrossEntropyLoss(ComputationNode):
    """A computation node that applies a softmax activation to its input then
    computes the cross entropy loss with a specified regularization loss for the
    weight parameters of the model.

    Parameters
    ------------
    Z : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation. Note that z.output must 
        either be an ndarray column vector or an ndarray m x n matrix, where the 
        softmax is performed column-wise in the matrix (i.e, each data point is stored
        as a column vector in z.output).
    Y_ohe : InputNode 
        The parent node from which to make the computation. This node corresponds to 
        the one-hot encoded data values associated with Z.output and therefore Y_ohe 
        must have the same dims as Z.
    """
    def __init__(self, Z, Y_ohe):
        super().__init__([Z, Y_ohe])
        self.func = lambda ZZ, YY: OUTPUT(softmax_cross_entropy_loss(ZZ.output, YY.output))
        self.vjps[Z] = lambda gg, cache, ZZ, YY: (softmax(ZZ.output) - YY.output) * gg / YY.output.shape[1]
        self.vjps[Y_ohe] = lambda gg, cache, ZZ, YY: None

class L2Loss(ComputationNode):
    """A computation node that applies an identity activation to its input then
    computes squared loss with a specified regularization loss for the
    weight parameters of the model.

    Parameters
    ------------
    Z : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation. Note that z.output must 
        either be an ndarray column vector or an ndarray m x n matrix, where the 
        softmax is performed column-wise in the matrix (i.e, each data point is stored
        as a column vector in z.output).
    Y : InputNode 
        The parent node from which to make the computation. This node corresponds to 
        the data values associated with Z.output and therefore Ymust have the same 
        dims as Z.
    """
    def __init__(self, Z, Y):
        super().__init__([Z, Y])
        self.func = lambda ZZ, YY: OUTPUT(l2loss(ZZ.output, YY.output))
        self.vjps[Z] = lambda gg, cache, ZZ, YY: (ZZ.output - YY.output) * gg / YY.output.shape[1]
        self.vjps[Y] = lambda gg, cache, ZZ, YY: None

class L2RegLoss(ComputationNode):
    """A computation node that computes the L2 regularization loss of the weights in the
    neural net.

    Parameters
    ------------
    W_node_list : list 
        A list of ParameterNodes which correspond to the weight matrices of the model.
    lmbda : float
        The L2 regularization strength (default = .1)
    """ 
    def __init__(self, *Ws, lmbda=.1):
        super().__init__(list(Ws))   
        self.lmbda = lmbda  
        self.func = lambda *WWs: OUTPUT(l2regloss(*(W.output for W in WWs), lmbda=lmbda))
        self.vjps = {W: lambda gg, cache, *all_parents, WW=W: lmbda * WW.output * gg for W in Ws}
        

################# CONVOLUTION NODES ################
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
        self.func = lambda VV, KK, bb: OUTPUT(*conv2d(VV.output, 
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
        self.func = lambda ZZ: OUTPUT(*MP2d(ZZ.output, 
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
        self.func = lambda ZZ: OUTPUT(GAP2d(ZZ.output))        
        self.vjps[Z] = lambda gg, cache, ZZ: _dZ_GAP2d(gg, ZZ.output)
        
class BN2d(ComputationNode):    
    pass