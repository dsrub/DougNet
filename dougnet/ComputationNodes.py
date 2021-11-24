import numpy as np
from dougnet.graph import ComputationNode
from dougnet.Activations import *
from dougnet.Losses import *

class sqr(ComputationNode):
    """An element-wise square computation node.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray or a float)
    """
    def __init__(self, x):
        super().__init__([x])
        self.computation = lambda xx: xx.output ** 2 
        self.vjps[x] = lambda gg, xx: 2 * xx.output * gg


class add(ComputationNode):
    """An element-wise addition node.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode  
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray or a float).
    y : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation. Note that y.output 
        must be the same dimensions as x.output, or x.output must be an m x n matrix
        and y.output must be a length m column vector to be broadcasted to m x n then 
        added with x.output.
    """
    def __init__(self, x, y):
        super().__init__([x, y])
        self.computation = lambda xx, yy: xx.output + yy.output
        self.vjps[x] = lambda gg, xx, yy: gg
        self.vjps[y] = lambda gg, xx, yy: gg if xx.output.shape == yy.output.shape \
        else np.sum(gg, axis=1).reshape(gg.shape[0], 1)


class subtract(ComputationNode):
    """An element-wise subtraction node.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode  
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray or a float).
    y : ComputationNode, InputNode or ParameterNode  
        The parent node from which to make the computation. Note that y.output 
        must be the same dimensions as x.output.
    """
    def __init__(self, x, y):
        super().__init__([x, y])
        self.computation = lambda xx, yy: xx.output - yy.output
        self.vjps[x] = lambda gg, xx, yy: gg
        self.vjps[y] = lambda gg, xx, yy: -gg


class mult(ComputationNode):
    """An element-wise multiplication (hadamard multiplication) computation node.  Note 
    that if the output of the two input nodes are scalars, then this node performs normal, 
    scalar multiplication.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode  
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray or a float).
    y : ComputationNode, InputNode or ParameterNode  
        The parent node from which to make the computation. Note that y.output 
        must be the same dimensions as x.output.
    """
    def __init__(self, x, y):
        super().__init__([x, y])
        self.computation = lambda xx, yy: xx.output * yy.output
        self.vjps[x] = lambda gg, xx, yy: yy.output * gg
        self.vjps[y] = lambda gg, xx, yy: xx.output * gg


class sigmoid(ComputationNode):
    """An element-wise sigmoid computation node.

    Parameters
    ------------
    z : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation (z.output can 
        be an arbitrary dim. ndarray or a float)
    """
    def __init__(self, z):
        super().__init__([z])
        self.computation = lambda zz: Sigmoid.func(zz.output)
        self.vjps[z] = lambda gg, zz: Sigmoid.deriv(zz.output) * gg


class matmult(ComputationNode):
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
        self.computation = lambda xx, yy: np.dot(xx.output, yy.output)
        self.vjps[x] = lambda gg, xx, yy: np.dot(gg, yy.output.T)
        self.vjps[y] = lambda gg, xx, yy: np.dot(x.output.T, gg)


class sqrt(ComputationNode):
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
        self.computation = lambda xx: np.sqrt(xx.output) 
        self.vjps[x] = lambda gg, xx: gg / (2.* np.sqrt(xx.output))


class cos(ComputationNode):
    """An element-wise cosine computation node.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray or a float)
    """
    def __init__(self, x):
        super().__init__([x])
        self.computation = lambda xx: np.cos(xx.output)
        self.vjps[x] = lambda gg, xx: -np.sin(xx.output) * gg


class exp(ComputationNode):
    """An element-wise exponentiation computation node.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray or a float)
    """
    def __init__(self, x):
        super().__init__([x])
        self.computation = lambda xx: np.exp(xx.output)
        self.vjps[x] = lambda gg, xx: np.exp(xx.output) * gg


class relu(ComputationNode):
    """An element-wise relu computation node.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation (z.output can 
        be an arbitrary dim. ndarray or a float)
    """
    def __init__(self, z):
        super().__init__([z])
        self.computation = lambda zz: Relu.func(zz.output)
        self.vjps[z] = lambda gg, zz: Relu.deriv(zz.output) * gg


class tanh(ComputationNode):
    """An element-wise tanh computation node.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray or a float)
    """
    def __init__(self, z):
        super().__init__([z])
        self.computation = lambda zz: Tanh.func(zz.output)
        self.vjps[z] = lambda gg, zz: Tanh.deriv(zz.output) * gg


class softmax(ComputationNode):
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
        self.computation = lambda zz: SoftMax.func(zz.output)
        self.vjps[z] = None


class softmax_crossentropy_loss(ComputationNode):
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
    W_loss : ComputationNode
        The parent node from which to make the computation. This node corresponds to
        the regularization loss applied to the weights of the model.
    """
    def __init__(self, Z, Y_ohe, W_loss):
        super().__init__([Z, Y_ohe, W_loss])
        self.computation = lambda ZZ, YY, WW_loss: SoftmaxCrossEntropyLoss.func(ZZ.output, YY.output) + WW_loss.output
        self.vjps[Z] = lambda gg, ZZ, YY, WW_loss: (SoftMax.func(ZZ.output) - YY.output) * gg
        self.vjps[Y_ohe] = lambda gg, ZZ, YY, WW_loss: None
        self.vjps[W_loss] = lambda gg, ZZ, YY, WW_loss: gg


class l2_loss(ComputationNode):
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
    W_loss : ComputationNode
        The parent node from which to make the computation. This node corresponds to
        the regularization loss applied to the weights of the model.
    """
    def __init__(self, Z, Y, W_loss):
        super().__init__([Z, Y, W_loss])

        self.computation = lambda ZZ, YY, WW_loss: .5 * np.sum((ZZ.output - YY.output)**2)/YY.output.shape[1] \
                                                    + WW_loss.output
        self.vjps[Z] = lambda gg, ZZ, YY, WW_loss: (ZZ.output - YY.output) * gg
        self.vjps[Y] = lambda gg, ZZ, YY, WW_loss: None
        self.vjps[W_loss] = lambda gg, ZZ, YY, WW_loss: gg


class l2_reg_loss(ComputationNode):
    """A computation node that computes the L2 regularization loss of the weights in the
    neural net.

    Parameters
    ------------
    W_node_list : list 
        A list of ParameterNodes which correspond to the weight matrices of the model.
    lmbda : float
        The L2 regularization strength (default = .1)
    """
    def __init__(self, W_node_list, lmbda=.1):
        super().__init__(W_node_list)     
        self.computation = lambda *Ws: lmbda * L2RegLoss.func([W.output for W in Ws])
        self.vjps = {W:lambda gg, *args, WW=W: 2 * lmbda * WW.output * gg for W in W_node_list}