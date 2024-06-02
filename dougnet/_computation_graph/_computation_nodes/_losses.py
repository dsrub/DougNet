from dougnet._computation_graph._graph_base import ComputationNode, output_container
from dougnet.nn_functions._activations import *
from dougnet.nn_functions._losses import *


class SoftmaxCrossEntropyLoss(ComputationNode):
    """
    A computation node that applies a softmax activation to its input then
    computes the cross entropy loss.

    Parameters
    ------------
    Z : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation. Note that Z.output must 
        either be an ndarray column vector or an ndarray m x n matrix, where the 
        softmax is performed column-wise in the matrix (i.e, each data point is stored
        as a column vector in Z.output).
    Y_ohe : InputNode 
        The parent node from which to make the computation. This node corresponds to 
        the one-hot encoded data values associated with Z.output and therefore Y_ohe 
        must have the same dims as Z.
    graph : ComputationGraph instance or "default", default="default"
        The graph to which the node is assigned.  If "default", the node is assigned to
        the currently active global default graph.
    """
    def __init__(self, Z, Y_ohe):
        super().__init__([Z, Y_ohe])
        self.func = lambda ZZ, YY: output_container(softmax_cross_entropy_loss(ZZ.output, YY.output))
        self.vjps[Z] = lambda gg, cache, ZZ, YY: (softmax(ZZ.output) - YY.output) * gg / YY.output.shape[1]
        self.vjps[Y_ohe] = lambda gg, cache, ZZ, YY: None

class L2Loss(ComputationNode):
    """
    An L2 loss computation node.

    Parameters
    ------------
    Z : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation. Note that z.output must 
        either be an ndarray column vector or an ndarray m x n matrix, where the 
        softmax is performed column-wise in the matrix (i.e, each data point is stored
        as a column vector in z.output).
    Y : InputNode 
        The parent node from which to make the computation. This node corresponds to 
        the data values associated with Z.output and therefore Y must have the same 
        dims as Z.
    graph : ComputationGraph instance or "default", default="default"
        The graph to which the node is assigned.  If "default", the node is assigned to
        the currently active global default graph.
    """
    def __init__(self, Z, Y):
        super().__init__([Z, Y])
        self.func = lambda ZZ, YY: output_container(l2loss(ZZ.output, YY.output))
        self.vjps[Z] = lambda gg, cache, ZZ, YY: (ZZ.output - YY.output) * gg / YY.output.shape[1]
        self.vjps[Y] = lambda gg, cache, ZZ, YY: None

class L2RegLoss(ComputationNode):
    """
    A computation node that applies L2 regularization on provided weight nodes.

    Parameters
    ------------
    Ws : ParameterNodes 
        All parameter nodes to be regularized.
    lmbda : float, default = .1
        The L2 regularization strength.
    graph : ComputationGraph instance or "default", default="default"
        The graph to which the node is assigned.  If "default", the node is assigned to
        the currently active global default graph.
    """ 
    def __init__(self, *Ws, lmbda=.1):
        super().__init__(list(Ws))   
        self.lmbda = lmbda  
        self.func = lambda *WWs: output_container(l2regloss(*(W.output for W in WWs), lmbda=lmbda))
        self.vjps = {W: lambda gg, cache, *all_parents, WW=W: lmbda * WW.output * gg for W in Ws}