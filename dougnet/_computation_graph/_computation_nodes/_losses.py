from dougnet._computation_graph._node_base import ComputationNode, output_container
from dougnet.functions._activations import *
from dougnet.functions._losses import *


class SoftmaxCrossEntropyLoss(ComputationNode):
    """
    A softmax cross-entropy loss computation node.
    """
    def __init__(self, Z, Y_ohe):
        super().__init__([Z, Y_ohe])
        self.func = lambda ZZ, YY: output_container(softmax_cross_entropy_loss(ZZ.output, YY.output))
        self.vjps[Z] = lambda gg, cache, ZZ, YY: (softmax(ZZ.output) - YY.output) * gg / YY.output.shape[1]
        self.vjps[Y_ohe] = lambda gg, cache, ZZ, YY: None

class L2Loss(ComputationNode):
    """An L2 loss computation node."""
    def __init__(self, Z, Y):
        super().__init__([Z, Y])
        self.func = lambda ZZ, YY: output_container(l2loss(ZZ.output, YY.output))
        self.vjps[Z] = lambda gg, cache, ZZ, YY: (ZZ.output - YY.output) * gg / YY.output.shape[1]
        self.vjps[Y] = lambda gg, cache, ZZ, YY: None

class L2RegLoss(ComputationNode):
    """A computation node that applies L2 regularization on provided weight nodes.""" 
    def __init__(self, *Ws, lmbda=.1):
        super().__init__(list(Ws))   
        self.lmbda = lmbda  
        self.func = lambda *WWs: output_container(l2regloss(*(W.output for W in WWs), lmbda=lmbda))
        self.vjps = {W: lambda gg, cache, *all_parents, WW=W: lmbda * WW.output * gg for W in Ws}