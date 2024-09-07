from math import sqrt  
import numpy as np

def clip_grads(graph, max_norm, params=None):
    """
    Gradient clipper.

    Parameters
    ----------
    graph : ComputationalGraph
        The computational graph instance associated with the parameters to 
        be clipped.
    max_norm : float
        The maximum allowed L2 norm of the gradients.
    params : None or list of ParameterNodes, default=None
        If provided, only these parameters will be clipped. Otherwise, all 
        parameters in graph.parameters will be clipped.
        
    Notes
    -----
    Should be used right after gradients are calculated and before the optimizer
    updates the parameters.
        
    Example Usage
    -------------
    graph = dn.ComputationGraph()
    ... 
    define graph and run a mini-batch
    ...
    
    L = dn.SoftmaxCrossEntropyLoss(predicted_logits, Y_train)
    _ = L.forward()
    L.backward()
    dn.optim.utils.clip_grads(graph, 1.)
    optim.step() 
    """
    if params is None:
        params = graph.parameters
    
    norm = sqrt(sum(np.sum(graph.grads_[parameter_node] ** 2) 
                    for parameter_node in params))
    if norm > max_norm:
        for parameter_node in params:
            graph.grads_[parameter_node] *= max_norm / norm