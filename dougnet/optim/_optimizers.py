import numpy as np


class Optimizer():
    """Optimizer base class"""
    def __init__(self, graph, eta=.001, params=None):
        self.graph = graph
        self.eta = eta
        self.params = params
        if self.params is None:
            self.params = graph.parameters
            
    def step(self):
        raise NotImplementedError()
    
class SGD(Optimizer):
    """
    SGD optimizer with optional momentum.

    Parameters
    ------------
    graph : ComputationalGraph
        The computational graph instance associated with the parameters to 
        be optimized.
    eta : float, default=0.001
        Learning rate.
    momentum : float, default=0 
        The momentum parameter.
    params : None or list of ParameterNodes, default=None
        If provided, the optimizer will update these parameters. Otherwise, 
        the optimizer will update all parameter nodes in the provided graph
        instance.
    """
    def __init__(self, graph, eta=.001, momentum=0, params=None):
        super().__init__(graph, eta, params)
        self.momentum = momentum
        self._step_func = self._step_vanilla
        if momentum != 0:
            self._v = {param: np.zeros(param.shape, param.dtype) for param in self.params}
            self._step_func = self._step_momentum
            
    def _step_momentum(self):
        for param in self.params:
            self._v[param] *= self.momentum
            self._v[param] += self.graph.grads_[param]
            param.output -= self.eta * self._v[param]
            
    def _step_vanilla(self):
        for param in self.params:
            param.output -= self.eta * self.graph.grads_[param]
            
    def step(self):
        self._step_func()
        
class Adam(Optimizer):
    """
    Adam optimizer.

    Parameters
    ------------
    graph : ComputationalGraph
        The computational graph instance associated with the parameters to 
        be optimized.
    eta : float, default=0.001
        Learning rate.
    betas: tuple, default=(.9, .999)
        The beta values for the first and second moment updates.
    eps : float, default=1e-8
        Safety parameter in denominator of update to avoid numerical instability.
    params : None or list of ParameterNodes, default=None
        If provided, the optimizer will update these parameters. Otherwise, 
        the optimizer will update all parameter nodes in the provided graph
        instance.
    """
    def __init__(self, graph, eta=.001, betas=(.9, .999), eps=1e-8, params=None):
        super().__init__(graph, eta, params)
        self.beta_1, self.beta_2 = betas
        self.eps = eps
        self._v = {param: np.zeros(param.shape, param.dtype) for param in self.params}
        self._s = {param: np.zeros(param.shape, param.dtype) for param in self.params}
        self._step_cnt = 0
        
    def step(self):
        self._step_cnt += 1
        for param in self.params:
            
            # update first moment
            self._v[param] *= self.beta_1
            self._v[param] += (1 - self.beta_1) * self.graph.grads_[param]
            
            # update second moment
            self._s[param] *= self.beta_2
            self._s[param] += (1 - self.beta_2) * self.graph.grads_[param] ** 2
            
            # de-bias
            v_hat = self._v[param] / (1 - self.beta_1 ** self._step_cnt)
            s_hat = self._s[param] / (1 - self.beta_2 ** self._step_cnt)
            
            # update param
            param.output -= self.eta * v_hat / (np.sqrt(s_hat) + self.eps)