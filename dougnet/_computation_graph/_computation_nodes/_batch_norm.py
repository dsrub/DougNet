import numpy as np
from dougnet._computation_graph._graph_base import ComputationNode, output_container
from dougnet._computation_graph._parameter_nodes import WeightNode, BiasNode
from dougnet.nn_functions._batch_norm import bn1d, _grads_bn1d


class BN1d(ComputationNode):
    """
    A 1d batch norm computation node.

    Parameters
    ------------
    Z : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation. It is assumed Z.output is an 
        m x |B| matrix with m "features" and |B| minibatch samples (i.e., each minibatch
        sample is a column vector in the design matrix).
    num_features : int
        The number of "features" in the input matrix.  Should equal Z.output.shape[0].
    alpha : float, default=.1
        The smoothing factor for computing the exponentially weighted moving average of the
        mean and variance statistics.
    eps : float, default=1e-5
        A safety constant added to the variance for numerical stability.
    dtype : np.dtype, default=np.float32
        The datatype with which to initialize gamma, beta, running_mean and running_var.
    graph : ComputationGraph instance or "default", default="default"
        The graph to which the node is assigned.  If "default", the node is assigned to
        the currently active global default graph.

    Notes: 
    ------
    In keeping with pytorch, the minibatch variance uses the standard biased estimate in 
    training, while running_var, which is used for inference, is computed with the standard
    unbiased estimate.
    """
    def __init__(self, Z, num_features, alpha=.1, eps=1e-5, dtype=np.float32):        
        self._grads_cache = None
        self.alpha = alpha
        self.eps = eps
        
        # instantiate gamma and beta and add to the graph
        self.gamma = WeightNode(num_features, 1, dtype=dtype, initializer="ones")
        self.beta = BiasNode(num_features, dtype=dtype, initializer="zeros")
        super().__init__([Z, self.gamma, self.beta])
        
        # intialize running mean and running variance
        self.running_mean = np.zeros((num_features, 1), dtype=dtype)
        self.running_var = np.ones((num_features, 1), dtype=dtype)

        # define forward function for parent class
        self.func = lambda ZZ, ggamma, bbeta: output_container(*self._func(ZZ.output, 
                                                                           ggamma.output, 
                                                                           bbeta.output))
        
        # define vjps for parent class
        self.vjps[self.gamma] = lambda gg, cache, ZZ, ggamma, bbeta: self._grads(0, gg, 
                                                                                 cache[0], 
                                                                                 ggamma.output, 
                                                                                 cache[1])
        self.vjps[self.beta] = lambda gg, cache, ZZ, ggamma, bbeta: self._grads(1, gg, 
                                                                                cache[0], 
                                                                                ggamma.output, 
                                                                                cache[1])
        self.vjps[Z] = lambda gg, cache, ZZ, ggamma, bbeta: self._grads(2, gg, 
                                                                        cache[0], 
                                                                        ggamma.output, 
                                                                        cache[1])
        
    def _func(self, Z, gamma, beta):
        """helper function for the self.func attribute"""
        if not self.graph.eval_mode:
            # run forward pass
            Z_BN, forward_cache = bn1d(Z, gamma, beta, eps=self.eps, return_cache=True)

            # update running statistics (use un-biased estimate of variance for the 
            # running variance statistic just like pytorch does)
            _, _, mu, var = forward_cache
            self.running_mean *= (1 - self.alpha)
            self.running_mean += self.alpha * mu
            self.running_var *= (1 - self.alpha)
            self.running_var += self.alpha * var * Z.shape[1] / (Z.shape[1] - 1)
            
            # reset _grads_cache to None so that the gradients will be computed fresh during 
            # the subsequent backward pass
            self._grads_cache = None
        else:
            # compute Z_BN with running statistics
            Z_prime = (Z - self.running_mean) / np.sqrt(self.running_var + self.eps)
            Z_BN = gamma * Z_prime + beta
            forward_cache = None
        
        return Z_BN, forward_cache       
        
    def _grads(self, x, dZ_BN, Z_prime, gamma, sigma):
        """helper function for the self.vjps attribute"""
        # return gradient immediately from cached if already computed
        if self._grads_cache is not None:
            return self._grads_cache[x]
        
        # otherwise, compute gradients, cache them and return required gradient
        self._grads_cache = _grads_bn1d(dZ_BN, Z_prime, gamma, sigma)
        return self._grads_cache[x]