import numpy as np
from numba import njit, prange
from dougnet._computation_graph._node_base import ComputationNode, output_container
from dougnet._computation_graph._parameter_nodes import WeightNode
from dougnet.functions import bn1d, bn2d
from dougnet.functions._batch_norm._batch_norm_main import _grads_bn


bn_docstring = """
A {0} batch norm computation node.

Parameters
------------
Z : Node
    The parent node from which to make the computation. It is assumed Z.output is an 
    {1} tensor with {2} "features" and {3} minibatch samples.
num_features : int
    The number of "features" in the input tensor.
alpha : float, default=.1
    The smoothing factor for computing the exponentially weighted moving average of the
    mean and variance statistics.
eps : float, default=1e-5
    A safety constant added to the variance for numerical stability.
dtype : np.dtype, default=np.float32
    The datatype with which to initialize gamma, beta, running_mean and running_var.
parallel : bool, default=True
    Compute batch norm operation with multi-threading.

Notes: 
------
In keeping with pytorch, the minibatch variance uses the standard biased estimate in 
training, while running_var, which is used for inference, is computed with the standard
unbiased estimate.
"""


# DEFINE FUNCS TO PERFORM INFERENCE FOR BN1d AND BN2d
@njit(parallel=True)
def _inference1d_parallel(Z, mu, var, gamma, beta, eps):
    C, N = Z.shape
    sigma = np.sqrt(var + eps)
    Z_BN = np.empty_like(Z)
    for c in prange(C):
        mu_c = mu[c]
        gamma_over_sigma_c = gamma[c] / sigma[c]
        beta_c = beta[c]
        for n in range(N):
            Z_BN[c, n] = gamma_over_sigma_c * (Z[c, n] - mu_c) + beta_c            
    return Z_BN

@njit(parallel=True)
def _inference2d_parallel(Z, mu, var, gamma, beta, eps):
    N, C, H, W = Z.shape
    sigma = np.sqrt(var + eps)
    Z_BN = np.empty_like(Z)
    for c in prange(C):
        mu_c = mu[c]
        gamma_over_sigma_c = gamma[c] / sigma[c]
        beta_c = beta[c]
        for n in range(N):
            for i in range(H):
                for j in range(W):
                    Z_BN[n, c, i, j] = gamma_over_sigma_c * (Z[n, c, i, j] - mu_c) + beta_c            
    return Z_BN

def _inference1d(Z, mu, var, gamma, beta, eps, parallel):
    if parallel:
        return _inference1d_parallel(Z, mu, var, gamma, beta, eps)
    Z_prime = (Z - mu.reshape(-1, 1)) / np.sqrt(var.reshape(-1, 1) + eps)
    Z_BN = gamma.reshape(-1, 1) * Z_prime + beta.reshape(-1, 1)
    return Z_BN

def _inference2d(Z, mu, var, gamma, beta, eps, parallel):
    if parallel:
        return _inference2d_parallel(Z, mu, var, gamma, beta, eps)
    Z_prime = (Z - mu.reshape(-1, 1, 1)) / np.sqrt(var.reshape(-1, 1, 1) + eps)
    Z_BN = gamma.reshape(-1, 1, 1) * Z_prime + beta.reshape(-1, 1, 1)
    return Z_BN


# DEFINE MAIN BASE CLASS
class _BNXd(ComputationNode):
    """A batch norm base class.  Should not be accessed by user."""
    def __init__(self, Z, num_features, alpha, eps, dtype, parallel, forward_func, inference_func):        
        self._grads_cache = None
        self.num_features = num_features
        self.alpha = alpha
        self.eps = eps
        self.parallel = parallel
        self._forward_func = forward_func
        self._inference_func = inference_func
        
        # instantiate gamma and beta and add to the graph
        self.gamma = WeightNode(self.num_features, dtype=dtype, initializer="ones")
        self.beta = WeightNode(self.num_features, dtype=dtype, initializer="zeros")
        super().__init__([Z, self.gamma, self.beta])
        
        # intialize running mean and running variance
        self.running_mean = np.zeros(self.num_features, dtype=dtype)
        self.running_var = np.ones(self.num_features, dtype=dtype)

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
            Z_BN, forward_cache = self._forward_func(Z, gamma, beta, eps=self.eps, 
                                                     return_cache=True, parallel=self.parallel)

            # update running statistics (use un-biased estimate of variance for the 
            # running variance statistic just like pytorch does)
            _, _, mu, var = forward_cache
            self.running_mean *= (1 - self.alpha)
            self.running_mean += self.alpha * mu
            self.running_var *= (1 - self.alpha)
            B = Z.shape[1]
            if Z.ndim == 4:
                B = Z.shape[0] * Z.shape[2] *Z.shape[3]
            self.running_var += self.alpha * var * (B / (B - 1))
            
            # reset _grads_cache to None so that the gradients will be computed fresh during 
            # the subsequent backward pass
            self._grads_cache = None
        else:
            # compute Z_BN with running statistics
            Z_BN = self._inference_func(Z, self.running_mean, self.running_var, 
                                        gamma, beta, self.eps, parallel=self.parallel)
            forward_cache = None
        
        return Z_BN, forward_cache       
        
    def _grads(self, x, dZ_BN, Z_prime, gamma, sigma):
        """helper function for the self.vjps attribute"""
        # return gradient immediately from cached if already computed
        if self._grads_cache is not None:
            return self._grads_cache[x]
        
        # otherwise, compute gradients, cache them and return required gradient
        self._grads_cache = _grads_bn(dZ_BN, Z_prime, gamma, sigma, parallel=self.parallel)
        return self._grads_cache[x]
    
    
# DEFINE 1d AND 2d BATCH NORM COMPUTATIONAL NODE CLASSES
class BN1d(_BNXd):
    def __init__(self, Z, num_features, alpha=.1, eps=1e-5, dtype=np.float32, parallel=True):  
        super().__init__(Z, num_features, alpha, eps, dtype, parallel, bn1d, _inference1d) 
BN1d.__doc__ = bn_docstring.format("1d", "m x |B|", "m", "|B|")
    
class BN2d(_BNXd):
    def __init__(self, Z, num_features, alpha=.1, eps=1e-5, dtype=np.float32, parallel=True):  
        super().__init__(Z, num_features, alpha, eps, dtype, parallel, bn2d, _inference2d) 
BN2d.__doc__ = bn_docstring.format("2d", "N x C x H x W", "C", "N")