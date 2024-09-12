import numpy as np
from dougnet._computation_graph._node_base import ComputationNode, output_container
from dougnet._computation_graph._parameter_nodes import WeightNode
from dougnet.functions._embed import embed, _dW_embed


def _cat_func(*Xs, axis=0):
    cumul_segment_sizes = np.cumsum(np.array([0] + [X.shape[axis] for X in Xs]))
    Xs_concat = np.concatenate(Xs, axis=axis)
    return Xs_concat, cumul_segment_sizes

def _cat_vjp(g, cumul_segment_sizes, i, axis):
    slice_oject = [slice(None, None, None)] * g.ndim
    slice_oject[axis] = slice(cumul_segment_sizes[i], cumul_segment_sizes[i + 1], None)
    slice_oject = tuple(slice_oject)
    return g[slice_oject].copy()

class Cat(ComputationNode):
    """A concatenation computation node."""
    def __init__(self, *Xs, axis=0):
        self.axis = axis
        super().__init__(list(Xs))
        self.func = lambda *XXs: output_container(*_cat_func(*(X.output for X in XXs), axis=axis))
        self.vjps = {Xs[i]: lambda gg, cache, *all_parents, jj=i: _cat_vjp(gg, cache, jj, axis=axis) 
                     for i in range(len(Xs))}

def _flatten(x, num_dims):
    orig_shape = list(x.shape)
    flattened_shape = [np.prod(orig_shape[:num_dims])] + orig_shape[num_dims:]
    x_flat = x.reshape(flattened_shape)
    return x_flat, orig_shape

class Flatten(ComputationNode):
    """
    A flatten computation node.  Flatten the first num_dims dimensions.  For example, 
    if X.output is of shape [D1 x D2 x D3 x D4], then Flatten(X, num_dims=3).output is 
    of shape [D1 * D2 * D3 x D4].
    """
    def __init__(self, X, num_dims=2):
        super().__init__([X])
        self.func = lambda XX: output_container(*_flatten(XX.output, num_dims))
        self.vjps[X] = lambda gg, cache, XX: gg.reshape(cache)
    
class Dropout(ComputationNode):
    """A dropout computation node."""
    def __init__(self, X, p=.5, random_state=None):
        super().__init__([X])
        self.p = p
        self.random_state = random_state
        if (type(self.random_state) == int) or self.random_state is None:
            self.random_state = np.random.RandomState(self.random_state)
            
        self.func = lambda XX: output_container(*self._func(XX.output))
        self.vjps[X] = lambda gg, cache, XX: cache * gg

    def _func(self, x):
        if not self.graph.eval_mode:
            mask = (self.random_state.binomial(1, 1-self.p, size=x.shape) / (1 - self.p)).astype(x.dtype)
            return mask * x, mask
        else:
            return x, None
     
class Embedding(ComputationNode):
    """
    An embedding computation node.  See the embed fuction in dn.functions for 
    further documentation.
    """
    @classmethod
    def instantiate_weight(cls, embedding_dim, cardinality):
        """a convenience class method to instantiate the embedding weight node"""       
        return WeightNode(embedding_dim, cardinality, initializer="normal")
    
    def __init__(self, x, weight):
        self.x = x
        self.weight = weight
        super().__init__([x, weight])
        self.func = lambda xx, ww: output_container(embed(xx.output, ww.output))
        self.vjps[x] = None
        self.vjps[weight] = lambda gg, cache, xx, ww: _dW_embed(gg, xx.output, ww.output)
        
    