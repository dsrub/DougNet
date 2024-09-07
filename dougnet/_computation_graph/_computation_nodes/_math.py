import numpy as np
from dougnet._computation_graph._node_base import ComputationNode, output_container


class Sqr(ComputationNode):
    """An element-wise square computation node."""
    def __init__(self, x):
        super().__init__([x])
        self.func = lambda xx: output_container(xx.output ** 2)
        self.vjps[x] = lambda gg, cache, xx: 2 * xx.output * gg

class Sqrt(ComputationNode):
    """An element-wise sqaure root computation node."""
    def __init__(self, x):
        super().__init__([x])
        self.func = lambda xx: output_container(np.sqrt(xx.output)) 
        self.vjps[x] = lambda gg, cache, xx: gg / (2. * np.sqrt(xx.output))

class Cos(ComputationNode):
    """An element-wise cosine computation node."""
    def __init__(self, x):
        super().__init__([x])
        self.func = lambda xx: output_container(np.cos(xx.output))
        self.vjps[x] = lambda gg, cache, xx: -np.sin(xx.output) * gg

class Exp(ComputationNode):
    """An element-wise exponentiation computation node."""
    def __init__(self, x):
        super().__init__([x])
        self.func = lambda xx: output_container(np.exp(xx.output))
        self.vjps[x] = lambda gg, cache, xx: np.exp(xx.output) * gg
        
class Sum(ComputationNode):
    """
    A summation computation node.  axis is one of None (indicating a sum over 
    all elements), 0 or 1.  If axis=0 or 1, x.output must be 2 dimensional.
    """
    def __init__(self, x, axis=None):
        super().__init__([x])
        self.func = lambda xx: output_container(np.sum(xx.output, axis=axis))
        if (axis is None) or (axis == 0):
            self.vjps[x] = lambda gg, cache, xx: gg + np.zeros_like(xx.output)
        else:
            self.vjps[x] = lambda gg, cache, xx: gg.reshape(-1, 1) + np.zeros_like(xx.output)
            
class Transpose(ComputationNode):
    """An transpose computation node."""
    def __init__(self, x, axes):
        super().__init__([x])
        reverse_permutation = [x[1] for x in sorted([(a, i) for i, a in enumerate(axes)])]
        self.func = lambda xx: output_container(xx.output.transpose(axes))
        self.vjps[x] = lambda gg, cache, xx: gg.transpose(reverse_permutation)
