import numpy as np
from dougnet._computation_graph._graph_base import ComputationNode, output_container


class Sqr(ComputationNode):
    """An element-wise square computation node."""
    def __init__(self, x):
        super().__init__([x])
        self.func = lambda xx: output_container(xx.output ** 2)
        self.vjps[x] = lambda gg, cache, xx: 2 * xx.output * gg

class MatMult(ComputationNode):
    """A matrix multiplication computation node."""
    def __init__(self, x, y):
        super().__init__([x, y])
        self.func = lambda xx, yy: output_container(np.dot(xx.output, yy.output))
        self.vjps[x] = lambda gg, cache, xx, yy: np.dot(gg, yy.output.T)
        self.vjps[y] = lambda gg, cache, xx, yy: np.dot(xx.output.T, gg)

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
    """A computation node summing all elements of the input tensor."""
    def __init__(self, x):
        super().__init__([x])
        self.func = lambda xx: output_container(np.sum(xx.output))
        self.vjps[x] = lambda gg, cache, xx: np.zeros_like(xx.output) + gg