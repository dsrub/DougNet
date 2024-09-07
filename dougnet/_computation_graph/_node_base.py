import numpy as np
from collections import namedtuple

output_container = namedtuple('Output', ('output', 'cache'), defaults=(None, ()))


class Node:
    """A node base class."""
    def __init__(self, value=None):
        self.children = []
        self.output = value
        
        # import the currently active default graph and add node to graph
        from dougnet._computation_graph._graph_base import _default_graph
        self.graph = _default_graph
        self.graph._add_node(self) 
        
    def __add__(self, other):
        """element-wise addition (incorporates broadcasting if applicable)"""
        other = other if isinstance(other, Node) else InputNode(other)
        return Add(self, other)
        
    def __sub__(self, other):
        """element-wise subtraction"""
        other = other if isinstance(other, Node) else InputNode(other)
        return Subtract(self, other)
    
    def __mul__(self, other):
        """element-wise multiplication"""
        other = other if isinstance(other, Node) else InputNode(other)
        return Mult(self, other)
    
    def __truediv__(self, other):
        """element-wise division"""
        other = other if isinstance(other, Node) else InputNode(other)
        return Div(self, other)
    
    def __pow__(self, n):
        """element-wise exponentiation"""
        return Power(self, n)
    
    def __matmul__(self, other):
        """matrix multiplication"""
        return MatMult(self, other)
    
    def __radd__(self, other):
        """other + self"""
        return self + other
    
    def __rsub__(self, other):
        """other - self"""
        other = other if isinstance(other, Node) else InputNode(other)
        return Subtract(other, self)
    
    def __rmul__(self, other):
        """other * self"""
        return self * other
    
    def __rtruediv__(self, other): 
        """other / self"""
        other = other if isinstance(other, Node) else InputNode(other)
        return Div(other, self)
    
    def __getitem__(self, slice_items):
        """slicing/indexing"""
        return Slice(self, slice_items)
    
    def __repr__(self):
        with np.printoptions(precision=3, suppress=True):
            prefix = f"Node(\noutput=\n{self.output}"
            if self.output is None or type(self.output) == int:
                prefix = f"Node(output={self.output}"    
            if hasattr(self, "parents"):
                prefix += f",\nparents={[type(p).__name__ for p in self.parents]}"
            return prefix + f",\nchildren={[type(p).__name__ for p in self.children]})"
    
    
class InputNode(Node):
    """ 
    A node class for the ComputationGraph data structure.  An InputNode typically stores
    inputs which do not require gradients, such as constants or training data.

    Parameters
    -----------
    value : np.ndarray or float
        The data to be stored in this node (default is None).
    """
    def __init__(self, value=None):
        super().__init__(value=value)

        
class ParameterNode(Node):
    """ 
    A node class for the ComputationGraph data structure.  An ParameterNode typically 
    stores the parameters of the model, such as weight tensors and bias vectors. 

    Parameters
    -----------
    value : typically a numpy ndarray
        The parameter to be stored in this node (default is None).
    """
    def __init__(self, value=None):
        super().__init__(value=value)
    
    
class ComputationNode(Node):
    """ 
    A node class for the ComputationGraph data structure.  A ComputationNode takes in
    one or more parent nodes and uses the output of these nodes to compute its own output.

    Parameters
    -----------
    parents : list with entries of type Node
        Parent nodes.
    """
    def __init__(self, parents=[]):
        self.parents = parents
        self.func = None
        self.vjps = {}
        
        # append this node to the children lists of all parent nodes
        for parent in parents:
            parent.children.append(self)
        super().__init__()
        
    def compute(self):
        """Compute output associated with this node and store in the output attribute."""
        self.output, self.cache = self.func(*self.parents)
    
    def VJP(self, parent, g):
        """ 
        Compute the vector-Jacobian product associated with this node for a specified 
        parent node, parent, given a gradient tensor, g.
        """
        return self.vjps[parent](g, self.cache, *self.parents)
    
    def forward(self):
        """Run forward method in graph data structure up until this node and return output."""        
        return self.graph.forward(self)
    
    def backward(self):
        """Run backward method in graph data structure up until this node."""        
        self.graph.backward(self)


# DEFINE VARIOUS COMPUTATION NODES USED IN MAGIC METHODS IN THE Node BASE CLASS 
# DEFINE HERE RATHER THAN IN THE _computation_nodes FOLDER TO AVOID CIRCULAR IMPORT ISSUES
_shape = lambda x: (1,) if type(x) != np.ndarray else x.shape 
class Add(ComputationNode):
    """
    A broadcastable element-wise addition computation node.  If y is to be broadcasted 
    to the shape of x in the addition, it must conform to the numpy rules of broadcasting.
    """
    def __init__(self, x, y):
        super().__init__([x, y])
        self.func = lambda xx, yy: output_container(xx.output + yy.output)
        self.vjps[x] = lambda gg, cache, xx, yy: gg
        self.vjps[y] = lambda gg, cache, xx, yy: gg if _shape(xx.output) == _shape(yy.output) \
            else np.sum(gg, axis=1).reshape(gg.shape[0], 1)

class Subtract(ComputationNode):
    """An element-wise subtraction computation node."""
    def __init__(self, x, y):
        super().__init__([x, y])
        self.func = lambda xx, yy: output_container(xx.output - yy.output)
        self.vjps[x] = lambda gg, cache, xx, yy: gg
        self.vjps[y] = lambda gg, cache, xx, yy: -gg
        
class Mult(ComputationNode):
    """An element-wise multiplication (hadamard multiplication) computation node."""
    def __init__(self, x, y):
        super().__init__([x, y])
        self.func = lambda xx, yy: output_container(xx.output * yy.output)
        self.vjps[x] = lambda gg, cache, xx, yy: yy.output * gg
        self.vjps[y] = lambda gg, cache, xx, yy: xx.output * gg
        
class Div(ComputationNode):
    """An element-wise division (hadamard division) computation node."""
    def __init__(self, x, y):
        super().__init__([x, y])
        self.func = lambda xx, yy: output_container(xx.output / yy.output)
        self.vjps[x] = lambda gg, cache, xx, yy: (yy.output ** -1)  * gg
        self.vjps[y] = lambda gg, cache, xx, yy: -xx.output * (yy.output ** -2)  * gg
        
class Power(ComputationNode):
    """An element-wise exponentiation computation node."""
    def __init__(self, x, n):
        super().__init__([x])
        self.func = lambda xx: output_container(xx.output ** n)
        self.vjps[x] = lambda gg, cache, xx: gg * n * (xx.output ** (n - 1))
        
class MatMult(ComputationNode):
    """A matrix multiplication computation node."""
    def __init__(self, x, y):
        super().__init__([x, y])
        self.func = lambda xx, yy: output_container(np.dot(xx.output, yy.output))
        self.vjps[x] = lambda gg, cache, xx, yy: np.dot(gg, yy.output.T)
        self.vjps[y] = lambda gg, cache, xx, yy: np.dot(xx.output.T, gg)
        
def _vjp_slice(g, x, slice_items):
    dx = np.zeros(x.shape, dtype=x.dtype)
    dx.__setitem__(slice_items, g)
    return dx

class Slice(ComputationNode):
    """A slice computation node. Supports basic slicing/indexing, but not well tested."""
    def __init__(self, x, slice_items):
        super().__init__([x])
        self.func = lambda xx: output_container(xx.output[slice_items])
        self.vjps[x] = lambda gg, cache, xx: _vjp_slice(gg, xx.output, slice_items)
        
