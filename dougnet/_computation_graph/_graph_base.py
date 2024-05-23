from collections import namedtuple
import numpy as np

OUTPUT = namedtuple('Output', ('output', 'cache'), defaults=(None, ()))


class ComputationNode:
    """ 
    A node class for the ComputationGraph data structure.  A ComputationNode takes in
    one or more parent nodes and uses the output of these nodes to compute its own output.

    Parameters
    -----------
    parents : list with entries of type ComputationNode, InputNode or ParameterNode.
        Parent nodes.
    """
    def __init__(self, parents=[]):
        self.parents = parents
        self.children = []
        self.output = None
        self.func = None
        self.vjps = {}
        self.graph = _default_graph
            
        # Append this node to the children lists of all parent nodes
        for parent in parents:
            parent.children.append(self)

        # Add this node to currently active default graph
        self.graph.computations.append(self)
        self.graph._topo_add(self)
        
    def compute(self):
        """ 
        Compute the output associated with this node from the outputs of the parent 
        nodes and store in the output attribute.
        """
        self.output, self.cache = self.func(*self.parents)
    
    def VJP(self, parent, g):
        """ 
        Compute the vector-Jacobian product associated with this node for a 
        specified parent node given a gradient tensor.

        Parameters
        -----------
        parent : ComputationNode, InputNode or ParameterNode.
            Parent node for which to compute the VJP. 
        g : ndarray or float
            The gradient of the loss wrt. n, dL/dn.

        Returns:
        ----------
        the VJP as an ndarray
        """
        return self.vjps[parent](g, self.cache, *self.parents)
    
    def forward(self):
        """
        Run forward method in graph data structure up until this node and return output.
        """        
        return self.graph.forward(self)
    
    def backward(self):
        """
        Run backward method in graph data structure up until this node
        """        
        self.graph.backward(self)
    
    def __add__(self, other):
        """element-wise addition (incorporates broadcasting)"""
        return Add(self, other)
    
    def __sub__(self, other):
        """element-wise subtraction"""
        return Subtract(self, other)
    
    def __mul__(self, other):
        """element-wise multiplication"""
        return Mult(self, other)
    
    def __pow__(self, n):
        """element-wise exponentiation"""
        return Power(self, n)


class InputNode:
    """ 
    A node class for the ComputationGraph data structure.  An InputNode typically 
    stores the data for the model and feeds its output (the data) to a computational node.

    Parameters
    -----------
    value : data type depends on the model (typically an ndarray)
        The data to be stored in this node (default is None).
    """
    def __init__(self, value=None):
        self.children = []
        self.output = value
        self.graph = _default_graph

        # Add this node to currently active default graph
        self.graph.inputs.append(self)
        self.graph._topo_add(self)
        
    def __add__(self, other):
        """element-wise addition (incorporates broadcasting)"""
        return Add(self, other)
    
    def __sub__(self, other):
        """element-wise subtraction"""
        return Subtract(self, other)
    
    def __mul__(self, other):
        """element-wise multiplication"""
        return Mult(self, other)
    
    def __pow__(self, n):
        """element-wise exponentiation"""
        return Power(self, n)

        
class ParameterNode:
    """ 
    A node class for the ComputationGraph data structure.  An ParameterNode typically 
    stores the parameters of the model, such as weight matrices and bias vectors. 

    Parameters
    -----------
    value : data type depends on the model (typically an ndarray)
        The parameter to be stored in this node (default is None).
    """
    def __init__(self, value=None):
        self.children = []
        self.output = value
        self.graph = _default_graph

        # Add this node to currently active default graph
        self.graph.parameters.append(self)
        self.graph._topo_add(self)
        
    def __add__(self, other):
        """element-wise addition (incorporates broadcasting)"""
        return Add(self, other)
    
    def __sub__(self, other):
        """element-wise subtraction"""
        return Subtract(self, other)
    
    def __mul__(self, other):
        """element-wise multiplication"""
        return Mult(self, other)
    
    def __pow__(self, n):
        """element-wise exponentiation"""
        return Power(self, n)


class ComputationGraph:
    """
    A computational graph data structure.
    
    Author: Douglas Rubin
    """
    def __init__(self, default=True):
        self.topo_sorted_nodes = []
        self.computations = []
        self.parameters = []
        self.inputs = []
        if default:
            self.as_default()
        
    def as_default(self):
        global _default_graph
        _default_graph = self
        
    def initialize_params(self, seed=None):
        # initialize all model parameters
        random_state = np.random.RandomState(seed)
        for parameter in self.parameters:
            parameter.initialize(random_state)
        
    def _topo_add(self, node):
        # add node to list and re-sort list
        self.topo_sorted_nodes.append(node)
        _TopologicalSort(self.topo_sorted_nodes)
        
    def forward(self, desired_node):
        """
        Compute the forward pass to populate all outputs in each computation node.
        Return the output of the last node in the DAG.
        """
        for node in self.topo_sorted_nodes:
            if isinstance(node, ComputationNode):
                node.compute()
            if node == desired_node:
                return desired_node.output
    
    def backward(self, desired_node):
        """
        Run backward pass to compute gradients of all parameter and computation
        nodes up until a desired node in the graph.  Note that the desired node must 
        have scalar output.  
        """
        self.grads_ = {}
        
        ancestor_of_desired_node = False
        for node in reversed(self.topo_sorted_nodes):
            if node == desired_node:
                self.grads_[desired_node] = 1
                ancestor_of_desired_node = True
            elif ancestor_of_desired_node and not isinstance(node, InputNode):
                # only require grads of parameter nodes and computation nodes
                self.grads_[node] = sum(child.VJP(node, self.grads_[child]) 
                                        for child in node.children
                                        )
        
        
############ graph algo utility funcs for ComputationGraph
def _TopologicalSortDFS(node, already_visited, topo_sorted_nodes):
    """Recursive util function for _TopologicalSort."""
    already_visited.add(node) 
       
    for child in node.children:
        if child not in already_visited:
            _TopologicalSortDFS(child, already_visited, topo_sorted_nodes)
    topo_sorted_nodes.append(node)
   
def _TopologicalSort(nodes):
    """
    Funciton to topologically sort the computation graph in-place.  This function 
    assumes the graph is a DAG (i.e., it does not detect cycles).
    """
    topo_sorted_nodes = []
    already_visited = set()
    
    for node in nodes:
        if node not in already_visited:
            _TopologicalSortDFS(node, already_visited, topo_sorted_nodes)
            
    # modify nodes list in-place
    for i, node in enumerate(reversed(topo_sorted_nodes)):
        nodes[i] = node
        
        
############ computation nodes used in the magic methods in ComputationNode
# nodes defined in this file rather than in a _computational.py node file 
# to avoid circular imports
_shape = lambda x: (1,) if type(x) != np.ndarray else x.shape 
class Add(ComputationNode):
    """A broadcastable element-wise addition node.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode  
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray or a float).
    y : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation. Note that y.output 
        must be the same dimensions as x.output, or x.output must be an m x n matrix
        and y.output must be a length m column vector to be broadcasted to m x n then 
        added with x.output.
    """
    def __init__(self, x, y):
        super().__init__([x, y])
        self.func = lambda xx, yy: OUTPUT(xx.output + yy.output)
        self.vjps[x] = lambda gg, cache, xx, yy: gg
        self.vjps[y] = lambda gg, cache, xx, yy: gg if _shape(xx.output) == _shape(yy.output) \
            else np.sum(gg, axis=1).reshape(gg.shape[0], 1)

            
class Subtract(ComputationNode):
    """An element-wise subtraction node.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode  
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray or a float).
    y : ComputationNode, InputNode or ParameterNode  
        The parent node from which to make the computation. Note that y.output 
        must be the same dimensions as x.output.
    """
    def __init__(self, x, y):
        super().__init__([x, y])
        self.func = lambda xx, yy: OUTPUT(xx.output - yy.output)
        self.vjps[x] = lambda gg, cache, xx, yy: gg
        self.vjps[y] = lambda gg, cache, xx, yy: -gg
        
class Mult(ComputationNode):
    """An element-wise multiplication (hadamard multiplication) computation node.  Note 
    that if the output of the two input nodes are scalars, then this node performs normal, 
    scalar multiplication.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode  
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray or a float).
    y : ComputationNode, InputNode or ParameterNode  
        The parent node from which to make the computation. Note that y.output 
        must be the same dimensions as x.output.
    """
    def __init__(self, x, y):
        super().__init__([x, y])
        self.func = lambda xx, yy: OUTPUT(xx.output * yy.output)
        self.vjps[x] = lambda gg, cache, xx, yy: yy.output * gg
        self.vjps[y] = lambda gg, cache, xx, yy: xx.output * gg
        
        
class Power(ComputationNode):
    """An element-wise square computation node.

    Parameters
    ------------
    x : ComputationNode, InputNode or ParameterNode 
        The parent node from which to make the computation (x.output can 
        be an arbitrary dim. ndarray or a float)
    n : float
        Exponent
    """
    def __init__(self, x, n):
        super().__init__([x])
        self.func = lambda xx: OUTPUT(xx.output ** n)
        self.vjps[x] = lambda gg, cache, xx: gg * n * (xx.output ** (n - 1))