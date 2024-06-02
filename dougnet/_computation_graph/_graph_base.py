from collections import namedtuple
import numpy as np

output_container = namedtuple('Output', ('output', 'cache'), defaults=(None, ()))

class Node:
    """
    A node base class with methods and code in the __init__ func common to 
    all 3 types of nodes. Not to be interacted with directly by user.
    """
    def __init__(self, value=None):
        self.children = []
        self.output = value
        self.graph = _default_graph
            
        # Add node to associated graph
        self.graph._add_node(self)
            
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
        

class ComputationNode(Node):
    """ 
    A node class for the ComputationGraph data structure.  A ComputationNode takes in
    one or more parent nodes and uses the output of these nodes to compute its own output.

    Parameters
    -----------
    parents : list with entries of type ComputationNode, InputNode or ParameterNode.
        Parent nodes.
    graph : ComputationGraph instance or "default", default="default"
        The computational graph instance to which this node is associated.  If "default", 
        use the globally active default graph.
    """
    def __init__(self, parents=[]):
        self.parents = parents
        self.func = None
        self.vjps = {}

        # Append this node to the children lists of all parent nodes
        for parent in parents:
            parent.children.append(self)
            
        # run __init__ in base class
        super().__init__()
        
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


class InputNode(Node):
    """ 
    A node class for the ComputationGraph data structure.  An InputNode typically 
    stores the data for the model and feeds its output (the data) to a computational node.

    Parameters
    -----------
    value : typically an ndarray of data
        The data to be stored in this node (default is None).
    graph : ComputationGraph instance or "default", default="default"
        The computational graph instance to which this node is associated.  If "default", 
        use the globally active default graph.
    """
    def __init__(self, value=None):
        super().__init__(value=value)

        
class ParameterNode(Node):
    """ 
    A node class for the ComputationGraph data structure.  An ParameterNode typically 
    stores the parameters of the model, such as weight matrices and bias vectors. 

    Parameters
    -----------
    value : typically an ndarray of weights
        The parameter to be stored in this node (default is None).
    graph : ComputationGraph instance or "default", default="default"
        The computational graph instance to which this node is associated.  If "default", 
        use the globally active default graph.
    """
    def __init__(self, value=None):
        super().__init__(value=value)

       
class ComputationGraph:
    """
    A computational graph data structure.
    """
    def __init__(self, default=True):
        self.topo_sorted_nodes = []
        self.computations = []
        self.parameters = []
        self.inputs = []
        self.eval_mode = False
        if default:
            self.as_default()
            
    def _add_node(self, node):
        if isinstance(node, ComputationNode):
            self.computations.append(node)
        elif isinstance(node, InputNode):
            self.inputs.append(node)
        else:
            self.parameters.append(node)
        
        # add node to topologically sorted list and re-sort list
        self.topo_sorted_nodes.append(node)
        _TopologicalSort(self.topo_sorted_nodes)
        
    def as_default(self):
        global _default_graph
        _default_graph = self
        
    def eval(self):
        self.eval_mode = True
        
    def train(self):
        self.eval_mode = False
        
    def initialize_params(self, seed=None):
        # initialize all model parameters
        random_state = np.random.RandomState(seed)
        for parameter in self.parameters:
            parameter.initialize(random_state)
        
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
        
        
### DEFINE TOPOLOGICAL SORT UTILITY FUNC FOR ComputationGraph CLASS
def _TopologicalSortDFS(node, already_visited, topo_sorted_nodes):
    """Recursive util function for _TopologicalSort."""
    already_visited.add(node) 
       
    for child in node.children:
        if child not in already_visited:
            _TopologicalSortDFS(child, already_visited, topo_sorted_nodes)
    topo_sorted_nodes.append(node)
   
def _TopologicalSort(nodes):
    """
    Function to topologically sort the computation graph in-place.  This function 
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
        
        
### DEFINE VARIOUS COMPUTATION NODES USED IN MAGIC METHODS IN THE Node BASE CLASS 
_shape = lambda x: (1,) if type(x) != np.ndarray else x.shape 

class Add(ComputationNode):
    """
    A broadcastable element-wise addition node. x.output and y.output may be ndarrays 
    or floats.  If y.output is to be broadcasted to the shape of x in the addition, it 
    must conform to the numpy rules of broadcasting.
    """
    def __init__(self, x, y):
        super().__init__([x, y])
        self.func = lambda xx, yy: output_container(xx.output + yy.output)
        self.vjps[x] = lambda gg, cache, xx, yy: gg
        self.vjps[y] = lambda gg, cache, xx, yy: gg if _shape(xx.output) == _shape(yy.output) \
            else np.sum(gg, axis=1).reshape(gg.shape[0], 1)

class Subtract(ComputationNode):
    """An element-wise subtraction node."""
    def __init__(self, x, y):
        super().__init__([x, y])
        self.func = lambda xx, yy: output_container(xx.output - yy.output)
        self.vjps[x] = lambda gg, cache, xx, yy: gg
        self.vjps[y] = lambda gg, cache, xx, yy: -gg
        
class Mult(ComputationNode):
    """
    An element-wise multiplication (hadamard multiplication) computation node.  If 
    x.output and y.output are floats, then this node performs normal, 
    scalar multiplication.
    """
    def __init__(self, x, y):
        super().__init__([x, y])
        self.func = lambda xx, yy: output_container(xx.output * yy.output)
        self.vjps[x] = lambda gg, cache, xx, yy: yy.output * gg
        self.vjps[y] = lambda gg, cache, xx, yy: xx.output * gg
        
class Power(ComputationNode):
    """An element-wise exponentiation computation node."""
    def __init__(self, x, n):
        super().__init__([x])
        self.func = lambda xx: output_container(xx.output ** n)
        self.vjps[x] = lambda gg, cache, xx: gg * n * (xx.output ** (n - 1))