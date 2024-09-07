from dougnet._computation_graph._node_base import ComputationNode, InputNode
import numpy as np

       
class ComputationGraph:
    """A computational graph data structure."""
    def __init__(self, default=True):
        self.computations = []
        self.parameters = []
        self.inputs = []
        self.topo_sorted = True
        self.eval_mode = False
        if default:
            self.as_default()
            
    def _add_node(self, node):
        if isinstance(node, ComputationNode):
            self.computations.append(node)
            self.topo_sorted = False
        elif isinstance(node, InputNode):
            self.inputs.append(node)
        else:
            self.parameters.append(node)
        
    def as_default(self):
        """make default graph"""
        global _default_graph
        _default_graph = self
        
    def eval(self):
        """set to evaluation/inference mode"""
        self.eval_mode = True
        
    def train(self):
        """set to training mode"""
        self.eval_mode = False
        
    def initialize_params(self, seed=None):
        """ Initialize all parameter nodes in graph in order in which they were added to graph."""
        random_state = np.random.RandomState(seed)
        for parameter in self.parameters:
            if hasattr(parameter, "initialize") and parameter.output is None:
                parameter.initialize(random_state)
            
    def forward(self, desired_node):
        """
        Compute the forward pass to populate all outputs in each computation node up until
        desired_node and return desired_node.output.  desired_node should be a computation 
        node.
        """
        self._TopologicalSort()
        for node in self.computations:
            node.compute()
            if node == desired_node:
                return desired_node.output
    
    def backward(self, desired_node):
        """
        Run backward pass to compute gradients of all parameter and computation nodes up 
        until a desired node in the graph.  Note that the desired node must have scalar output. 
        """
        self._TopologicalSort()
        
        self.grads_ = {}
        ancestor_of_desired_node = False
        for node in reversed(self.parameters + self.computations):
            if node == desired_node:
                self.grads_[desired_node] = 1
                ancestor_of_desired_node = True
            elif ancestor_of_desired_node:
                self.grads_[node] = sum(child.VJP(node, self.grads_[child]) 
                                        for child in node.children)
                
    def delete_nonparams(self):
        """ Delete all input nodes and computation nodes from graph (useful for RNNs)."""
        # children of a parameter node can only be computation nodes so delete all children
        for node in self.parameters:
            node.children = [] 
        self.inputs = []
        self.computations = []
        self.topo_sorted = True
                
    def _TopologicalSort(self):
        """
        Function to topologically sort the computation graph in-place.  This function 
        assumes the graph is a DAG (i.e., it does not detect cycles).
        """
        if self.topo_sorted:
            return
        
        topo_sorted_nodes = []
        already_visited = set()
        for node in self.computations:
            if node not in already_visited:
                self._TopologicalSortDFS(node, already_visited, topo_sorted_nodes)
                
        # modify nodes list in-place
        for i, node in enumerate(reversed(topo_sorted_nodes)):
            self.computations[i] = node
        self.topo_sorted = True
        
    @staticmethod
    def _TopologicalSortDFS(node, already_visited, topo_sorted_nodes):
        """Recursive util function for _TopologicalSort."""
        already_visited.add(node) 
        for child in node.children:
            if child not in already_visited:
                ComputationGraph._TopologicalSortDFS(child, already_visited, topo_sorted_nodes)
        topo_sorted_nodes.append(node)
        
    def __len__(self):
        """number of scalar parameters associated with graph"""
        if any(param.output is None for param in self.parameters):
            return 0
        return sum(1 if type(param) == int else param.output.size for param in self.parameters)