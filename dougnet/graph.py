import numpy as np
from datetime import datetime
from dougnet.training_utils import *


class ComputationNode:
    """ A node class for the ComputationGraph data structure.  A ComputationNode takes in
    one or more parent nodes and uses the output of these nodes to compute its own output.

    Parameters
    -----------
    parents : list with entries of type ComputationNode, InputNode or ParameterNode.
        Parent nodes.
    
    Author: Douglas Rubin

    """
    def __init__(self, parents=[]):
        self.parents = parents
        self.children = []
        self.output = None
        self.computation = None
        self.vjps = {}

        # Append this node to the children lists of all parent nodes
        for parent in parents:
            parent.children.append(self)

        # Add this node to the currently active default graph
        _default_graph.computations.append(self)

    def compute(self):
        """ Compute the output associated with this node from the outputs of the parent 
        nodes and store in the output attribute.

        Returns:
        ----------
        None
        """
        self.output = self.computation(*self.parents)
    
    def VJP(self, parent, g):
        """ Compute the vector-Jacobian product associated with this node (n) for a 
        specified parent node given a gradient tensor (dL/dn).  That is, compute the
        VJP, (dn/dparent)*(dL/dn).

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
        return self.vjps[parent](g, *self.parents)


class InputNode:
    """ A node class for the ComputationGraph data structure.  An InputNode typically 
    stores the data for the model and feeds its output (the data) to a computational node.

    Parameters
    -----------
    value : data type depends on the model (typically an ndarray)
        The data to be stored in this node (default is None).
    
    Author: Douglas Rubin
    """
    def __init__(self, value=None):
        self.children = []
        self.output = value

        # Add this node to the currently active default graph
        _default_graph.inputs.append(self)

        
class ParameterNode:
    """ A node class for the ComputationGraph data structure.  An ParameterNode typically 
    stores the parameters of the model, such as weight matrices and bias vectors. 

    Parameters
    -----------
    value : data type depends on the model (typically an ndarray)
        The parameter to be stored in this node (default is None).
    
    Author: Douglas Rubin
    """
    def __init__(self, value=None):
        self.children = []
        self.output = value

        # Add this node to the currently active default graph
        _default_graph.parameters.append(self)



class ComputationGraph:
    """A computational graph data structure with methods to train neural network 
    architectures.

    Attributes
    -----------
    loss_train_ : List
        Training loss after each epoch
    loss_val_ : List
        Validation loss after each epoch
    score_train_ : List
        Training metric score after each epoch
    score_val_ : List
        Validation metric score after each epoch
        
    Methods
    -----------
    fit(Xtrain, Ytrain, Xval, Yval, *args, *kwargs)
        Fit the NN with mini-batch SGD and return fitted model.
    predict(Xpred)
        Predict and return targets from supplied design matrix.
    
    Author: Douglas Rubin
    """

    def __init__(self):
        self.computations = []
        self.inputs = []
        self.parameters = []
        
        
    def as_default(self):
        global _default_graph
        _default_graph = self
        
        
    def _Forward(self, operation_node, iterative=True):
        """Compute the forward pass to populate all net inputs and all activations.  
        Returns matrix of target predictions for training batch.  Can be computed 
        iteratively or recursively with a DFS."""
        if iterative:
            return ForwardIterative(operation_node, self.topo_sorted_operations)
        else:
            return ForwardDFS(operation_node, set())
        

    def _Backward(self, derivative_node, iterative=True):
        """Compute backward pass with dynamic programming to compute gradients for all 
        model parameters. Can be implemented iteratively (bottom-up) or recursively 
        (top-down).
        """
        if iterative:
            self.grads_ = GradDPIterative(derivative_node, self.topo_sorted_operations)
        else:
            self.grads_ = GradDPRecursive(derivative_node, self.parameters)

    
    def fit(self, Xtrain, Ytrain, Xval, Yval, X, Y, L, Z_L, activation_L, iterative=True, \
        B = 100, eta = .01, n_epochs=100, seed=None, track_progress=True, \
        progress_metric = None, verbose=True):
        """ Use mini-batch SGD to learn weights from training data.

        Parameters
        -----------
        Xtrain : array, shape = [n_features, n_examples]
            Training design matrix
        Ytrain : array, shape = [n_classes, n_examples]
            Training class labels
        Xval : array, shape = [n_features, n_examples]
            Validation design matrix used to compute loss/score during training.
        Yval : array, shape = [n_classes, n_examples]
            Validation class labels
        X : InputNode reference
            Reference to node containing the training data
        Y : InputNode reference
            Reference to node to containing the training target data
        L : ComputationNode reference
            Reference to node that computes the loss
        Z_L: ComputationNode reference
            Reference to node that computes the activation in the final output (Lth) layer
        activation_L : dougnet activations function
            Activation function in the final layer (must correspond to the activation 
            associated with the supplied loss node, L)
        iterative : bool (default: True)
            Whether to run the forward and backward methods iteratively or recursively
        progress_metric : dougnet metric function (default: None, indicating to not track 
            progress on a validation set) 
            If set, records the training and validation loss and provided metric score for each 
            epoch in loss_train_, loss_val_, score_train_ and score_val_.  As of now, only 
            Accuracy, RMSE and R^2 metrics are implemented.
        verbose : bool (default: True)
            if set, print progress during training.

        Returns:
        ----------
        self

        """
        self.start_time = datetime.now()
        self.X = X
        self.Y = Y 
        self.L = L 
        self.Z_L = Z_L 
        self.activation_L = activation_L
        self.iterative = iterative
        
        if self.iterative:
            self.topo_sorted_operations = []
            TopologicalSort(L, self.topo_sorted_operations, set())  
        
        def total_loss(z, y):
            self.Y.output = y
            return self._Forward(self.L, self.iterative)
        self.total_loss = total_loss    
        
        self.B = B
        self.eta = eta
        self.n_epochs = n_epochs
        self.random = np.random.RandomState(seed)
        self.progress_metric = progress_metric
        self.track_progress = track_progress
        self.verbose = verbose
        if track_progress:
            self.epoch_str_len = len(str(self.n_epochs))
            self.loss_train_ = []
            self.loss_val_ = []
            self.score_train_ = []
            self.score_val_ = []   
        

        # initialize all model parameters
        for parameter in self.parameters:
            parameter.initialize(self.random)

        # loop over epochs
        for epoch in range(self.n_epochs):

            # perform mini batch updates to parameters
            for X_B, Y_B in GetMiniBatches(self, Xtrain, Ytrain):
                self.X.output, self.Y.output = X_B, Y_B
                
                _ = self._Forward(self.L, self.iterative)
                self._Backward(self.L, self.iterative)
                
                # update parameters 
                for parameter in self.parameters:
                    parameter.output -= (self.eta/self.B) * self.grads_[parameter]

            if self.progress_metric or self.verbose:
                ProgressHelper(self, Xtrain, Ytrain, Xval, Yval, epoch)

        return self
                
    
    def predict(self, Xpred, return_net_input=False):
        """Predict target 

        Parameters
        -----------
        Xpred : array, shape = [n_features, n_examples]
            Design matrix corresponding to desired predictions.
        return_net_input : bool (default: False)
            if set, returns both the Y_hat matrix as well as Z, the net input matrix of the 
            final layer. 

        Returns:
        ----------
        Y_hat : array, shape = [n_classes, n_examples]
            Prediction targets for each example
        Z : array, shape = [n_classes, n_examples]
            The net input for each example.  Only returned if return_net_input set.
        
        """
        
        self.X.output = Xpred
        net_input = self._Forward(self.Z_L, self.iterative)
        Y_hat = self.activation_L.func(net_input)
        
        if return_net_input:
            return Y_hat, net_input
        else:
            return Y_hat


##########define some computational graph utility functions for use above#########

def TopologicalSort(node, topo_sorted_operations, already_visited):
    """Topologically sort all ancestors from the supplied node in the DAG and
    store topological sorting in topo_sorted_operations"""
    
    if isinstance(node, ComputationNode):
        for parent in node.parents:
            if parent not in already_visited:
                already_visited.add(parent)
                TopologicalSort(parent, topo_sorted_operations, already_visited)     
    topo_sorted_operations.append(node)


def ForwardIterative(operation_node, topo_sorted_operations):
    """Run forward pass up until operation_node by iteratively looping through
    the topologically sorted operations in the DAG"""
    
    for node in topo_sorted_operations:
        if isinstance(node, ComputationNode):
            node.compute()
        if node == operation_node:
            return operation_node.output



def ForwardDFS(node, already_visited):
    """Run forward pass up until node recursively with a DFS"""
    
    if isinstance(node, ComputationNode):
        for parent in node.parents:
            if parent not in already_visited:
                already_visited.add(parent)
                _ = ForwardDFS(parent, already_visited)        
        node.compute()
        
    return node.output



def GradDPIterative(derivative_node, topo_sorted_operations): 
    """Iteratively run backward pass with dynamic programming (bottom-up) to compute the 
    gradients for the ParameterNodes and ComputationNodes."""
    
    grads_memo = {}
    for node in reversed(topo_sorted_operations):
        if node == derivative_node:
            grads_memo[node] = 1.
        
        # do not require gradients of input nodes
        elif isinstance(node, ParameterNode) or isinstance(node, ComputationNode):
            grads_memo[node] = sum(child.VJP(node, grads_memo[child]) 
                for child in node.children)
    
    return grads_memo



def GradDPRecursive(derivative_node, parameters):
    """Recursiely run backward pass with dynamic programming (top-down) to compute the 
    gradients for the ParameterNodes and ComputationNodes.""" 

    grads_memo = {}
    def GradDPRecursive_helper(node, derivative_node):
        if node in grads_memo:
            return grads_memo[node]
        
        if node == derivative_node:
            grads_memo[node] = 1
            return 1

        grads_memo[node] = sum(child.VJP(node, GradDPRecursive_helper(child, 
            derivative_node)) for child in node.children)

        return grads_memo[node]

    
    for parameter_node in parameters:
        _ = GradDPRecursive_helper(parameter_node, derivative_node)
    
    return grads_memo
