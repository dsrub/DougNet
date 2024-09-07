import numpy as np
import dougnet as dn
from dougnet._computation_graph._node_base import InputNode
from dougnet._computation_graph._parameter_nodes import WeightNode, BiasNode

     
class Module:
    """
    Module base class.  This is a thin wrapper to provide functionality similar to
    pytorch's module class.  The method, forward_, must be specified which is 
    typically code to perform a computation in a computational graph.  The forward_
    method is run during instantiation of the Module class.  The output of the forward_
    method can be anything (but is typically a reference to the last node in the 
    computation performed) and is saved in the module_output attribute.
    """
    def __init__(self):
        self.module_output = self.forward_()

    def forward_(self):
        raise NotImplementedError()
        
class Linear(Module):
    """
    A linear layer computing Z = W X + b.

    Parameters
    ------------
    X : Node
        The parent node from which to make the computation with X.output.shape = 
        [in_features x batch_size].
    out_features : int
        number of output features
    in_features : int
        number of input features
    dtype : numpy data type (default=np.float32)
        data type of the created weights
    weight_init: str (default="normal")
        weight initializer 
    bias_init : str (default="zeros")
        bias initializer
    weight_init_kwargs : dict (default={})
        keyword arguments for weight initializer
    bias_init_kwargs : dict (default={})
        keyword arguments for bias initializer
    weight : WeightNode or None (default=None)
        If provided, use this weight.  If provided, weight_init and 
        weight_init_kwargs have no effect.
    bias : BiasNode or None (default=None)
        If provided, use this bias.  If provided, bias_init and 
        bias_init_kwargs have no effect.
    
    Attributes
    ----------
    module_output : ComputationNode
        The node, Z.
    """
    def __init__(self, 
                 X, 
                 out_features, 
                 in_features, 
                 dtype=np.float32, 
                 weight_init="normal", 
                 bias_init="zeros", 
                 weight_init_kwargs={},
                 bias_init_kwargs={}, 
                 weight=None, 
                 bias=None):
        self.X = X
        self.out_features = out_features
        self.in_features = in_features
        self.dtype = dtype
        self.weight = weight
        self.bias = bias
        if self.weight is None:
            self.weight = WeightNode(self.out_features, 
                                     self.in_features, 
                                     dtype=self.dtype, 
                                     initializer=weight_init, 
                                     **weight_init_kwargs)
        if self.bias is None:
            self.bias = BiasNode(self.out_features, 
                                 dtype=self.dtype,
                                 initializer=bias_init, 
                                 **bias_init_kwargs)
        super().__init__()
        
    def forward_(self):
        return self.weight @ self.X + self.bias 
    
class RNNCell(Module):
    """
    An RNN cell which updates the hidden state, H, according to:
    
    H = tanh(W_xh X + W_hh H + b).
    
    Parameters
    ------------
    X : ComputationNode, InputNode or ParameterNode 
        The current input node with output.shape = [input_dim x batch_size].
    weight_xh : ParameterNode
        The weight node for the input-to-hidden multiplication with output.shape =
        [hidden_dim x input_dim].
    weight_hh : ParameterNode
        The weight node for the hidden-to-hidden multiplication with output.shape =
        [hidden_dim x hidden_dim].
    bias : ParameterNode
        The bias node with output.shape = [hidden_dim x 1].
    H : ComputationNode, InputNode or ParameterNode 
        The previous hidden state node with output.shape = [hidden_dim x batch size].
        
    Attributes
    ----------
    module_output : ComputationNode
        The node, H.
        
    Note
    ----
    Unlike Pytorch, the parameters for the RNNCell (weight_xh, weight_hh, bias) and the 
    previous hidden state (H) are not created upon instantiation of the RNNCell and must be 
    explicitly provided to the class upon instantiation.  This has to do with fundamental 
    differences between how a computational graph is created in dougnet versus how it is 
    created in pytorch (static vs. dynamic). The following convenience class methods to 
    instantiate these nodes before instantiating the RNNCell class are therefore provided: 
    instantiate_weight_xh, instantiate_weight_hh, instantiate_bias and instantiate_state.  
    For more detail and examples of how to properly define and train RNN models in dougnet 
    please see notebook 5 in the examples section.
    """
    @classmethod
    def instantiate_weight_xh(cls, hidden_dim, input_dim):
        """a convenience class method to instantiate the weight_xh node for an RNN"""
        return WeightNode(hidden_dim, input_dim, initializer="xavier")
        
    @classmethod
    def instantiate_weight_hh(cls, hidden_dim):
        """a convenience class method to instantiate the weight_hh node for an RNN"""
        return WeightNode(hidden_dim, hidden_dim, initializer="xavier")
    
    @classmethod
    def instantiate_bias(cls, hidden_dim):
        """a convenience class method to instantiate the bias node for an RNN"""
        return BiasNode(hidden_dim, initializer="zeros")
    
    @classmethod
    def instantiate_state(cls, hidden_dim, batch_size):
        """a convenience class method to instantiate the initial hidden state for an RNN"""
        return InputNode(np.zeros((hidden_dim, batch_size), dtype=np.float32))
    
    def __init__(self, X, weight_xh, weight_hh, bias, H):
        self.X = X
        self.H = H
        self.weight_xh = weight_xh
        self.weight_hh = weight_hh
        self.bias = bias
        super().__init__()
        
    def forward_(self):
        Z = self.weight_xh @ self.X + self.weight_hh @ self.H + self.bias
        return dn.Tanh(Z)
    
class LSTMCell(Module):
    """
    An LSTM cell which updates the hidden and cell states, H, C according to:
    
    Z = W_xh X + W_hh H + bias
        
    I = sigma(Z[:hidden_dim, :])
    F = sigma(Z[hidden_dim: 2 * hidden_dim, :])
    G = tanh(Z[2 * hidden_dim: 3 * hidden_dim, :])
    O = sigma(Z[3 * hidden_dim: 4 * hidden_dim, :])
    
    C = F * C + I * G
    H = O * tanh(C) 

    Parameters
    ------------
    X : ComputationNode, InputNode or ParameterNode 
        The current input node with output.shape = [input_dim x batch_size].
    weight_xh : WeightNode
        The weight node for the input-to-hidden multiplication output.shape = 
        [4 * hidden_dim x input_dim].
    weight_hh : WeightNode
        The weight node for the hidden-to-hidden multiplication output.shape = 
        [4 * hidden_dim x hidden_dim].
    bias : BiasNode
        The bias node with output.shape = [4 * hidden_dim x 1].
    H : ComputationNode, InputNode or ParameterNode 
        The previous hidden state node with output.shape = [hidden_dim x batch_size].
    C : ComputationNode, InputNode or ParameterNode 
        The previous cell state node with output.shape = [hidden_dim x batch_size].
        
    Attributes
    ----------
    module_output : tuple(ComputationNode, ComputationNode)
        The nodes, (H, C).
        
    Note
    ----
    See note in RNNCell module class.
    """
    @classmethod
    def instantiate_weight_xh(cls, hidden_dim, input_dim):
        """a convenience class method to instantiate the weight_xh node for an LSTM"""
        return WeightNode(4 * hidden_dim, input_dim, initializer="xavier", fan_out=hidden_dim) 
    
    @classmethod
    def instantiate_weight_hh(cls, hidden_dim):
        """a convenience class method to instantiate the weight_hh node for an LSTM"""
        return WeightNode(4 * hidden_dim, hidden_dim, initializer="xavier", fan_out=hidden_dim)
    
    @classmethod
    def instantiate_bias(cls, hidden_dim):
        """a convenience class method to instantiate the bias node for an LSTM"""
        return BiasNode(4 * hidden_dim, initializer="zeros")
    
    @classmethod
    def instantiate_state(cls, hidden_dim, batch_size):
        """a convenience class method to instantiate the initial hidden states, H, C for an LSTM"""
        H = InputNode(np.zeros((hidden_dim, batch_size), dtype=np.float32))
        C = InputNode(np.zeros((hidden_dim, batch_size), dtype=np.float32))
        return H, C
    
    def __init__(self, X, weight_xh, weight_hh, bias, H, C):
        self.X = X
        self.H = H
        self.C = C
        self.weight_xh = weight_xh
        self.weight_hh = weight_hh
        self.bias = bias
        self.hidden_dim = weight_xh.output.shape[0] // 4
        super().__init__()
        
    def forward_(self):
        
        # pre-activation
        Z = self.weight_xh @ self.X + self.weight_hh @ self.H + self.bias
        
        # activations
        I = dn.Sigmoid(Z[:self.hidden_dim, :])
        F = dn.Sigmoid(Z[self.hidden_dim: 2 * self.hidden_dim, :])
        G = dn.Tanh(Z[2 * self.hidden_dim: 3 * self.hidden_dim, :])
        O = dn.Sigmoid(Z[3 * self.hidden_dim: 4 * self.hidden_dim, :])
        
        # update hidden states
        C = F * self.C + I * G
        H = O * dn.Tanh(C) 
        
        return H, C
    
class RNN(Module):
    """
    An RNN applied to an input sequence.

    Parameters
    ------------
    X : ComputationNode, InputNode or ParameterNode 
        The input node with output.shape = [batch_size x sequence_length x input_dim].
    weight_xh : List[WeightNode]
        The weight nodes for the input-to-hidden multiplication for each layer in the RNN.  
        The weight node for the first layer will have output.shape = [hidden_dim x input_dim],
        while all subsequent layers will have output.shape = [hidden_dim x hidden_dim].
    weight_hh : List[WeightNode]
        The weight nodes for the hidden-to-hidden multiplication for each layer in the RNN.  
        The weight nodes have output.shape = [hidden_dim x hidden_dim].
    bias : List[BiasNode]
        The bias nodes for each layer in the RNN.  The bias nodes have output.shape = 
        [hidden_dim x 1].
    sequence_length : int
        The length of the provided sequence (X.output.shape[1]).  This is necessary since 
        X.output will usually not actually contain data until loss.forward() is called (after
        the RNN has been instantiated).
    H : ComputationNode, InputNode or ParameterNode 
        The previous hidden state nodes for each layer in the RNN with output.shape = 
        [hidden_dim x batch_size].
    num_layers : int (default=1)
        The number of layers in the RNN.
        
    Attributes
    ----------
    module_output : tuple(List[ComputationNode], List[ComputationNode])
        The tuple, (output, hidden), which are lists providing the output nodes and hidden 
        nodes respectively.  The output nodes are defined as the hidden nodes in the last 
        layer of the RNN.  Thus, len(output) = sequence_length and all nodes in this list 
        have output.shape = [hidden_dim x batch_size].  The hidden nodes are defined as the 
        hidden nodes (Hs) across all layers for the last token of the input sequence.
        Thus, len(hidden) = num_layers and all nodes in this list have output.shape = 
        [hidden_dim x batch_size].
        
    Note
    ----
    See note in RNNCell module class.
    """
    @classmethod
    def instantiate_weight_xh(cls, hidden_dim, input_dim, num_layers=1):
        """a convenience class method to instantiate the weight_xh nodes for each layer in an RNN"""
        f = RNNCell.instantiate_weight_xh
        return [f(hidden_dim, input_dim)] + [f(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
    
    @classmethod
    def instantiate_weight_hh(cls, hidden_dim, num_layers=1):
        """a convenience class method to instantiate the weight_hh nodes for each layer in an RNN"""
        return [RNNCell.instantiate_weight_hh(hidden_dim) for _ in range(num_layers)]
    
    @classmethod
    def instantiate_bias(cls, hidden_dim, num_layers=1):
        """a convenience class method to instantiate the bias nodes for each layer in an RNN"""
        return [RNNCell.instantiate_bias(hidden_dim) for _ in range(num_layers)]
    
    @classmethod
    def instantiate_state(cls, hidden_dim, batch_size, num_layers=1):
        """a convenience class method to instantiate the initial hidden state nodes for each layer in an RNN"""
        return [RNNCell.instantiate_state(hidden_dim, batch_size) for _ in range(num_layers)]
    
    def __init__(self, X, weight_xh, weight_hh, bias, sequence_length, H, num_layers=1):
        self.X = X
        self.H = H
        self.weight_xh = weight_xh
        self.weight_hh = weight_hh
        self.bias = bias
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        super().__init__()
        
    def forward_(self):
        hidden = []
        output = [dn.Transpose(self.X[:, t, :], (1, 0)) for t in range(self.sequence_length)]
        for layer in range(self.num_layers):
            H = self.H[layer]
            weight_xh = self.weight_xh[layer]
            weight_hh = self.weight_hh[layer] 
            bias = self.bias[layer]
            for t in range(self.sequence_length):
                H = RNNCell(output[t], weight_xh, weight_hh, bias, H).module_output # [hidden x batch]
                output[t] = H
            hidden.append(H)
                 
        return output, hidden
    
class LSTM(Module):
    """
    An LSTM applied to an input sequence.

    Parameters
    ------------
    X : ComputationNode, InputNode or ParameterNode 
        The input node with output.shape = [batch_size x sequence_length x input_dim].
    weight_xh : List[WeightNode]
        The weight nodes for the input-to-hidden multiplication for each layer in the LSTM.  
        The weight node for the first layer will have output.shape = [4 * hidden_dim x input_dim],
        while all subsequent layers will have output.shape = [4 * hidden_dim x hidden_dim].
    weight_hh : List[WeightNode]
        The weight nodes for the hidden-to-hidden multiplication for each layer in the LSTM.  
        The weight nodes have output.shape = [4 * hidden_dim x hidden_dim].
    bias : List[BiasNode]
        The bias nodes for each layer in the LSTM.  The bias nodes have output.shape = 
        [4 * hidden_dim x 1].
    sequence_length : int
        The length of the provided sequence (X.output.shape[1]).  This is necessary since 
        X.output will usually not actually contain data until loss.forward() is called (after
        the LSTM has been instantiated).
    H : ComputationNode, InputNode or ParameterNode 
        The previous hidden state nodes for each layer in the LSTM with output.shape = 
        [hidden_dim x batch_size].
    C : ComputationNode, InputNode or ParameterNode 
        The previous cell state nodes for each layer in the LSTM with output.shape = 
        [hidden_dim x batch_size].
    num_layers : int (default=1)
        The number of layers in the LSTM.
        
    Attributes
    ----------
    module_output : List[ComputationNode], tuple(List[ComputationNode], List[ComputationNode])
        output, (hidden, cell).  output and hidden are described in the RNN.  cell provides 
        exactly the same information as hidden, except for the cell state values and not the 
        hidden state values. 
        
    Note
    ----
    See note in RNNCell module class.
    """
    @classmethod
    def instantiate_weight_xh(cls, hidden_dim, input_dim, num_layers=1):
        """a convenience class method to instantiate the weight_xh nodes for each layer in an LSTM"""
        f = LSTMCell.instantiate_weight_xh
        return [f(hidden_dim, input_dim)] + [f(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
    
    @classmethod
    def instantiate_weight_hh(cls, hidden_dim, num_layers=1):
        """a convenience class method to instantiate the weight_hh nodes for each layer in an LSTM"""
        return [LSTMCell.instantiate_weight_hh(hidden_dim) for _ in range(num_layers)]
    
    @classmethod
    def instantiate_bias(cls, hidden_dim, num_layers=1):
        """a convenience class method to instantiate the bias nodes for each layer in an LSTM"""
        return [LSTMCell.instantiate_bias(hidden_dim) for _ in range(num_layers)]
    
    @classmethod
    def instantiate_state(cls, hidden_dim, batch_size, num_layers=1):
        """a convenience class method to instantiate the initial hidden state nodes for each layer in an RNN"""
        H_C = [LSTMCell.instantiate_state(hidden_dim, batch_size) for _ in range(num_layers)]
        H, C = [x[0] for x in H_C], [x[1] for x in H_C]
        return H, C
        
    def __init__(self, X, weight_xh, weight_hh, bias, sequence_length, H, C, num_layers=1):
        self.X = X
        self.H = H
        self.C = C
        self.weight_xh = weight_xh
        self.weight_hh = weight_hh
        self.bias = bias
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        super().__init__()
        
    def forward_(self):
        hidden, cell = [], []
        output = [dn.Transpose(self.X[:, t, :], (1, 0)) for t in range(self.sequence_length)]
        for layer in range(self.num_layers):
            H = self.H[layer]
            C = self.C[layer]
            weight_xh = self.weight_xh[layer]
            weight_hh = self.weight_hh[layer] 
            bias = self.bias[layer]
            for t in range(self.sequence_length):                
                H, C = LSTMCell(output[t], weight_xh, weight_hh, bias, H, C).module_output 
                # H: [hidden x batch], C: [hidden x batch]
                output[t] = H
            hidden.append(H)
            cell.append(C)
                 
        return output, (hidden, cell)