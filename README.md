# DougNet 

<p align="center">
    <img src="single_layer_MLP.jpg" width="70%">
</p>

DougNet is a deep learning api written entirely in python and is intended as a pedogogical tool for understanding the inner-workings of a deep learning library.  The api is written from scratch and nowhere uses commercial deep learning libraries like [PyTorch](https://pytorch.org) or [TensorFlow](https://www.tensorflow.org) (although it does utilize PyTorch to unit test for correctness).  For ease of use, the syntax and api of DougNet is very similar to that of PyTorch.  Unlike PyTorch, DougNet was written so that its source code is *readable*.  The source code is lightweight: the amount of code and type-checking is kept to a minimum and the file structure is compact.  For readability, the source code is also written entirely in python, using [Numpy's](https://numpy.org) `ndarray` data structure as its main tensor api.  A few computationally intensive functions require [Numba](https://numba.pydata.org) which compiles python functions to optimized machine code and allows for multi-threading.  In keeping with DougNet's philosophy of readability, Numba is a good choice for speeding up slow functions since it requires only a python decorator function and usually almost no changes to the python code.

Even though DougNet was not written for performance, it compares surprisingly well to PyTorch.  In most cases it seems that DougNet is only a factor of $\sim 1$ to $2$ times slower than the equivalent PyTorch cpu implementation.

Some of the math and algorithms behind DougNet can be complicated.  For example, understanding the automatic differentiation engine that powers DougNet requires knowledge of graph data structures, dynamic programming, matrix calculus and tensor contractions.  I am currently working on a companion text, possibly a book or a blog, reviewing the math behind deep learning libraries.  

## Main features of DougNet

DougNet's main features are:
- a computational graph data structure for reverse mode automatic differentiation
- functions specific to neural network models, such as multi-channel convolution
- optimization algorithms for training neural networks
- initialization and regularization methods for neural networks
- utilities for fetching datasets and loading data during training
- functionality for multilayer perceptrons, convolutional neural networks and recurrent neural networks

## Installation

To install DougNet, simply type:
```bash
pip install dougnet
```

## Example usage

The jupyter notebooks in the [examples](https://github.com/dsrub/DougNet/tree/master/examples) directory contain plenty of examples on how to use DougNet and highlight how the underlying code *actually* works.  Almost all of the DougNet examples are compared to PyTorch implementations and the results agree remarkably well.  PyTorch is therefore required to run the notebooks, as well as matplotlib. Both can be installed via:
```bash
pip install torch
pip install matplotlib
```

### Linear regression

The following toy example uses DougNet to train a linear regression model.
```python
import dougnet as dn
import numpy as np

N_EPOCHS = 1_000
N_FEAT = 3
N_EXAMPLES = 100
LR = 0.1
```
```python
# create data according to Y = beta X + b + eps
rng = np.random.RandomState(1)

X_train = rng.uniform(0, 1, (N_FEAT, N_EXAMPLES)) # examples in cols of design matrix
eps = rng.normal(0, .01, (1, N_EXAMPLES))
beta = np.array([.1, .2, .3])
Y_train = beta @ X_train + .4 + eps
```
```python
# instantiate computation graph
graph = dn.ComputationGraph()

# data nodes
X = dn.InputNode(X_train)
Y = dn.InputNode(Y_train)

# output layer
W = dn.ParameterNode(np.zeros((1, 3)))
b = dn.ParameterNode(0.)
Yhat = W @ X + b

# loss node
L = dn.L2Loss(Yhat, Y)

# train
for _ in range(N_EPOCHS):
    # run forward and backward methods
    _ = L.forward()
    L.backward()

    # update parameters 
    for parameter in graph.parameters:
        parameter.output -= LR * graph.grads_[parameter]

print(W.output) # prints [[0.09982835 0.20112498 0.30432221]]
print(b.output) # prints [[0.39778574]]
```

### MLP

As a slightly more complex example, the following code trains a shallow MLP on the MNIST data.  In contrast to the example above, this code utilizes some of the convenience functionality provided by DougNet, such as: a module class to package model code, weight initialization methods, mini-batch data loaders and optimization algos to update parameters.  The model takes only a few seconds to train and achieves $\sim 96$% accuracy on the validation set.  As one can see, DougNet code feels very much like PyTorch.
```python
import dougnet as dn
import numpy as np

SEED_WEIGHT = 1
SEED_DATA = 2
N_EPOCHS = 10
BATCH_SIZE = 100
LR = 0.001
```
```python
# load data
def PrepData(X, y):
    """one hot encode Ys and standardize X"""
    Y_ohe = np.zeros((y.size, 10))
    Y_ohe[np.arange(y.size), y] = 1
    X = ((X / 255.) - .5) * 2
    return X.astype(np.float32).T, Y_ohe.astype(np.float32).T

X_train, Y_train, X_val, Y_val = dn.data.LoadMNIST()
X_train, Y_train = PrepData(X_train, Y_train)
X_val, Y_val = PrepData(X_val, Y_val)
```
```python
# define model
class MLP(dn.Module):
    def __init__(self, n_neurons=100):
        self.n_neurons = n_neurons
        self.X = dn.InputNode()
        self.Y = dn.InputNode()
        super().__init__()
        
    def forward_(self):
        z = dn.Linear(self.X, self.n_neurons, 28 * 28, weight_init="xavier")
        a = dn.Relu(z.module_output)
        return dn.Linear(a, 10, self.n_neurons).module_output

# instantiate graph/model and initialize weights
graph = dn.ComputationGraph()
model = MLP()
graph.initialize_params(SEED_WEIGHT)

# train
logits_hat = model.module_output
L = dn.SoftmaxCrossEntropyLoss(logits_hat, model.Y)
dataloader = dn.training.DataLoader(X_train, Y_train, BATCH_SIZE, random_state=SEED_DATA)
optim = dn.optim.Adam(graph, eta=LR)
for _ in range(N_EPOCHS):
    for X_B, Y_B in dataloader.load():
        model.X.output, model.Y.output = X_B, Y_B
        _ = L.forward()
        L.backward()
        optim.step()

# compute validation accuracy (prints validation accuracy = 0.96)
model.X.output = X_val
print(f"validation accuracy = {round(dn.metrics.accuracy(logits_hat.forward(), Y_val), 2)}")
```

## Running tests

To run tests locally, install DougNet in editable mode:
```bash
git clone https://github.com/dsrub/DougNet.git
cd DougNet
pip install -r ./requirements/requirements_tests.txt
pip install --editable .
```
This will install DougNet as well as PyTorch, which most unit tests use to verify the correctness of DougNet, and pytest for running the tests.

To run all tests, navigate to the root directory and run:
```bash
pytest
```
To run specific tests, provide the path to the desired testing file.  For example, to run the unit tests for DougNet's multi-channel convolution functions, run:
```bash
cd DougNet
pytest ./dougnet/functions/tests/test_convolution.py
```

## Notes on DougNet

There are a few things to be aware of when using DougNet.  

- DougNet can only be used on a cpu.  There is a lot of interesting engineering to optimize deep learning libraries on gpus, but this is beyond the scope of DougNet.

- For two dimensional data (for example, the design matrix input to a multilayer perceptron) DougNet uses the convention that examples are in columns and features are in rows.  This is in contrast to the typical machine learning convention which has examples in rows and features in columns.  This is mainly to make the math slightly more readable. 

- DougNet utilizes the *denominator* layout convention for Jacobians.  This means that if the tensor, $\mathsf{Z} \in \mathbf{R}^{m_1 \times m_2 \times \ldots \times m_M}$, depends on the tensor $\mathsf{X} \in \mathbf{R}^{n_1 \times n_2 \times \ldots \times n_N}$, the Jacobian of partial derivatives, $\frac{\partial \mathsf{Z} }{\partial \mathsf{X}}$, is arranged such that $\frac{\partial \mathsf{Z} }{\partial \mathsf{X}} \in \mathbf{R}^{(n_1 \times n_2 \times \ldots \times n_N) \times (m_1 \times m_2 \times \ldots \times m_M)}$.

- There are a few noteable differences between how DougNet implements a computational graph and how other commercial libraries implement computational graphs.  The differences were an intentional choice for the sake of making DougNet as pedagogical as possible.  First of all, DougNet implements a *static* graph, although some functionality is added for removing computational nodes for training RNNs.  Also, for backprop, DougNet computes gradients using a pure dynamic programming algorithm on the graph describing the forward pass, whereas many commercial libraries compute gradients by augmenting this graph with computational nodes that can compute the desired gradients.