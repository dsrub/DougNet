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

The jupyter notebooks in the [examples](https://github.com/dsrub/DougNet/tree/master/examples) directory contain plenty of examples on how to use DougNet and highlight what the underlying code is actually doing.  Almost all of the DougNet examples are compared to PyTorch implementations and the results compare remarkably well.  PyTorch is therefore required to run the notebooks and can be installed via:
```bash
pip install torch
```

## Running tests

To run tests locally, clone DougNet and install the dependencies:
```bash
git clone https://github.com/dsrub/DougNet.git
cd DougNet
pip install -r ./requirements/requirements_tests.txt
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