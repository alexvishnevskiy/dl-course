import numpy as np
from typing import Union
from math import exp


class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other: Union[int, float, "Value"]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: Union[int, float, "Value"]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad

        out._backward = _backward

        return out

    def __pow__(self, other: Union[int, float]) -> "Value":
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), _op="power {}".format(other))

        def _backward():
            self.grad += (other*self.data**(other-1))*out.grad

        out._backward = _backward

        return out

    def exp(self):
        out = Value(exp(self.data), (self,), _op="exp")

        def _backward():
            self.grad += exp(self.data)*out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), _op = 'relu')

        def _backward():
            self.grad += (self.data>0)*out.grad

        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __le__(self, other):
        if isinstance(other, Value):
            return self.data <= other.data
        return self.data <= other

    def __lt__(self, other):
        if isinstance(other, Value):
            return self.data < other.data
        return self.data < other

    def __gt__(self, other):
        if isinstance(other, Value):
            return self.data > other.data
        return self.data > other

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


class Tensor:
    """
    Tensor is a kinda array with expanded functianality.

    Tensor is very convenient when it comes to matrix multiplication,
    for example in Linear layers.
    """
    def __init__(self, data):
        self.data = np.array(data)

    def __add__(self, other):
        if isinstance(other, Tensor):
            assert self.shape() == other.shape()
            return Tensor(np.add(self.data, other.data))
        return Tensor(self.data + other)
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            raise ValueError('matmul: Input operand does not have enough dimensions')
        return Tensor(self.data @ other.data)
            
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        return Tensor(self.data * other)
    
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        return Tensor(self.data / other)
    
    def __floordiv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data // other.data)
        return Tensor(self.data//other)

    def __radd__(self, other):
        return self + other
    
    def __rmull__(self, other):
        return self * other

    def exp(self):
        return Tensor(np.exp(self.data))

    def dot(self, other):
        if isinstance(other, Tensor):
            return Tensor(np.dot(self.data, other))
        return Tensor(self.data*other)

    def shape(self):
        return self.data.shape

    def argmax(self, dim=None):
        return np.argmax(self.data, axis = dim)

    def max(self, dim=None):
        return np.max(self.data, axis = dim)

    def reshape(self, *args, **kwargs):
        self.data = ...
        return self

    def backward(self):
        for value in self.data.flatten():
            value.backward()

    def parameters(self):
        return list(self.data.flatten())

    def __repr__(self):
        return "Tensor\n" + str(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def item(self):
        return self.data.flatten()[0].data
