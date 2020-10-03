import numpy as np
from typing import Union, Tuple
from math import exp
import math


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
    
    def log(self):
        out = Value(math.log(self.data), (self,), _op = 'log')
        
        def _backward():
            self.grad += out.grad/(self.data)

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


class Tensor(Value):
    """
    Tensor is a kinda array with expanded functianality.

    Tensor is very convenient when it comes to matrix multiplication,
    for example in Linear layers.
    """
    def __init__(self, data, _children=(), _op=""):
        super().__init__(data, _children, _op)
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        if isinstance(other, Tensor):
            #if two of them are Tensors just add
            out =  Tensor(self.data + other.data, _children = (self, other), _op = '+')
        else:
            #convert to Tensor
            #add Tensor and Tensor
            other = Tensor(np.ones_like(self.data)*other.data) if isinstance(other, Value) else \
                                                               Tensor(np.ones_like(self.data)*other)
            out = Tensor(self.data + other.data, _children = (self, other), _op = '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        
        return out
    
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), 'matmul')
        
        def _backward():
            self.grad += out.grad @ other.data.transpose()
            other.grad += self.data.transpose() @ out.grad
        
        out._backward = _backward
        
        return out
            
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.ones_like(self.data)*other)
        out =  Tensor(self.data * other.data, (self, other), 'multiply')
        
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        
        out._backward = _backward
        
        return out
    
    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), _op="power {}".format(other))

        def _backward():
            self.grad += (other*self.data**(other-1))*out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), _op="exp")

        def _backward():
            self.grad += np.exp(self.data)*out.grad

        out._backward = _backward
        return out

    def dot(self, other):
        assert isinstance(other, Tensor)
        assert other.shape() == self.shape()
        
        out = Tensor(np.dot(self.data, other), (self, other), _op = 'dot')
        
        def _backward():
            self.grad += other.data*(np.ones_like(other.data)*out.grad)
            other.grad += self.data*(np.ones_like(self.data)*out.grad)
            
        out._backward = _backward
        return out
    
    def backward(self, jacobian = []):

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
        self.grad = jacobian if len(jacobian) else np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def shape(self):
        return self.data.shape

    def argmax(self, dim=None):
        return np.argmax(self.data, axis = dim)

    def max(self, dim=None):
        return np.max(self.data, axis = dim)
    
    def relu(self):
        out = Tensor(np.where(self.data>0, self.data, 0), (self,), _op = 'relu')

        def _backward():
            self.grad += (self.data>0)*out.grad

        out._backward = _backward

        return out
    
    def T(self):
        out = Tensor(self.data.transpose(), (self,), _op = 'transpose')
        return out

    def reshape(self, shape: Tuple[int], order = 'C'):
        out = Tensor(self.data.reshape(shape, order = order))
        return out
    
    def squeeze(self, axis = None):
        self.data = self.data.squeeze(axis = axis)
        return self

    def parameters(self):
        return self

    def __repr__(self):
        return "Tensor\n" + str(self.data)

    def __getitem__(self, item):
        return self.data[item]
    
    def __le__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data <= other.data)
        return Tensor(self.data <= other)
    
    def __lt__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data < other.data)
        return Tensor(self.data < other)

    def __gt__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data > other.data)
        return Tensor(self.data > other)

    def item(self):
        return float(self.data.squeeze())