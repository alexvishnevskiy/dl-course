import numpy as np
from engine import Value, Tensor


class Module:
    """
    Base class for every layer.
    """
    def forward(self, *args, **kwargs):
        """Depends on functionality"""
        pass

    def __call__(self, *args, **kwargs):
        """For convenience we can use model(inp) to call forward pass"""
        return self.forward(*args, **kwargs)

    def parameters(self):
        """Return list of trainable parameters"""
        return []


class Linear(Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        """Initializing model"""
        # Create Linear Module

    def forward(self, inp):
        """Y = W * x + b"""
        return ...

    def parameters(self):
        return ...


class ReLU(Module):
    """The most simple and popular activation function"""
    def forward(self, inp):
        # Create ReLU Module
        return ...


class CrossEntropyLoss(Module):
    """Cross-entropy loss for multi-class classification"""
    def forward(self, inp: "Tensor", label: "Tensor") -> "Value":
        # Create CrossEntropy Loss Module
        #inp.size = (batch_size, number of classes)
        #label.size = (batch_size)
        #loss = -log(exp(x_i)/np.sum(exp)) cross entropy loss
        #construct jacobian
        grads = []
        loss = []
        for i, row in enumerate(inp.data):
            init = list(map(Value, row))
            exp_ = list(map(lambda x: x.exp(), init))
            softmax = list(map(lambda x: x/sum(exp_), exp_))
            outputs = -softmax[label.data[i]].log()
            outputs.backward()
            grads.append(list(map(lambda x: x.grad, init)))
            loss.append(outputs)
        jacobian = np.array(grads)
        loss = sum(loss)/len(loss)
        loss.backward = lambda : inp.backward(jacobian)
        return loss