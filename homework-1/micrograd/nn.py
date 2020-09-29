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
        stdv = 1./np.sqrt(in_features)
        self.W = np.random.uniform(-stdv, stdv, size = (out_features, in_features))
        self.b = np.random.uniform(-stdv, stdv, size = out_features)

    def forward(self, inp):
        """Y = W * x + b"""
        return inp @ self.W.T + self.b

    def parameters(self):
        return [self.W, self.b]


class ReLU(Module):
    """The most simple and popular activation function"""
    def forward(self, inp):
        # Create ReLU Module
        return inp.relu()


class CrossEntropyLoss(Module):
    """
    Cross-entropy loss for multi-class classification
    loss = -log(exp(x_i)/np.sum(exp))
       
    input.size = (batch_size, number of classes)
    label.size = (batch_size,)
    """
    def forward(self, inp: "Tensor", label: "Tensor") -> "Value":
        #construct jacobian
        grads = []
        loss = []
        for i, row in enumerate(inp.data):
            #convert all rows to Value
            init = list(map(Value, row))
            exp_ = list(map(lambda x: x.exp(), init))
            softmax = list(map(lambda x: x/sum(exp_), exp_))
            #cross entropy loss
            outputs = -softmax[label.data[i]].log()
            outputs.backward()
            #append gradients
            grads.append(list(map(lambda x: x.grad, init)))
            loss.append(outputs)
        jacobian = np.array(grads)
        loss = sum(loss)/len(loss)
        #backward pass with jacobian
        loss.backward = lambda : inp.backward(jacobian)
        return loss