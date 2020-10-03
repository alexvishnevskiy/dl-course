import numpy as np


class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """Applying gradient descent to parameters"""
        for parameter in self.parameters:
            gradient = parameter.grad if parameter.data.shape == parameter.grad.shape else \
                                                           np.sum(parameter.grad, axis = 0)
            parameter.data -= self.lr*gradient

    def zero_grad(self):
        """Resetting gradient for all parameters (set gradient to zero)"""
        for i, parameter in enumerate(self.parameters):
            self.parameters[i].grad = 0 
            

class StepLR:
    def __init__(self, optimizer, step_size, gamma):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.count_steps = 0
        
    def step(self):
        self.count_steps += 1
        if self.count_steps == self.step_size:
            self.optimizer.lr *= self.gamma
            self.count_steps = 0
