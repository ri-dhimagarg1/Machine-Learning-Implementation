import numpy as np

class AddGate:
    def forward(self,x1,x2):
        return x1+x2
    def backward(self,x1,x2):
        pass

class MultiplyGate:
    def forward(self,W,x):
        return np.dot(W,x)
    def backward():
        pass
