import numpy as np

class AddGate:
    def forward(self,x1,x2):
        return x1+x2
    def backward(self,x1,x2):
        pass

class MultiplyGate:
    def forward(self,W,x):
        return np.dot(W,x)
    def backward(self, W, x, dz):
        dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)), np.asmatrix(x) ))
        dx = np.dot(np.transpose(W), dz)
        return dW, dx        
