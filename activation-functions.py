import numpy as np

class Sigmoid:
    def forward(self,x):
        return 1.0/(1 + np.exp(-x))
    def backward():
        pass

class Tanh:
    def forward(self,x):
        return np.tanh(x)
    def bcakward():
        pass