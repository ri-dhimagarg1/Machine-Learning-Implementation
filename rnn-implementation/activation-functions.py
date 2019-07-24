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

class Softmax:
    def predict(self,x):
        exp_scores = np.exp(x)
        return exp_scores/np.sum(exp_scores)
    
    def diff():
        pass