import numpy as np

class Sigmoid:
    def forward(self,x):
        return 1.0/(1 + np.exp(-x))
    def backward():
        pass

class Tanh:
    def forward(self,x):
        return np.tanh(x)
    def backward():
        pass

class Softmax:
    def predict(self,x):
        exp_scores = np.exp(x)
        return exp_scores/np.sum(exp_scores)
        
    ## Simple cross entropy loss without any derivative (i.e, for forward pass)
    def loss(self,x,y):
        probs = self.predict(x)
        return -np.log(probs[y])

    ## diff -> yt-y^t
    def diff(self, x, y):
        probs = self.predict(x)
        probs[y] = probs[y] - 1.0 ## As we are subtracting from 1 because at that index in probability vector will subtract from original vector and it contains 1 at that index only.
        return probs