from datetime import datetime
import numpy as np
import sys
from layers import RNNLayer
from activation-functions import Softmax

class Model:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.wax = np.random.uniform(-np.sqrt(1. /word_dim),\
             np.sqrt(1. /word_dim), (hidden_dim, word_dim))
        self.waa = np.random.uniform(-np.sqrt(1. /hidden_dim),\
             np.sqrt(1. /hidden_dim), (hidden_dim, hidden_dim))
        self.waa = np.random.uniform(-np.sqrt(1. /hidden_dim),\
             np.sqrt(1. /hidden_dim), (word_dim, hidden_dim))
