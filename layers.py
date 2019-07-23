from activation import Tanh
from gate import AddGate, MultiplyGate

mulgate = MultiplyGate()
addgate = AddGate()
tanh = Tanh()

class RNNLayer:
    def foward(self, x, prev_a, waa, wax, way):
        x1 = mulgate.forward(waa, prev_a)
        x2 = mulgate.forward(wax, x)
            addgate.forward()
        a = tanh.forward(x1, x2)
        x3 = mulgate.forward(wya, a)
        y = 
        
        