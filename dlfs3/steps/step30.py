if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import math
from dezero import Variable, Function
from dezero.utils import plot_dot_graph

from dezero import sin
x = Variable(np.array(1.0))
y = sin(x)
print(y, type(y))