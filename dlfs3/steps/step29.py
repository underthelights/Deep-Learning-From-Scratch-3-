# import matplotlib.pyplot as plt 
# import numpy as np 

# x = np.linspace(-10 , 10, 100)
# y = np.sin(x) 
# plt.plot(x, y, marker="x")
# plt.show()

# cd \Documents\DLFS\steps
# python step27.py

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import math
from dezero import Variable, Function
from dezero.utils import plot_dot_graph

def f(x):
    y = x **4 - 2*x**2
    return y

def gx2(x):
    return 12*x**2-4

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i,x)
    
    y=f(x)
    x.cleargrad()
    y.backward()
    
    x.data -= x.grad/gx2(x.data)
    if x.data ==1:
        
        print('### converge after {}-th iterations ###'.format(i))
        break