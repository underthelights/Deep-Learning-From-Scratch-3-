if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F

x = np.array([[1,2,3], [4,5,6]])
y = np.reshape(x, (6,))
print(x, y)

x = Variable(np.array([[0, 1, 2], [3, 4, 5]]))
y = F.reshape(x, (6,))  # y = x.reshape(6)
y.backward(retain_grad=True)
print(x.grad)


x = np.random.randn(1,2,3)
y = x.reshape((2,3))
y = x.reshape([2,3])
y = x.reshape(2,3)

x = np.array([[2,3,4], [5,6,7]])
y = np.transpose(x)
print(y)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)  # y = x.T
y.backward()
print(x.grad)


A,B,C,D = 0,1,2,3
x = np.random.randn(A,B,C,D)
y = x.transpose(1,0,3,2)
print(x,y)