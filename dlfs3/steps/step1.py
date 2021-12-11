#1-1 Class
class Variable:
    def __init__(self, data):
        self.data = data

import numpy as np

data = np.array(1.0)
x = Variable(data)
print(x.data)

x.data = np.array(2.0)
print(x.data)

# 1-3. NumPy n-dim array
print(np.array(1).ndim,
      np.array([1,2,3]).ndim,
      np.array([[1,2,3],[4,5,6]]).ndim)