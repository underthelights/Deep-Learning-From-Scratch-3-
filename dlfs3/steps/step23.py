import sys

# import dezero
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# sys.path.append('.')
#core_simple.py 파일은 parent directory에 위치해 있기 때문에
#sys.path에 해당 module이 위치한 경로, parent directory('..') 를 추가해주어야 함.

import numpy as np
from dezero.core_simple import Variable

x = Variable(np.array(2.0))
print(x)

y = -x
print(y)  # variable(-2.0)

y1 = 2.0 - x
y2 = x - 1.0
print(y1)  # variable(0.0)
print(y2)  # variable(1.0)

y = 3.0 / x
print(y)  # variable(1.5)

y = x ** 3
y.backward()
print(y)  # variable(8.0)



x = Variable(np.array(1.0))
y = (x+3)**2
y.backward()

print(y, x.grad)
    