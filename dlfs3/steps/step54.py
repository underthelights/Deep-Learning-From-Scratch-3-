if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import test_mode
import dezero.functions as F

dropout_ratio = 0.6
x = np.ones(10)
mask = np.random.rand(10) > dropout_ratio
y = x*mask
print(y)
scale = 1 - dropout_ratio
y = x*scale
print(y)
mask = np.random.rand(*x.shape) > dropout_ratio
y = x*mask / scale
y=x
print(x)
print('-'*15)
x = np.ones(5)
print(x)

# When training
y = F.dropout(x)
print(y)

# When testing (predicting)
with test_mode():
    y = F.dropout(x)
    print(y)