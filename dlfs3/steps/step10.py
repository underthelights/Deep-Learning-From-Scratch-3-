import unittest
import numpy as np



class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
      
      # for every bp., 
      # omit y.grad = np.array(1.0)
      if self.grad is None:
        self.grad = np.ones_like(self.data)
        #np.ones_like : Variable data type  == grad type
      funcs=[self.creator]

      while funcs:
        f = funcs.pop()
        x,y = f.input, f.output
        x.grad = f.backward(y.grad)

        if x.creator is not None:
          funcs.append(x.creator)


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)  # Set parent(function)
        self.input = input
        self.output = output  # Set output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = x **2
        return y
  
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x)*gy
        return gx
      
# use as python fn
x = Variable(np.array(0.5))
f = Square() # create class instance
y = f(x) # call class instance
print(y)
y =Square()(x)
print(y)



def square(x):
  f = Square()
  return f(x)
#   return Square()(x)

def exp(x):
  f = Exp()
  return f(x)
#   return Exp()(x)


import unittest

class SquareTest(unittest.TestCase):
  def tests_forward(self):
    x = Variable(np.array(2.0))
    y = square(x)
    expected = np.array(4.0)
    self.assertEqual(y.data, expected)
unittest.main()

class SquareTest(unittest.TestCase):
  def tests_backward(self):
    x = Variable(np.array(3.0))
    y = square(x)
    y.backward()
    expected = np.array(6.0)
    self.assertEqual(y.data, expected)

# !python -m unittest 

unittest.main()

def numerical_diff(f, x, eps = 1e-4):
  x0 = Variable(x.data - eps)
  x1 = Variable(x.data + eps)
  
  y0 = f(x0)
  y1 = f(x1)

  return (y1.data - y0.data)/(2*eps)

class SquareTest(unittest.TestCase):
  def test_gradient_check(self):
    x = Variable(np.random.rand(1))
    y = square(x)
    y.backward()

    num_grad = numerical_diff(square, x)
    flg = np.allclose(x.grad, num_grad)
    self.assertTrue(flg)

unittest.main