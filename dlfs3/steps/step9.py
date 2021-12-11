import numpy as np
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func
    
    # with Recursion
    def backward(self):
      f = [self.creator]
      if f is not None:
        x = f.input
        x.grad = f.backward(self.grad)
        x.backward()

    def backward(self):
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


x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)
# 3.297442541400256

x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.grad = np.array(1.0)
y.backward()
print(x.grad)
# 3.297442541400256





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

x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward() #backward 호출만으로도 미분값 구해짐
print(x.grad)

