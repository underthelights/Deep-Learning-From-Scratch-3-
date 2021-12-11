import numpy as np

class Variable:
    def __init__(self, data):

        if data is not None:
          if not isinstance(data, np.ndarray):
            raise TypeError('{}는 지원되지 않습니다'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func
    
    # with Recursion
    # def backward(self):
    #   f = self.creator
    #   if f is not None:
    #     x = f.input
    #     x.grad = f.backward(self.grad)
    #     x.backward()

    def backward(self):
      # y.grad = np.array(1.0) 생략
      # 
      if self.grad is None:
        self.grad = np.ones_like(self.data)

      funcs=[self.creator]

      while funcs:
        f = funcs.pop()
        x,y = f.input, f.output
        x.grad = f.backward(y.grad)

        if x.creator is not None:
          funcs.append(x.creator)

x = Variable(np.array(1.0))
x = Variable(None)
# x = Variable(1.0)
# TypeError: <class 'float'>는 지원되지 않습니다

x = np.array([1.0])
y = x **2

print(type(x), x.ndim)
print(type(y))

x = np.array(1.0)
y = x **2

# <class 'numpy.ndarray'> 1
# <class 'numpy.ndarray'>

print(type(x), x.ndim)
print(type(y))

# <class 'numpy.ndarray'> 0
# <class 'numpy.float64'>

print(type(np.float64(1.0)), type(2.0)) 
print("\n")
print(type(np.array(1.0)), type(np.array([1,2,3])))
print(np.isscalar(np.float64(1.0)), np.isscalar(2.0), np.isscalar(np.array(1.0)), np.isscalar(np.array([1,2,3])))


def as_array(x):
  if np.isscalar(x):
    return np.array(x)
  return x

class Function:
  def __call__ (self, input):
    x = input.data
    y = self.forward(x)
    
    #result of propagation -> cover with Variable
    #output : always ndarray instance
    output = Variable(as_array(y))
    output.set_creator(self)

    self.input = input
    self.output = output

    return output
