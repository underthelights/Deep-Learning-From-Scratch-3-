import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
  def __call__(self, *inputs):
    # *inputs : immutable length variables
    xs = [x.data for x in inputs]
    ys = self.forward(xs)

    outputs = [Variable(as_array(y)) for y in ys]

    for output in outputs:
      output.set_creator(self)

    self.inputs = inputs
    self.outputs = outputs

    return outputs if len(outputs)>1 else outputs[0]
    # list 원소가 하나 -> return first element
    
    
# use asterisk as an immutable length var
def f(*x):
  print(x)

print(f(1,2,3), f(1,2,3,4,5))
#(1,2,3), (1,2,3,4,5)


class Add(Function):
    def forward(self, xs):
        #factor들이 list로 전달
        x0, x1 = xs 
        y = x0 + x1
        #tuple로 return
        return (y,)


x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
f = Add()
y = f(x0, x1)
print(y.data)


class Function:
  def __call__(self, *inputs):
    xs = [x.data for x in inputs]
    #list unpack :
    ys = self.forward(*xs) #=self.forward(x0, x1)
    if not isinstance(ys, tuple):
      ys = (ys,) #list가 tuple이 아닌 경우 tuple로 return
    outputs = [Variable(as_array(y)) for y in ys]

    for output in outputs:
      output.set_creator(self)

    self.inputs = inputs
    self.outputs = outputs

    return outputs if len(outputs)>1 else outputs[0]

#2nd refinement : better use of fn
class Add(Function):
  def forward(self, x0, x1):
    y = x0+x1
    return y
    
# use Add class as Python Fn   
def add(x0, x1):
  return Add()(x0, x1)

x0 = Variable(np.array(2))
x1 = Variable(np.array(3))

y = add(x0, x1)
print(y.data)

#multiplication
#division

class Mult(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

def mult(x0, x1):
    return Mult()(x0, x1)

x0 = Variable(np.array(2))
x1 = Variable(np.array(3))

y = mult(x0, x1)
print(y.data)

class Divi(Function):
    def forward(self, x0, x1):
        y = x0/x1
        return y

def divv(x0, x1):
    return Divi()(x0, x1)
x0 = Variable(np.array(9))
x1 = Variable(np.array(3))

y = divv(x0, x1)
print(y.data)
        