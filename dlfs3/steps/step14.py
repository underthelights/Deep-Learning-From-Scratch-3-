import numpy as np


class Variable:
    
    def cleargrad(self):
        self.grad = None
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
            # outputs에 담긴 미분값들을 list에 담음 (list comprehension)
            gys = [output.grad for output in f.outputs]
            # backprop. of function f
            # list unpack
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                #if gxs is not tuple, convert to tuple
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                # x.grad = gx 
                
                #mistaskes : 출력 쪽 미분 값을 그대로 대입
                # -> 같은 변수를 반복해서 사용하면, 전파되는 미분 값이 덮어씌어짐
                
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                
                if x.creator is not None:
                    funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        #self.input,data 에서 바뀐거
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    f = Square()
    return f(x)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


x = Variable(np.array(3.0))
y = add(x,x)

print('y', y.data)

y.backward()
print('x.grad', x.grad)


x = Variable(np.array(3.0))
y = add(x,x)
y.backward()
print(x.grad)


x = Variable(np.array(3.0))
y = add(add(x,x),x)
y.backward()
print(x.grad)



x = Variable(np.array(3.0))
y = add(x,x)
y.backward()
print(x.grad)

y = add(add(x,x),x)
y.backward()
print(x.grad)

        

x = Variable(np.array(3.0))
y = add(x,x)
y.backward()
print(x.grad)
# clear gradients of x
x.cleargrad()
y = add(add(x,x),x)
y.backward()
print(x.grad)