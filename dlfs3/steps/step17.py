class obj:
    pass

def f(x):
    print(x)

a = obj()
b = obj()
c = obj()

a.b = b
b.c = c
c.a = a

a = b= c =None

#gc : garbage collection
import gc


def test():
    class A:
        pass

    class B:
        def __init__(self, obj): 
            self.obj = obj

    a = A()
    b = B(a)

    gc.collect() # make sure all garbage cleared before collecting referrers.    
    print( gc.get_referents(b))

test()
#  b라는 변수는 현재 'obj'라는 멤버변수를 가지고 있고 B라는 클래스로 부터 만들어 졌으므로 아래와 같이 출력된다.
#[{'obj': <__main__.A object at 0x7fa49038e630>}, <class '__main__.B'>]



import weakref
import numpy as np

a = np.array([1,2,3])
b = weakref.ref(a)

print('\n\n')
print('weakref')
print(b)
print(b())

a = None
print('invalid ref')
print(b)
print(b())

print('\n\n')
import weakref
import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)


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

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        # weak reference
        self.outputs = [weakref.ref(output) for output in outputs]
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
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)




for i in range(10):
    x = Variable(np.random.randn(10000))  # big data
    y = square(square(square(x)))

print(x,y)

#memory profiler