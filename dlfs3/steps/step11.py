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
    def __call__(self, input):
        # Variable 에서 실제 data를 꺼낸 다음
        # forward method에서 구체적 계산 
        x = input.data
        y = self.forward(x)

        # 계산 결과를 Variable에 넣고
        # creator 처리
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output

        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Function:
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs= [Variable(as_array(y)) for y in ys]
        
        for output in outputs:
            output.set_creator(self)
        
        self.inputs = inputs
        self.outputs = outputs

        return outputs

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, xs):
        x0, x1 = xs 
        y = x0 + x1
        return (y,)



# class Variable:
#     def __init__(self, data):
#         if data is not None:
#             if not isinstance(data, np.ndarray):
#                 raise TypeError('{} is not supported'.format(type(data)))

#         self.data = data
#         self.grad = None
#         self.creator = None

#     def set_creator(self, func):
#         self.creator = func

#     def backward(self):
#         if self.grad is None:
#             self.grad = np.ones_like(self.data)

#         funcs = [self.creator]
#         while funcs:
#             f = funcs.pop()
#             x, y = f.input, f.output
#             x.grad = f.backward(y.grad)

#             if x.creator is not None:
#                 funcs.append(x.creator)


# def as_array(x):
#     if np.isscalar(x):
#         return np.array(x)
#     return x


# class Function:
#     def __call__(self, inputs):
#         xs = [x.data for x in inputs]  # Get data from Variable
#         ys = self.forward(xs)
#         outputs = [Variable(as_array(y)) for y in ys]  # Wrap data

#         for output in outputs:
#             output.set_creator(self)
#         self.inputs = inputs
#         self.outputs = outputs
#         return outputs

#     def forward(self, xs):
#         raise NotImplementedError()

#     def backward(self, gys):
#         raise NotImplementedError()


# class Add(Function):
#     def forward(self, xs):
#         x0, x1 = xs
#         y = x0 + x1
#         return (y,)


# import numpy as np

xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
ys = f(xs) # tuple
y = ys[0]
print(type(ys), type(y), type(f))
print(y.data)