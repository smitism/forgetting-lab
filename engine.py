"""
engine.py - a tiny neural network engine (the same idea as Karpathy's micrograd).

A `Value` is one number that remembers how it was made, so that
`backward()` can work out how much each earlier number (each weight)
affected the final result. That "how much" is the gradient.
"""
import math
import random


class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = data          # the actual number
        self.grad = 0.0           # d(final loss) / d(this number)
        self._parents = parents   # the Values this one was computed from
        self._backward = lambda: None
        self._op = op

    # ---- basic maths. Each one also records how to pass the gradient back ----
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * (self.data ** (k - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    def backward(self):
        """Run the chain rule from this value back to every weight."""
        order, seen = [], set()

        def build(v):
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    build(p)
                order.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()


class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x):
        total = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return total.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """A stack of layers. sizes=[2, 8, 8, 1] means 2 inputs, two hidden layers of 8, 1 output."""

    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    # ---- helpers for plotting: copy weights out so numpy can run the network fast ----
    def export(self):
        out = []
        for layer in self.layers:
            W = [[w.data for w in n.w] for n in layer.neurons]   # (n_out, n_in)
            b = [n.b.data for n in layer.neurons]
            out.append((W, b))
        return out
