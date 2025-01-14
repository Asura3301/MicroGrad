import math

class Value:

  def __init__(self, data, _children=(), _op="", label=""):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children) # we will use as a set for efficiency
    self._op = _op   # operation sign of
    self.label = label

  # __repr__ providing us a way to print nicer looking expression in Python, and its not look like cryptic
  def __repr__(self):
    return f"Value(data={self.data})"

  def __add__(self, other):
    """
    addition operation
    """
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), "+")

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward

    return out

  def __mul__(self, other):
    """
    multiplication operation
    """
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
      # we should accumulate these gradients so we should "+=""
      self.grad += other.data * out.grad # local derivative
      other.grad += self.data * out.grad
    out._backward = _backward

    return out

  def __pow__(self, other):
    """
    power operation
    """
    assert isinstance(other, (int, float)) # only support int or float powers
    out = Value(self.data ** other, (self, ), f"**{other}")

    def _backward():
      self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward

    return out

  def __radd__(self, other): # other + self
    return self + other

  def __sub__(self, other): # self - other
    return self + (-other)

  def __rsub__(self, other): # other - self
    return other + (-self)

  def __rmul__(self, other): # other * self
    return self * other

  def __truediv__(self, other): # self / other
    return self * other**(-1)

  def __rtruediv__(self, other): # other / self
    return other * self**(-1)

  def __neg__(self): # -self
    return self * (-1)

  def tanh(self):
    """
    tanh function
    """
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), "tanh")

    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out

  def exp(self):
    """
    Exponential function
    """
    x = self.data
    out = Value(math.exp(x), (self, ), "exp")

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward

    return out
  
  def relu(self):
    """
    Relu function
    """
    x = self.data
    out = Value(0 if x < 0 else x, (self, ), "relu")

    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward

  def backward(self):
    """
    Implementing reversed Topological sort and calculating gradients for each node in DAG/Computational Graph
    """
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1.0

    for node in reversed(topo):
      node._backward()
