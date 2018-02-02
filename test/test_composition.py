# Some more complicated expression
from iufl import icompile
from dolfin import *
import numpy as np
import pytest


def error(a, b): return np.linalg.norm(a - b)


# Sum
def test_1():
    mesh = UnitSquareMesh(10, 10)
    
    V = FunctionSpace(mesh, 'CG', 1)
    v = interpolate(Expression('x[0]+2*x[1]', degree=1), V)
    
    W = FunctionSpace(mesh, 'CG', 2)
    w = interpolate(Expression('x[0]*x[1]', degree=2), W)

    f = icompile(grad(v*w))

    x = 0.3
    y = 0.23234
    e = error(f(x, y), np.array([2*x*y + 2*y**2, x**2 + 4*x*y]))
    assert near(e, 0.0, 1E-15), e
