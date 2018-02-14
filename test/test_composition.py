# Some more complicated expression
from iufl import icompile
from dolfin import *
import numpy as np
import pytest
# Sympy code gen comes in handy
from isympy import *
import sympy as sp


def error(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return np.linalg.norm(a - b)



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

    
def test_2():
    mesh = UnitSquareMesh(10, 10)

    W = VectorFunctionSpace(mesh, 'CG', 2)
    w = interpolate(Expression(('x[0]*x[1]', 'x[0]*x[0]'), degree=2), W)

    foo = icompile(div(w) - tr(grad(w)))
    e = max(abs(foo(p)) for p in np.random.rand(10, 2))

    assert near(e, 0.0), e

    
def test_3():
    mesh = UnitSquareMesh(10, 10)

    W = FunctionSpace(mesh, 'CG', 3)
    w = interpolate(Expression('x[0]*x[1]*2*x[2]', degree=3), W)

    foo = icompile(div(grad(w)) - sum(w.dx(i, i) for i in range(2)))
    e = max(abs(foo(p)) for p in np.random.rand(10, 2))

    assert near(e, 0.0), e
