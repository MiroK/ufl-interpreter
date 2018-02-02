# Are the nodes evaluated correctly (simple arguments)
from iufl import icompile
from dolfin import *
import numpy as np
import pytest


def error(a, b): return np.linalg.norm(a - b)


# Sum
def test_Sum():
    mesh = UnitSquareMesh(10, 10)
    
    V = FunctionSpace(mesh, 'CG', 1)
    v = interpolate(Expression('x[0]+x[1]', degree=1), V)
    
    W = FunctionSpace(mesh, 'CG', 2)
    w = interpolate(Expression('x[0]*x[1]', degree=2), W)

    f = icompile(v + w)

    x = 0.3
    y = 0.23234
    e = error(f(x, y), (x+y)+(x*y))
    assert near(e, 0.0), e

# Division
# Product
# Math functions
# Inner
# Dot
# Cross
# Outer
# Det
# Cofac
# Inverse
# Transpose
# Trace
# Sym
# Skew
# Dev
# Conditionals
