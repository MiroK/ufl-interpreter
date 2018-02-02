from iufl import icompile
from dolfin import *
import numpy as np
import pytest


def error(a, b): return np.linalg.norm(a - b)


def test_grad():
    mesh = UnitSquareMesh(10, 10)
    
    V = FunctionSpace(mesh, 'CG', 2)
    v = interpolate(Expression('x[0]*x[1]', degree=2), V)
    
    f = icompile(grad(v))
    
    x = 0.3
    y = 0.23234
    e = error(f(x, y), np.array([y, x]))
    assert near(e, 0.0), e

# Grad
# Div 
# Curl
