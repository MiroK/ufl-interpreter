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

# Indexing
def test_Indexed():
    mesh = UnitSquareMesh(10, 10)

    V = FunctionSpace(mesh, 'CG', 1)
    v0 = interpolate(Expression('2*x[0]+x[1]', degree=1), V)
    v1 = interpolate(Expression('x[0]-x[1]', degree=1), V)
    v = as_vector((v0, v1))
    
    iv0 = icompile(v[0])
    e = error(iv0(0.5, 0.5), v0(0.5, 0.5))
    assert near(e, 0.0), e

    iv1 = icompile(v[1])
    e = error(iv1(0.5, 0.5), v1(0.5, 0.5))
    assert near(e, 0.0), e

    # ---
    
    v = as_tensor(((v0, Constant(1)), (-Constant(2), v1)))
    iv00 = icompile(v[0, 0])
    e = error(iv00(0.5, 0.5), v0(0.5, 0.5))
    assert near(e, 0.0), e

    iv01 = icompile(v[0, 1]) 
    e = error(iv01(0.5, 0.5), 1.0)
    assert near(e, 0.0), e

    iv10 = icompile(v[1, 0]) 
    e = error(iv10(0.5, 0.5), -2.0)
    assert near(e, 0.0), e

    iv11 = icompile(v[1, 1])
    e = error(iv11(0.5, 0.5), v1(0.5, 0.5))
    assert near(e, 0.0), e
 
   # ----
    # v0 1
    # 1  v1
    t = as_vector((v[0, :], v[:, 1]))
    x = icompile(t[0, :])
    e = error(x(0.5, 0.5), np.array([v0(0.5, 0.5), 1]))
    assert near(e, 0.0), e

    x = icompile(t[1, :])
    e = error(x(0.35, 0.25), np.array([1, v1(0.35, 0.25)]))
    assert near(e, 0.0), e

    x = icompile(t[1, :])
    e = error(x(0.35, 0.25), np.array([1, v1(0.35, 0.25)]))
    assert near(e, 0.0), e

    print '='*79
    T = Constant(((1, 2), (3, 4)))
    #x = as_vector((T[0, 0], T[1, 1]))
    x = diag(T)
    print 'x', type(x), x.ufl_operands
    x = icompile(x)
    print '<<<', x(0.2, 0.2)
    #e = error(x(0.35, 0.25), np.array([v0(0.35, 0.25), v1(0.35, 0.25)]))
    #assert near(e, 0.0), e

test_Indexed()
