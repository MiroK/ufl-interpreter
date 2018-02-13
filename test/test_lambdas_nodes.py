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
    
# Product
def test_Product():
    mesh = UnitCubeMesh(4, 4, 4)
    V = FunctionSpace(mesh, 'CG', 1)

    u = interpolate(Expression('x[0]', degree=1), V)
    v = interpolate(Expression('x[1]', degree=1), V)
    w = interpolate(Expression('x[2]', degree=1), V)

    x, y, z = 0.12344, 0.53423, 0.125214
    f = u*v + Constant(8)*u*w - 2*v*w

    f_ = icompile(f)
    e = error(f_(x, y, z), x*y + 8*x*z - 2*y*z)
    assert near(e, 0.0), e

    # Polymorf *
    A = Constant(((1, 2), (3, 4)))
    v = Constant((1, -2))

    f_ = icompile(dot(A, v))
    
    e = error(f_(0.123, 0.234, 0.12), np.array([-3, -5]))
    assert near(e, 0.0), e
    
# Division
def test_Div():
    mesh = UnitCubeMesh(4, 4, 4)
    V = FunctionSpace(mesh, 'CG', 1)

    u = interpolate(Expression('x[0]', degree=1), V)
    v = interpolate(Expression('x[1]', degree=1), V)
    w = interpolate(Expression('x[2]', degree=1), V)

    x, y, z = 0.12344, 0.53423, 0.125214
    f = u/v + Constant(8)*u/w - 2*v/w

    f_ = icompile(f)
    e = error(f_(x, y, z), x/y + 8*x/z - 2*y/z)
    assert near(e, 0.0), e

# Pow
def test_Pow():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, 'CG', 1)

    u = interpolate(Expression('x[0]+x[1]', degree=1), V)
    v = interpolate(Expression('2*x[0]-3*x[1]', degree=1), V)
    
    f = (u-v)**3

    f_ = icompile(f)

    x, y = 0.789, 0.34244
    e = error(f_(x, y), ((x+y)-(2*x-3*y))**3)
    assert near(e, 0.0), e
    
# Math functions
def test_Fn():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, 'CG', 1)

    u = interpolate(Expression('x[0]+x[1]', degree=1), V)
    v = interpolate(Expression('2*x[0]-3*x[1]', degree=1), V)
    
    f = sqrt((u-v)**2)

    f_ = icompile(f)

    x, y = 0.789, 0.34244
    e = error(f_(x, y), sqrt(((x+y)-(2*x-3*y))**2))
    assert near(e, 0.0), e

    f = bessel_I(2, f_)
    assert icompile(f)(0.2, 0.12) > 0

# Inner
def test_Inner():
    mesh = UnitSquareMesh(4, 4)
    x, y = 0.128945, 0.97035
    
    # Scalar
    V1 = FunctionSpace(mesh, 'CG', 1)
    V2 = FunctionSpace(mesh, 'CG', 2)

    v1 = interpolate(Expression('x[0]+x[1]', degree=1), V1)
    v2 = interpolate(Expression('x[0]*x[1]', degree=2), V2)

    f = inner(v1, v2)
    f_ = icompile(f)
    
    e = error(f_(x, y), (x+y)*(x*y))
    assert near(e, 0.0), e

    # Vector
    V1 = VectorFunctionSpace(mesh, 'CG', 1)
    V2 = VectorFunctionSpace(mesh, 'CG', 2)

    v1 = SpatialCoordinate(mesh)
    v2 = interpolate(Expression(('x[0]*x[1]', 'x[0]-x[1]'), degree=2), V2)

    f = inner(v1, v2)
    f_ = icompile(f)
    
    e = error(f_(x, y), x*(x*y) + y*(x-y))
    assert near(e, 0.0), e

    # Tensor
    A = outer(v1, v2)
    B = outer(v2, v1)

    f = inner(A, B)
    f_ = icompile(f)

    A_ = np.array([[x*(x*y), x*(x-y)], [y*(x*y), y*(x-y)]])
    B_ = np.array([[x*y*(x), x*y*(y)], [(x-y)*x, (x-y)*y]])
    
    e = error(f_(x, y), np.tensordot(A_, B_))
    assert near(e, 0.0), e

# Dot
def test_Dot():
    mesh = UnitSquareMesh(4, 4)
    x, y = 0.128945, 0.97035
    
    # Scalar
    V1 = FunctionSpace(mesh, 'CG', 1)
    V2 = FunctionSpace(mesh, 'CG', 2)

    v1 = interpolate(Expression('x[0]+x[1]', degree=1), V1)
    v2 = interpolate(Expression('x[0]*x[1]', degree=2), V2)

    f = dot(v1, v2)
    f_ = icompile(f)
    
    e = error(f_(x, y), (x+y)*(x*y))
    assert near(e, 0.0), e

    # Vector
    V1 = VectorFunctionSpace(mesh, 'CG', 1)
    V2 = VectorFunctionSpace(mesh, 'CG', 2)

    v1 = SpatialCoordinate(mesh)
    v2 = interpolate(Expression(('x[0]*x[1]', 'x[0]-x[1]'), degree=2), V2)

    f = dot(v1, v2)
    f_ = icompile(f)
    
    e = error(f_(x, y), x*(x*y) + y*(x-y))
    assert near(e, 0.0), e

    # Mat vec
    A = outer(v1, v2)
    B = outer(v2, v1)
    
    A_ = np.array([[x*(x*y), x*(x-y)], [y*(x*y), y*(x-y)]])
    B_ = np.array([[x*y*(x), x*y*(y)], [(x-y)*x, (x-y)*y]])
    v1_ = np.array([x, y])

    f = dot(A, v1)
    f_ = icompile(f)

    e = error(f_(x, y), np.dot(A_, v1_))
    assert near(e, 0.0), e

    f = dot(A, B)
    f_ = icompile(f)

    e = error(f_(x, y), np.dot(A_, B_).flatten())
    assert near(e, 0.0), e
    
# Outer
def test_Outer():
    mesh = UnitSquareMesh(4, 4)
    x, y = 0.128945, 0.97035
    
    # Scalar
    V1 = FunctionSpace(mesh, 'CG', 1)
    V2 = FunctionSpace(mesh, 'CG', 2)

    v1 = interpolate(Expression('x[0]+x[1]', degree=1), V1)
    v2 = interpolate(Expression('x[0]*x[1]', degree=2), V2)

    f = outer(v1, v2)
    f_ = icompile(f)
    
    e = error(f_(x, y), (x+y)*(x*y))
    assert near(e, 0.0), e

    # Vector
    V1 = VectorFunctionSpace(mesh, 'CG', 1)
    V2 = VectorFunctionSpace(mesh, 'CG', 2)

    v1 = SpatialCoordinate(mesh)
    v2 = interpolate(Expression(('x[0]*x[1]', 'x[0]-x[1]'), degree=2), V2)

    # Mat vec
    A = outer(v1, v2)
    A_ = np.array([[x*(x*y), x*(x-y)], [y*(x*y), y*(x-y)]])

    f = icompile(A)
    e = error(f(x, y), A_.flatten())
    assert near(e, 0.0), (e, f_(x, y), A_.flatten())

    B = outer(v2, v1)
    B_ = np.array([[x*y*(x), x*y*(y)], [(x-y)*x, (x-y)*y]])

    f = icompile(B)
    e = error(f(x, y), B_.flatten())
    assert near(e, 0.0), e

    C = outer(A, B)

    f = icompile(C)
    e = error(f(x, y), np.outer(A_, B_).flatten())
    assert near(e, 0.0), e

# Conditionals
def test_Cond():
    mesh = UnitSquareMesh(4, 4)
    x, y = 0.128945, 0.97035
    
    V1 = FunctionSpace(mesh, 'CG', 1)
    V2 = FunctionSpace(mesh, 'CG', 2)

    v1 = interpolate(Expression('x[0]+x[1]', degree=1), V1)
    v2 = interpolate(Expression('x[0]*x[1]', degree=2), V2)
    v1_ = x + y
    v2_ = x*y

    f = conditional(v1 < v2, Constant(1), Constant(2))
    f_ = icompile(f)

    e = error(f_(x, y), 1 if v1_ < v2_ else 2)
    assert near(e, 0.0), e

    f = conditional(v1 <= v1, Constant(1), Constant(2))
    f_ = icompile(f)

    e = error(f_(x, y), 1 if v1_ <= v1_ else 2)
    assert near(e, 0.0), (e, f_(x, y))

    f = conditional(2*v1 > -3*v2, v1+v2, v1*v2)
    f_ = icompile(f)

    e = error(f_(x, y), v1_+v2_ if 2*v1_ > -3*v2_ else v1_*v2_)
    assert near(e, 0.0), (e, f_(x, y))

    f = conditional(Not(2*v1 > -3*v2), v1+v2, v1*v2)
    f_ = icompile(f)

    e = error(f_(x, y), v1_+v2_ if not(2*v1_ > -3*v2_) else v1_*v2_)
    assert near(e, 0.0), (e, f_(x, y))

    f = conditional(And(v1 < Constant(1), v1 > 0), v1+v2, v1*v2)
    f_ = icompile(f)

    e = error(f_(x, y), v1_+v2_ if (v1_ < 1 and v1_ > 0) else v1_*v2_)
    assert near(e, 0.0), (e, f_(x, y))

    f = conditional(Or(v1 < Constant(1), v2 > 0), v1+v2, v1*v2)
    f_ = icompile(f)

    e = error(f_(x, y), v1_+v2_ if (v1_ < 1 or v2_ > 0) else v1_*v2_)
    assert near(e, 0.0), (e, f_(x, y))

# Cross
def test_Cross():
    mesh = UnitCubeMesh(3, 3, 3)
    V = VectorFunctionSpace(mesh, 'CG', 1)

    u = interpolate(Constant((1, 2, 3)), V)
    v = interpolate(Constant((1, 0, 1)), V)

    w = cross(u, v)
    w0 = project(w, V)

    w_ = icompile(w)

    x, y, z = 0.2, 0.123432, 0.2565
    e = error(w_(x, y, z), w0(x, y, z))
    assert near(e, 0.0, 1E-13)
    
# TODO - these nodes need degree handling 
# Det
# Cof
# Inverse
# Transpose
# Trace
# Sym
# Skew
# Dev
