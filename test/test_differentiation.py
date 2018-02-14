from iufl import icompile
from dolfin import *
import numpy as np
import pytest
# Sympy code gen comes in handy
from isympy import *


def error(a, b): return np.linalg.norm(a - b)

# Grad
def test_grad_2d():
    mesh = UnitSquareMesh(2, 2)
    x_, y_ = 0.1223, 0.58596
    x, y = sp.symbols('x[0] x[1]')

    # Scalar
    f = x**2 + 2*y**2 - x*y*x
    grad_f = Grad(f)
    
    V = FunctionSpace(mesh, 'CG', 3)
    v = interpolate(as_expression(f, degree=3), V)

    f_ = icompile(grad(v))

    print sp_eval(grad_f, (x_, y_))
    e = error(f_(x_, y_), sp_eval(grad_f, (x_, y_)))
    assert near(e, 0.0, 1E-13), e

    # Vector
    f_ = icompile(grad(f_), mesh=mesh)
    grad_grad_f = Grad(grad_f)

    e = error(f_(x_, y_), sp_eval(grad_grad_f, (x_, y_)))
    assert near(e, 0.0, 1E-13), e
    
def test_grad_3d():
    mesh = UnitCubeMesh(2, 2, 2)
    x_, y_, z_ = 0.1223, 0.58596, 0.679
    x, y, z = sp.symbols('x[0] x[1] x[2]')

    # Scalar
    f = x**2 + 2*y**2 - x*y*z
    grad_f = Grad(f, 3)
    
    V = FunctionSpace(mesh, 'CG', 3)
    v = interpolate(as_expression(f, degree=3), V)

    f_ = icompile(grad(v))

    print sp_eval(grad_f, (x_, y_, z_))
    e = error(f_(x_, y_, z_), sp_eval(grad_f, (x_, y_, z_)))
    assert near(e, 0.0, 1E-13), e

    # Vector
    f_ = icompile(grad(f_), mesh=mesh)
    grad_grad_f = Grad(grad_f, 3)

    e = error(f_(x_, y_, z_), sp_eval(grad_grad_f, (x_, y_, z_)))
    assert near(e, 0.0, 1E-13), e
    
# Div
def test_div_2d():
    mesh = UnitSquareMesh(2, 2)
    x_, y_ = 0.1223, 0.58596
    x, y = sp.symbols('x[0] x[1]')

    # Vector
    f = Grad(x**2 + 2*y**2 - x*y*x)
    
    V = VectorFunctionSpace(mesh, 'CG', 3)
    v = interpolate(as_expression(f, degree=3), V)

    divv = icompile(div(v))
    divv0 = Div(f)

    e = error(divv(x_, y_), sp_eval(divv0, (x_, y_)))
    assert near(e, 0.0, 1E-13), e

    # Tensor
    f = Grad(f)

    V = TensorFunctionSpace(mesh, 'CG', 3)
    v = interpolate(as_expression(f, degree=1), V)

    divv = icompile(div(v))
    divv0 = Div(f)

    e = error(divv(x_, y_), sp_eval(divv0, (x_, y_)))
    assert near(e, 0.0, 1E-13), e

def test_div_3d():
    mesh = UnitCubeMesh(3, 3, 2)
    
    x_, y_, z_ = 0.1223, 0.58596, 0.231356
    x, y, z = sp.symbols('x[0] x[1] x[2]')
    p_ = (x_, y_, z_)    
    
    # Vector
    f = Grad(x*z + 2*z**2 - x*y*z, 3)
    
    V = VectorFunctionSpace(mesh, 'CG', 3)
    v = interpolate(as_expression(f, degree=3), V)

    divv = icompile(div(v))
    divv0 = Div(f)

    e = error(divv(*p_), sp_eval(divv0, p_))
    assert near(e, 0.0, 1E-13), e

    # Tensor
    f = Grad(f, 3)

    V = TensorFunctionSpace(mesh, 'CG', 3)
    v = interpolate(as_expression(f, degree=1), V)

    divv = icompile(div(v))
    divv0 = Div(f)

    e = error(divv(*p_), sp_eval(divv0, p_))
    assert near(e, 0.0, 1E-13), e

# Curl
def test_curl_2d():
    mesh = UnitSquareMesh(10, 10)
    p_ = (0.1223, 0.58596)
    
    xy = SpatialCoordinate(mesh)
    x, y = xy
    # Scalar
    f = curl(x+y)
    f0 = project(f, VectorFunctionSpace(mesh, 'CG', 1))
    f_ = icompile(f, mesh)
    
    e = error(f_(*p_), f0(*p_))
    assert near(e, 0.0, 1E-13), e

    # Vector
    f = curl(as_vector((-y, x)))
    f0 = project(f)
    f_ = icompile(f, mesh)
    
    e = error(f_(*p_), f0(*p_))
    assert near(e, 0.0, 1E-13), (e, f_(*p_), f0(*p_))

def test_curl_3d():
    mesh = UnitCubeMesh(10, 10, 10)
    p_ = (0.1223, 0.58596, 0.987)
    
    xyz = SpatialCoordinate(mesh)
    x, y, z = xyz

    f = curl(as_vector((-y*y, x*z, y*z)))
    f0 = project(f)
    f_ = icompile(f, mesh)
    
    e = error(f_(*p_), f0(*p_))
    assert near(e, 0.0, 1E-13), (e, f_(*p_), f0(*p_))
