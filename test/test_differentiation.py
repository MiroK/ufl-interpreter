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

def test_dx_1():
    mesh = UnitCubeMesh(2, 2, 2)
    x, y, z = sp.symbols('x[0] x[1] x[2]')
    x_, y_, z_ = 0.1223, 0.58596, 0.679
    p_ = (x_, y_, z_)
    
    # Scalar
    f = x**2 + 2*y**2 - x*y*z
    v = interpolate(as_expression(f, degree=3), FunctionSpace(mesh, 'CG', 3))

    fdx = icompile(v.dx(0))
    fdy = icompile(v.dx(1))
    fdz = icompile(v.dx(2))

    true = sp_eval(Grad(f, 3), p_)
    me = np.array([fdxi(*p_) for fdxi in (fdx, fdy, fdz)])
    e = error(me, true)
    assert near(e, 0.0, 1E-13), (e, me, true)

    # Vector
    x = SpatialCoordinate(mesh)
    grad_f = np.array([icompile(x.dx(i), mesh)(p_) for i in range(3)])

    e = error(grad_f, np.eye(3))
    assert near(e, 0.0, 1E-13), e

    # Tensor
    xx = outer(x, x)
    # First make sure that grad work for this
    me = icompile(grad(xx), mesh)(p_)
    true_s = ([[2*x_, y_, z_], [y_, 0, 0], [z_, 0, 0]],
              [[0, x_, 0], [x_, 2*y_, z_], [0, z_, 0]],
              [[0, 0, x_], [0, 0, y_], [x_, y_, 2*z_]])
    true = np.zeros((3, 3, 3))
    for i, piece in enumerate(true_s): true[:, :, i] = np.array(piece)
    
    # Make sure first that grad of higher works
    e = error(me, true)
    assert near(e, 0.0, 1E-13), (e, me, true)

    # dx
    me = icompile(xx.dx(0), mesh)(p_)
    true = true[:, :, 0]
    e = error(me, true)

    assert near(e, 0.0, 1E-13), (e, me, true)

    
def test_dx_2():
    mesh = UnitCubeMesh(2, 2, 2)
    x, y, z = sp.symbols('x[0] x[1] x[2]')
    x_, y_, z_ = 0.1223, 0.58596, 0.679
    p_ = (x_, y_, z_)
    
    # Scalar
    f = x**2 + 2*y**2 - x*y*z
    v = interpolate(as_expression(f, degree=3), FunctionSpace(mesh, 'CG', 3))

    fdx = icompile(v.dx(0, 1))

    true = sp_eval(f.diff(x, 1).diff(y, 1), p_)
    me = fdx(p_)
    e = error(me, true)
    assert near(e, 0.0, 1E-13), (e, me, true)

    # Vector
    x = SpatialCoordinate(mesh)
    f = as_vector((x[0]**2, x[0]*x[1], x[0]*x[1]*x[2]))

    df = icompile(f.dx(0, 1), mesh)
    df0 = (sp.S(0), sp.S(1), z)

    me = df(p_)
    true = sp_eval(df0, p_)
    e = error(me, true)
    assert near(e, 0.0, 1E-13), (e, me, true)

    # # Tensor
    xx = outer(x, x)
    # x
    true = [[2*x_, y_, z_], [y_, 0, 0], [z_, 0, 0]]
    # z
    true = [[0, 0, 1.], [0, 0, 0], [1., 0, 0]]

    # dx
    me = icompile(xx.dx(0, 2), mesh)(p_)
    e = error(me, true)
    assert near(e, 0.0, 1E-13), (e, me, true)
