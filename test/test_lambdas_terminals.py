# Tests for lead nodes in the expression tree

from dolfin import *
import numpy as np
import pytest
import iufl


def error(a, b): return np.linalg.norm(a - b)


def test_scalar_function_1d():
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, 'CG', 1)
    v = interpolate(Expression('2*x[0]-1', degree=1), V)

    e = error(v(0.3), iufl.icompile(v)(0.3))
    assert near(e, 0.0), e

    
def test_scalar_function_2d():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'CG', 2)
    v = interpolate(Expression('2*x[0]*x[1]-1', degree=2), V)

    e = error(v(0.3, 0.3), iufl.icompile(v)(0.3, 0.3))
    assert near(e, 0.0), e

    
def test_scalar_function_3d():
    mesh = UnitCubeMesh(10, 10, 3)
    V = FunctionSpace(mesh, 'DG', 2)
    v = interpolate(Expression('2*x[0]*x[1]-x[1]*x[2]', degree=2), V)

    e = error(v(0.3, 0.3, 0.1), iufl.icompile(v)(0.3, 0.3, 0.1))
    assert near(e, 0.0), e


def test_vector_function_2d():
    mesh = UnitSquareMesh(10, 10)
    V = VectorFunctionSpace(mesh, 'CG', 2)
    v = interpolate(Expression(('2*x[0]*x[1]-1', 'x[0]*x[0]-x[1]'), degree=2), V)

    e = error(v(0.3, 0.3), iufl.icompile(v)(0.3, 0.3))
    assert near(e, 0.0), e

    
def test_vector_function_3d():
    mesh = UnitCubeMesh(10, 10, 3)
    V = VectorFunctionSpace(mesh, 'DG', 2)

    v = interpolate(Expression(('2*x[0]*x[1]-x[2]',
                                'x[0]*x[0]-x[1]',
                                 'x[2]*x[2]+x[1]*x[0]'), degree=2), V)

    e = error(v(0.3, 0.3, 0.1), iufl.icompile(v)(0.3, 0.3, 0.1))
    assert near(e, 0.0), e

    
def test_tensor_function_2d():
    mesh = UnitSquareMesh(10, 10)
    V = TensorFunctionSpace(mesh, 'CG', 2)
    v = interpolate(Expression((('2*x[0]*x[1]-1', '1'),
                                ('x[0]*x[0]-x[1]', '0')), degree=2), V)

    e = error(v(0.3, 0.3), iufl.icompile(v)(0.3, 0.3))
    assert near(e, 0.0), e

    
def test_tensor_function_3d():
    mesh = UnitCubeMesh(10, 10, 3)
    V = TensorFunctionSpace(mesh, 'DG', 2)

    v = interpolate(Expression((('2*x[0]*x[1]-x[2]', '1', '0'),
                                ('0', 'x[0]*x[0]-x[1]', '-1'),
                                ('-1', '-2', 'x[2]*x[2]+x[1]*x[0]')), degree=2), V)

    e = error(v(0.3, 0.3, 0.1), iufl.icompile(v)(0.3, 0.3, 0.1))
    assert near(e, 0.0), e

# Expressions
def test_Expression():
    v = Expression('2*x[0]-1', degree=1)
    v_ = iufl.icompile(v)

    e = error(v(0.3, 0.2), v_(0.3, 0.2))
    assert near(e, 0.0), e

    v = Expression(('2*x[0]-1', 'x[1]'), degree=1)
    v_ = iufl.icompile(v)

    e = error(v(0.3, 0.2), v_(0.3, 0.2))
    assert near(e, 0.0), e

    v = Expression((('2*x[0]-1', 'x[1]', 'x[2]'),
                    ('x[2]', 'x[1]+x[2]', 'x[1]'),
                    ('1', '2', '3')), degree=1)
    v_ = iufl.icompile(v)

    e = error(v(0.3, 0.2, 0.1), v_(0.3, 0.2, 0.1))
    assert near(e, 0.0), e

# CExpr
def test_CExpression():
    v = iufl.icompile(Expression('2*x[0]-1', degree=1))
    v_ = iufl.icompile(v)

    e = error(v(0.3, 0.2), v_(0.3, 0.2))
    assert near(e, 0.0), e

    v = iufl.icompile(Expression(('2*x[0]-1', 'x[1]'), degree=1))
    v_ = iufl.icompile(v)

    e = error(v(0.3, 0.2), v_(0.3, 0.2))
    assert near(e, 0.0), e

    v = iufl.icompile(Expression((('2*x[0]-1', 'x[1]', 'x[2]'),
                                  ('x[2]', 'x[1]+x[2]', 'x[1]'),
                                  ('1', '2', '3')), degree=1))
    v_ = iufl.icompile(v)

    e = error(v(0.3, 0.2, 0.1), v_(0.3, 0.2, 0.1))
    assert near(e, 0.0), e

# Constant
def test_Constant():
    x, y = 0.1, 0.8
    
    a = Constant(1)
    a_ = iufl.icompile(a)
    e = error(a_(x, y), [1])
    assert near(e, 0.0)
    
    b = Constant((2, 3))
    b_ = iufl.icompile(b)
    e = error(b_(x, y), [2, 3])
    assert near(e, 0.0)
    
    c = Constant(((1, 2), (3, 4)))
    c_ = iufl.icompile(c)
    e = error(c_(x, y), [1, 2, 3, 4])
    assert near(e, 0.0)

# Number
def test_Number():
    x, y = 0.1, 0.8
    
    a = 123.456
    a_ = iufl.icompile(a)
    e = error(a_(x, y), [123.456])
    assert near(e, 0.0)

# Idenity
def test_Identity():
    x, y = 0.1, 0.8
    
    a = Identity(2)
    a_ = iufl.icompile(a)

    mesh = UnitSquareMesh(10, 10)
    V = TensorFunctionSpace(mesh, 'Real', 0)

    a0 = project(a, V)
    
    e = error(a_(x, y), a0(x, y))
    assert near(e, 0.0)

# Spatial coordinate
def test_SpatialCoordinate():
    x, y, z = 0.1, 0.8, 0.23426
    
    mesh = UnitCubeMesh(4, 4, 2)
    a = SpatialCoordinate(mesh)
    a_ = iufl.icompile(a)

    e = error(a_(x, y, z), [x, y, z])
    assert near(e, 0.0)
