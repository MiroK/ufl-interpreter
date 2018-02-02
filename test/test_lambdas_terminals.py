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
    
# TODO
# Expressions
# CExpr
# Constant
# Number
# Idenity
# Zero
# Spatial coordinate
