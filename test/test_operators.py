from iufl.operators import eigw, eigv
from iufl import icompile
from dolfin import *
import numpy as np
import pytest


def error(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return np.linalg.norm(a - b)


def test_eig():
    mesh = UnitSquareMesh(10, 10)
        
    W = FunctionSpace(mesh, 'CG', 1)
    g0 = interpolate(Expression('x[0]+2*x[1]', element=W.ufl_element()), W)
    g1 = interpolate(Expression('x[0]-x[1]', element=W.ufl_element()), W)

    A = sym(outer(grad(g0+g1), grad(g1*g0)))

    w = icompile(eigw(A))(0.5, 0.5)
    v = icompile(eigv(A))(0.5, 0.5).reshape((2, 2))

    A = icompile(A)(0.5, 0.5).reshape((2, 2))
    for wi, vi in zip(w, v):
        e = error(A.dot(vi), wi*vi)
        assert near(e, 0.0, 1E-12)
