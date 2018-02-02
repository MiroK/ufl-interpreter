from ufl.algorithms.estimate_degrees import estimate_total_polynomial_degree
from ufl.algorithms import extract_unique_elements
from ufl.corealg.traversal import traverse_unique_terminals
from dolfin import (FiniteElement, VectorElement, TensorElement, MixedElement,
                    Function, Constant)
from cexpr import build_cexpr
import lambdas


def icompile(expression, mesh=None, family='Discontinuous Lagrange'):
    '''Expression to CEexpresion'''

    print expression
    
    element = get_element(expression)
    # Here the choice is made to represent everything in a DG space
    shape = expression.ufl_shape
    cell = element.cell()
    # FIXME: to this properly!
    degree = element.degree()#get_degreee(expression)  # Of output

    element = construct_element(family, cell, degree, shape)
    
    if mesh is None: mesh = get_mesh(expression)
    body = lambdas.lambdify(expression, mesh)
    
    return build_cexpr(element, shape, body)

                            
def get_element(expr):
    try:
        return extract_unique_elements(expr)[0]
    except IndexError:
        # This typically happens with numbers
        return get_element(Constant(expr))

    
def get_degreee(expr):
    return estimate_total_polynomial_degree(expr)


def get_mesh(expr):
    for arg in traverse_unique_terminals(expr):
        print 'Arg', arg, type(arg)
        if isinstance(arg, Function):
            return arg.function_space().mesh()
    return None


def construct_element(family, cell, degree, shape=()):
    if len(shape) == 0:
        return FiniteElement(family, cell, degree)
    elif len(shape) == 1:
        return VectorElement(family, cell, degree, shape[0])
    elif len(shape) == 2:
        return TensorElement(family, cell, degree, shape=shape)
    else:
        nfirst = shape[0]
        shape = shape[1:]
        component = construct_element(family, cell, degree, shape)

        return MixedElement([component]*nfirst)

    
# --------------------------------------------------------------------


if __name__ == '__main__':
    from operators import eigw, eigv
    from dolfin import *
    parameters['form_compiler']['no-evaluate_basis_derivatives'] = False

    u = Constant(3)

    mesh = UnitSquareMesh(10, 10)
    V = VectorFunctionSpace(mesh, 'CG', 2, 4)
    v0 = interpolate(Expression(('x[0]*x[0]-2*x[1]*x[1]',
                                 'x[1]',
                                 'x[0]+x[1]',
                                 'x[0]*x[1]'), degree=2), V)
    
    v1 = interpolate(Expression(('3*x[0]*x[0]-2*x[1]*x[1]',
                                 'x[1]',
                                 'x[0]+x[1]',
                                 'x[0]*x[1]'), degree=2), V)

    # print icompile(Constant(2))(0.5, 0.5)
    
    #f = icompile(2*u+sin(u))
    #print f(0.5, 0.5)
    W = FunctionSpace(mesh, 'CG', 1)
    g0 = interpolate(Expression('x[0]+2*x[1]', element=W.ufl_element()), W)
    g1 = interpolate(Expression('x[0]-x[1]', element=W.ufl_element()), W)

    f = sym(outer(grad(g0+g1), grad(g1*g0)))
    A = icompile(f)

    w = icompile(eigw(A))(0.5, 0.5)
    v = icompile(eigv(A))(0.5, 0.5).reshape((2, 2))

    A = A(0.5, 0.5).reshape((2, 2))
    for wi, vi in zip(w, v):
        print A.dot(vi) - wi*vi
    
    #g = icompile(bessel_I(1, x))
    #print g(0.2, 0.3)
    
    # f0 = icompile(div(grad(v0) + grad(v1)))
    #f = grad(2*v0) + grad(v1)

    #f0 = icompile(grad(outer(f, f)))
    # print f0(0.23, 0.123), f0.ufl_shape
    #print f0(0.223, 0.124)
    #print f0(0.123, 0.123)

    #print outer(f0, f0)
    # f1 = icompile(outer(f0, f0))
    #print f1(0.5, 0.5)
    
    #print f.ufl_shape, f(0.5, 0.5)