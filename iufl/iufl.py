from ufl.algorithms.estimate_degrees import estimate_total_polynomial_degree
from ufl.algorithms import extract_unique_elements
from dolfin import FiniteElement, VectorElement, TensorElement
from cexpr import build_cexpr
import lambdas


def icompile(expression, mesh=None, family='Discontinuous Lagrange'):
    '''Expression to CEexpresion'''
    
    element = get_element(expression)
    # Here the choice is made to represent everything in a DG space
    shape = expression.ufl_shape
    cell = element.cell()
    degree = get_degreee(expression)  # Of output

    print expression, '@@', shape, degree
    
    if len(shape) == 0:
        element = FiniteElement(family, cell, degree)
    elif len(shape) == 1:
        element = VectorElement(family, cell, degree, shape[0])
    elif len(shape) == 2:
        element = TensorElement(family, cell, degree, shape=shape)
    else:
        raise TypeError('Scalar/Vector/rank-2 tensor valued expression only! Not %s' % str(shape))
                            
    body = lambdas.lambdify(expression, mesh)
    
    return build_cexpr(element, shape, body)

                            
def get_element(expr):
    return extract_unique_elements(expr)[0]

                            
def get_degreee(expr):
    return estimate_total_polynomial_degree(expr)


# --------------------------------------------------------------------


if __name__ == '__main__':
    from dolfin import *
    parameters['form_compiler']['no-evaluate_basis_derivatives'] = False

    u = Constant(3)

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'CG', 2)
    v = interpolate(Expression('x[0]*x[0]-2*x[1]*x[1]', degree=2), V)
    
    #f = icompile(2*u+sin(u))
    #print f(0.5, 0.5)

    f0 = icompile(grad(grad(2*v)), mesh=mesh)
    print f0(0.5, 0.5), f0.ufl_element()

    #print outer(f0, f0)
    # f1 = icompile(outer(f0, f0))
    #print f1(0.5, 0.5)
    
    #print f.ufl_shape, f(0.5, 0.5)

    
    
