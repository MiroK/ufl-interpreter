from dolfin import VectorElement, TensorElement
from scipy.linalg import svdvals, eigvals, eig
import numpy as np
import iufl, cexpr


def eigw(expr, mesh=None):
    '''
    Vector of eigenvalues for square matrix/singular values for rectangular
    matrix. NOTE: only real part
    '''
    if not hasattr(expr, 'is_CExpr'):
        return eigw(iufl.icompile(expr, None))
    
    # For CExpr of right shape we can proceed
    assert len(expr.ufl_shape) == 2, 'Matrix valued expression expected'
    
    n, m = expr.ufl_shape
    degree = expr.ufl_element().degree()
    family = expr.ufl_element().family()
    cell = expr.ufl_element().cell()
    
    if m == n:
        element = VectorElement(family, cell, degree, n)
        shape = (m, )
        body = lambda x, expr=expr, shape=(m, n): np.real(eigvals(expr(x).reshape(shape)))

        return cexpr.build_cexpr(element, shape, body)
    else:
        element = VectorElement(family, cell, degree, min(m, n))
        shape = (min(m, n), )
        body = lambda x, expr=expr, shape=(m, n): svdvals(expr(x).reshape(shape))

        return cexpr.build_cexpr(element, shape, body)

    
def eigv(expr, mesh=None):
    '''Matrix of eigenvalues[they form ROWS] of square matrix'''
    if not hasattr(expr, 'is_CExpr'):
        return eigv(iufl.icompile(expr, None))
    
    # For CExpr of right shape we can proceed
    assert len(expr.ufl_shape) == 2, 'Matrix valued expression expected'
    
    n, m = expr.ufl_shape
    assert m == n, 'Square matrix expected'
    
    degree = expr.ufl_element().degree()
    family = expr.ufl_element().family()
    cell = expr.ufl_element().cell()
    
    element = TensorElement(family, cell, degree, (n, n))
    shape = (n, n)
    body = lambda x, expr=expr, shape=shape: np.real(eig(expr(x).reshape(shape))[1].T)

    return cexpr.build_cexpr(element, shape, body)