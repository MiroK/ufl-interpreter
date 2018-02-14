import sympy as sp
from sympy.printing import ccode
from dolfin import Expression


def expr_body(expr, **kwargs):
    if not hasattr(expr, '__len__'):
        # Defined in terms of some coordinates
        xyz = set(sp.symbols('x[0], x[1], x[2]'))
        xyz_used = xyz & expr.free_symbols
        assert xyz_used <= xyz
        # Expression params which need default values
        params = (expr.free_symbols - xyz_used) & set(kwargs.keys())
        # Body
        expr = ccode(expr).replace('M_PI', 'pi')
        # Default to zero
        kwargs.update(dict((str(p), 0.) for p in params))
        # Convert
        return expr
    # Vectors, Matrices as iterables of expressions
    else:
        return tuple(expr_body(e, **kwargs) for e in expr)

    
def Grad(f, gdim=None):
    xyz = list(sp.symbols('x[0], x[1], x[2]'))
    # Scalar
    if not isinstance(f, tuple):
        if gdim is None:
            gdim = 3 if xyz[-1] in f.free_symbols else 2
        return tuple(f.diff(xj, 1) for xj in xyz[:gdim])
    # Recurse
    else:
        return tuple(Grad(fi, gdim) for fi in f)

    
def Div(f):
    xyz = list(sp.symbols('x[0], x[1], x[2]'))
    # Vector
    try:
        return sum(fi.diff(xi) for fi, xi in zip(f, xyz))
    # Recurse
    except AttributeError:
        return tuple(Div(fj) for fj in f)

    
def sp_eval(f, coords):
    xyz = list(sp.symbols('x[0], x[1], x[2]'))
    try:
        return [float(f.subs({x: value for x, value in zip(xyz, coords)}))]
    except AttributeError:
        return sum((sp_eval(fi, coords) for fi in f), [])
    

def as_expression(expr, degree=4, **kwargs):
    '''Turns sympy expressions to Dolfin expressions.'''
    return Expression(expr_body(expr), degree=degree, **kwargs)
