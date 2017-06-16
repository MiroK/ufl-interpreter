import numpy as np
import math
import dolfin
import ufl


FUNCTION_MAP = {ufl.mathfunctions.Sin: math.sin,
                ufl.mathfunctions.Cos: math.cos}

def expr_eval(expression):
    print expression, type(expression)
    # Terminals
    if isinstance(expression, (dolfin.Function, dolfin.Expression)):
        return lambda x: expression(x)
    if isinstance(expression, dolfin.Constant):
        return lambda x: expression.values()
    if isinstance(expression, (int, float)):
        return lambda x: np.array(x)
    if isinstance(expression, (ufl.algebra.IntValue, ufl.algebra.ScalarValue)):
        return lambda x: np.array(expression.value())

    # Algebra, FIXME:
    if isinstance(expression, ufl.algebra.Sum):
        args = expression.ufl_operands
        first, rest = args[0], args[1]
        print first, rest
        return lambda x: expr_eval(first)(x) + expr_eval(rest)(x)

    if isinstance(expression, ufl.algebra.Product):
        args = expression.ufl_operands
        first, rest = args[0], args[1]
        return lambda x: expr_eval(first)(x) * expr_eval(rest)(x)

    if isinstance(expression, ufl.algebra.Division):
        args = expression.ufl_operands
        first, rest = args[0], args[1]
        return lambda x: expr_eval(first)(x) / expr_eval(rest)(x)

    # Functions,
    if isinstance(expression, ufl.mathfunctions.MathFunction):
        return lambda x: FUNCTION_MAP[type(expression)](expr_eval(expression.ufl_operands[0])(x)) 

    assert False

# ----------------------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *

    u = Constant(3)

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'CG', 1)
    f = Expression('x[0]', degree=1)
    v = interpolate(f, V)

    foo = expr_eval(-sin(u+2*f*cos(v)))
    print foo([0.2, 0.3])

    # FIXME: how to handle tensor-valued functions - det, eig, cofac, ...
    # FIXME: complete functions ...
    # FIXME: things such as CellVolume

    # FIXME: derivatives?
    # FIXME: wrapping as Expression, eval_cell?
    # FIXME: cpp Expression

    # READ SICC: interpreters
