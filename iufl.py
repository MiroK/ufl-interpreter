import numpy as np
import math
import dolfin
import ufl
from ufl.conditional import LE, GE, LT, GT, EQ, NE


# Representation of ufl nodes that are MathFunctions
# FIXME: Bessel functions are missing
FUNCTION_MAP = {ufl.mathfunctions.Sin:   math.sin,
                ufl.mathfunctions.Cos:   math.cos,
                ufl.mathfunctions.Sqrt:  math.sqrt,
                ufl.mathfunctions.Exp:   math.exp,
                ufl.mathfunctions.Ln:    math.log,
                ufl.mathfunctions.Tan:   math.tan,
                ufl.mathfunctions.Sinh:  math.sinh,
                ufl.mathfunctions.Cosh:  math.cosh,
                ufl.mathfunctions.Tanh:  math.tanh,
                ufl.mathfunctions.Asin:  math.asin,
                ufl.mathfunctions.Acos:  math.acos,
                ufl.mathfunctions.Atan:  math.atan,
                ufl.mathfunctions.Atan2: math.atan2,
                ufl.mathfunctions.Erf:   math.erf}

def expr_eval(expression):
    '''Compile UFL expression into lambda taking spatial point.'''
    ######################################################################################
    # Terminals
    ######################################################################################
    if isinstance(expression, (dolfin.Function, dolfin.Expression)):
        return lambda x: expression(x)

    if isinstance(expression, dolfin.Constant):
        return lambda x: expression.values().reshape(expression.ufl_shape)

    if isinstance(expression, (int, float)):
        return lambda x: np.array(expression)

    if isinstance(expression, (ufl.algebra.IntValue, ufl.algebra.ScalarValue)):
        return lambda x: np.array(expression.value())

    if isinstance(expression, ufl.constantvalue.Identity):
        return lambda x: np.eye(expression.ufl_shape[0])

    if isinstance(expression, ufl.constantvalue.Zero):
        return lambda x: np.zeros(expression.ufl_shape)

    if isinstance(expression, ufl.geometry.SpatialCoordinate): # Idenity
        return lambda x: np.array(x) if not isinstance(x, np.ndarray) else x

    ######################################################################################
    # Algebra
    ######################################################################################
    if isinstance(expression, ufl.algebra.Sum):
        args = expression.ufl_operands
        first, second = args[0], args[1]
        return lambda x: expr_eval(first)(x) + expr_eval(second)(x)

    if isinstance(expression, ufl.algebra.Division):
        args = expression.ufl_operands
        first, second = args[0], args[1]
        return lambda x: expr_eval(first)(x) / expr_eval(second)(x)

    if isinstance(expression, ufl.algebra.Product):
        args = expression.ufl_operands
        first, second = args[0], args[1]
        return lambda x: expr_eval(first)(x) * expr_eval(second)(x)

    ######################################################################################
    # Functions
    ######################################################################################
    if isinstance(expression, ufl.mathfunctions.MathFunction):
        return lambda x: FUNCTION_MAP[type(expression)](expr_eval(expression.ufl_operands[0])(x)) 

    ######################################################################################
    # Tensor algebra
    ######################################################################################
    if isinstance(expression, ufl.tensoralgebra.Inner):
        args = expression.ufl_operands
        first, second = args[0], args[1]
        # Scalars
        if first.ufl_shape == ():
            return lambda x: expr_eval(first)(x) * expr_eval(second)(x)
        # Tensors
        else:
            return lambda x: np.inner(expr_eval(first)(x), expr_eval(second)(x))

    if isinstance(expression, ufl.tensoralgebra.Dot):
        args = expression.ufl_operands
        first, second = args[0], args[1]
        # Scalars
        if first.ufl_shape == ():
            return lambda x: expr_eval(first)(x) * expr_eval(second)(x)
        # Tensors
        else:
            return lambda x: np.dot(expr_eval(first)(x), expr_eval(second)(x))

    if isinstance(expression, ufl.tensoralgebra.Cross):
        args = expression.ufl_operands
        first, second = args[0], args[1]

        return lambda x: np.cross(expr_eval(first)(x), expr_eval(second)(x))

    if isinstance(expression, ufl.tensoralgebra.Outer):
        args = expression.ufl_operands
        first, second = args[0], args[1]

        return lambda x: np.outer(expr_eval(first)(x), expr_eval(second)(x))

    if isinstance(expression, ufl.tensoralgebra.Determinant):
        args = expression.ufl_operands
        first = args[0]

        return lambda x: np.linalg.det(expr_eval(first)(x))

    if isinstance(expression, ufl.tensoralgebra.Inverse):
        args = expression.ufl_operands
        first = args[0]

        return lambda x: np.linalg.inv(expr_eval(first)(x))

    if isinstance(expression, ufl.tensoralgebra.Transposed):
        args = expression.ufl_operands
        first = args[0]

        return lambda x: expr_eval(first)(x).T

    if isinstance(expression, ufl.tensoralgebra.Trace):
        args = expression.ufl_operands
        first = args[0]

        return lambda x: np.trace(expr_eval(first)(x))

    if isinstance(expression, ufl.tensoralgebra.Sym):
        args = expression.ufl_operands
        first = args[0]

        return lambda x: (lambda A: (A + A.T)/2)(expr_eval(first)(x))

    if isinstance(expression, ufl.tensoralgebra.Skew):
        args = expression.ufl_operands
        first = args[0]

        return lambda x: (lambda A: (A - A.T)/2)(expr_eval(first)(x))

    if isinstance(expression, ufl.tensoralgebra.Deviatoric):
        args = expression.ufl_operands
        first = args[0]

        return lambda x: (lambda A: A - np.trace(A))(expr_eval(first)(x))

    if isinstance(expression, ufl.tensoralgebra.Cofactor):
        args = expression.ufl_operands
        first = args[0]

        return lambda x: (lambda A: np.det(A)*np.inv(A))(expr_eval(first)(x))

    ######################################################################################
    # Conditionals
    ######################################################################################
    if isinstance(expression, EQ):
        args = expression.ufl_operands
        first, second = args

        return lambda x: expr_eval(first)(x) == expr_eval(second)(x)

    if isinstance(expression, NE):
        args = expression.ufl_operands
        first, second = args

        return lambda x: expr_eval(first)(x) != expr_eval(second)(x)

    if isinstance(expression, GT):
        args = expression.ufl_operands
        first, second = args

        return lambda x: expr_eval(first)(x) > expr_eval(second)(x)

    if isinstance(expression, LT):
        args = expression.ufl_operands
        first, second = args

        return lambda x: expr_eval(first)(x) < expr_eval(second)(x)

    if isinstance(expression, GE):
        args = expression.ufl_operands
        first, second = args

        return lambda x: expr_eval(first)(x) >= expr_eval(second)(x)

    if isinstance(expression, LE):
        args = expression.ufl_operands
        first, second = args

        return lambda x: expr_eval(first)(x) <= expr_eval(second)(x)

    if isinstance(expression, ufl.operators.AndCondition):
        args = expression.ufl_operands
        first, second = args

        return lambda x: expr_eval(first)(x) and expr_eval(second)(x)

    if isinstance(expression, ufl.operators.OrCondition):
        args = expression.ufl_operands
        first, second = args

        return lambda x: expr_eval(first)(x) or expr_eval(second)(x)

    if isinstance(expression, ufl.operators.NotCondition):
        arg = expression.ufl_operands[0]

        return lambda x: not(expr_eval(first)(x))

    if isinstance(expression, ufl.operators.Conditional):
        cond, true_v, false_v = expression.ufl_operands

        return lambda x: expr_eval(true_v)(x) if expr_eval(cond)(x) else expr_eval(false_v)(x)

    assert False

# ----------------------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *

    u = Constant(3)

    # mesh = UnitSquareMesh(10, 10)
    # V = FunctionSpace(mesh, 'CG', 1)
    # f = Expression('x[0]', degree=1)
    # v = interpolate(f, V)

    A = Constant(((1, 0), (2, 3)))
    B = Constant(((2, 0), (2, 3)))
    v = Constant((1, 0))

    f = expr_eval(det(inv(sym(A)+skew(B)+outer(v, v)+dot(A, B))))
    print f((0.2, 0.3))
    f = expr_eval(conditional(gt(1, 2), dot(A, B)+Identity(2), A))
    print f((0.2, 0.3))

    # FIXME: as_vector, as_matrix, Indexed
    # FIXME: things such as CellVolume

    # FIXME: differentiation
    # FIXME: wrapping as Expression, eval_cell?, cpp Expression

    # READ SICC: interpreters
