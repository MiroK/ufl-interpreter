from ufl.conditional import LE, GE, LT, GT, EQ, NE
import differentiation
import iufl
import numpy as np
import dolfin
import math
import ufl

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


def lambdify(expression, mesh=None):
    '''Compile UFL expression into lambda taking spatial point.'''
    ##################################################################
    # Terminals
    ##################################################################
    if isinstance(expression, (dolfin.Function, dolfin.Expression)):
        return lambda x, expression=expression: expression(x)

    if isinstance(expression, dolfin.Constant):
        return lambda x, expression=expression: expression.values().reshape(expression.ufl_shape)

    if isinstance(expression, (int, float)):
        return lambda x, expression=expression: np.array(expression)

    if isinstance(expression, (ufl.algebra.IntValue, ufl.algebra.ScalarValue)):
        return lambda x, expression=expression: np.array(expression.value())

    if isinstance(expression, ufl.constantvalue.Identity):
        return lambda x, expression=expression: np.eye(expression.ufl_shape[0])

    if isinstance(expression, ufl.constantvalue.Zero):
        return lambda x, expression: np.zeros(expression.ufl_shape)

    if isinstance(expression, ufl.geometry.SpatialCoordinate): # Idenity
        return lambda x: np.array(x) if not isinstance(x, np.ndarray) else x

    ##################################################################
    # Algebra
    ##################################################################
    if isinstance(expression, ufl.algebra.Sum):
        args = expression.ufl_operands
        first, second = args[0], args[1]
        return lambda x, first=first, second=second:\
            lambdify(first, mesh)(x) + lambdify(second, mesh)(x)

    if isinstance(expression, ufl.algebra.Division):
        args = expression.ufl_operands
        first, second = args[0], args[1]
        return lambda x, first=first, second=second:\
            lambdify(first, mesh)(x) / lambdify(second, mesh)(x)

    if isinstance(expression, ufl.algebra.Product):
        args = expression.ufl_operands
        first, second = args[0], args[1]
        return lambda x, firs=first, second=second:\
            lambdify(first, mesh)(x) * lambdify(second, mesh)(x)

    ##################################################################
    # Functions
    ##################################################################
    if isinstance(expression, ufl.mathfunctions.MathFunction):
        return lambda x, expression=expression:\
            FUNCTION_MAP[type(expression)](lambdify(expression.ufl_operands[0], mesh)(x)) 

    ##################################################################
    # Tensor algebra
    ##################################################################
    if isinstance(expression, ufl.tensoralgebra.Inner):
        args = expression.ufl_operands
        first, second = args[0], args[1]
        # Scalars
        if first.ufl_shape == ():
            return lambda x, first=first, second=second:\
                lambdify(first, mesh)(x) * lambdify(second, mesh)(x)
        # Tensors
        else:
            return lambda x, first=first, second=second:\
                np.inner(lambdify(first, mesh)(x), lambdify(second, mesh)(x))

    if isinstance(expression, ufl.tensoralgebra.Dot):
        args = expression.ufl_operands
        first, second = args[0], args[1]
        # Scalars
        if first.ufl_shape == ():
            return lambda x, first=first, second=second:\
                lambdify(first, mesh)(x) * lambdify(second, mesh)(x)
        # Tensors
        else:
            return lambda x, first=first, second=second: \
                np.dot(lambdify(first, mesh)(x), lambdify(second, mesh)(x))

    if isinstance(expression, ufl.tensoralgebra.Cross):
        args = expression.ufl_operands
        first, second = args[0], args[1]

        return lambda x, first=first, second=second:\
            np.cross(lambdify(first, mesh)(x), lambdify(second, mesh)(x))

    if isinstance(expression, ufl.tensoralgebra.Outer):
        args = expression.ufl_operands
        first, second = args[0], args[1]

        return lambda x, first=first, second=second:\
            np.outer(lambdify(first, mesh)(x), lambdify(second, mesh)(x))

    if isinstance(expression, ufl.tensoralgebra.Determinant):
        args = expression.ufl_operands
        first = args[0]

        return lambda x, first=first: np.linalg.det(lambdify(first, mesh)(x))

    if isinstance(expression, ufl.tensoralgebra.Inverse):
        args = expression.ufl_operands
        first = args[0]

        return lambda x, first=first: np.linalg.inv(lambdify(first, mesh)(x))

    if isinstance(expression, ufl.tensoralgebra.Transposed):
        args = expression.ufl_operands
        first = args[0]

        return lambda x, first=first: lambdify(first, mesh)(x).T

    if isinstance(expression, ufl.tensoralgebra.Trace):
        args = expression.ufl_operands
        first = args[0]

        return lambda x, first=first: np.trace(lambdify(first, mesh)(x))

    if isinstance(expression, ufl.tensoralgebra.Sym):
        args = expression.ufl_operands
        first = args[0]

        return lambda x, first=first: (lambda A: (A + A.T)/2)(lambdify(first, mesh)(x))

    if isinstance(expression, ufl.tensoralgebra.Skew):
        args = expression.ufl_operands
        first = args[0]

        return lambda x, first=first: (lambda A: (A - A.T)/2)(lambdify(first, mesh)(x))

    if isinstance(expression, ufl.tensoralgebra.Deviatoric):
        args = expression.ufl_operands
        first = args[0]

        return lambda x, first=first: (lambda A: A - np.trace(A))(lambdify(first, mesh)(x))

    if isinstance(expression, ufl.tensoralgebra.Cofactor):
        args = expression.ufl_operands
        first = args[0]

        return lambda x, first=first: (lambda A: np.det(A)*np.inv(A))(lambdify(first, mesh)(x))

    ##################################################################
    # Conditionals
    ##################################################################
    if isinstance(expression, EQ):
        args = expression.ufl_operands
        first, second = args

        return lambda x, first=first, second=second:\
            lambdify(first, mesh)(x) == lambdify(second, mesh)(x)

    if isinstance(expression, NE):
        args = expression.ufl_operands
        first, second = args

        return lambda x, first=first, second=second:\
            lambdify(first, mesh)(x) != lambdify(second, mesh)(x)

    if isinstance(expression, GT):
        args = expression.ufl_operands
        first, second = args

        return lambda x, first=first, second=second:\
            lambdify(first, mesh)(x) > lambdify(second, mesh)(x)

    if isinstance(expression, LT):
        args = expression.ufl_operands
        first, second = args

        return lambda x, first=first, second=second:\
            lambdify(first, mesh)(x) < lambdify(second, mesh)(x)

    if isinstance(expression, GE):
        args = expression.ufl_operands
        first, second = args

        return lambda x, first=first, second=second:\
            lambdify(first, mesh)(x) >= lambdify(second, mesh)(x)

    if isinstance(expression, LE):
        args = expression.ufl_operands
        first, second = args

        return lambda x, first=first, second=second:\
            lambdify(first, mesh)(x) <= lambdify(second, mesh)(x)

    if isinstance(expression, ufl.operators.AndCondition):
        args = expression.ufl_operands
        first, second = args

        return lambda x, first=first, second=second:\
            lambdify(first, mesh)(x) and lambdify(second, mesh)(x)

    if isinstance(expression, ufl.operators.OrCondition):
        args = expression.ufl_operands
        first, second = args

        return lambda x, first=first, second=second:\
            lambdify(first, mesh)(x) or lambdify(second, mesh)(x)

    if isinstance(expression, ufl.operators.NotCondition):
        arg = expression.ufl_operands[0]

        return lambda x, first=first: not(lambdify(first, mesh)(x))

    if isinstance(expression, ufl.operators.Conditional):
        cond, true_v, false_v = expression.ufl_operands

        return lambda x, cond=cond, true_v=true_v, false_v=false_v:\
            lambdify(true_v, mesh)(x) if lambdify(cond, mesh)(x) else lambdify(false_v, mesh)(x)

    ##################################################################
    # Indexing (limited)
    ##################################################################
    if isinstance(expression, ufl.indexed.Indexed):
        indexed, index = expression.ufl_operands

        return lambda x, indexed=indexed, index=index:\
            lambdify(indexed, mesh)(x)[extract_index(index)]
    
    ##################################################################
    # Differentation (limited)
    ##################################################################
    if isinstance(expression, ufl.differentiation.Grad):
        operand = expression.ufl_operands[0]
        
        # It is mandatory that FFC has generated deriv eval code
        assert not dolfin.parameters['form_compiler']['no-evaluate_basis_derivatives']

        # Primitives
        if isinstance(operand, dolfin.Function):
            # We are about to take the derivative so it better make sense
            assert operand.ufl_element().degree() >= 1

            return differentiation.eval_grad_foo(operand)
        # Needs mesh!
        elif isinstance(operand, dolfin.Expression) or hasattr(expression, 'is_CExpr'):
            # We are about to take the derivative so it better make sense
            assert iufl.get_element(operand).degree() >= 1
            
            return differentiation.eval_grad_expr(operand, mesh)
        # Composite
        else:
            return differentiation.eval_grad_expr(iufl.icompile(operand, mesh), mesh)

    # if isinstance(expression, ufl.differentiation.Curl):
    #     arg = expression.ufl_operands[0]

    #     if arg.ufl_shape == ():
    #         # scalar <- R grad (expr)
    #         f = lambda x, arg=arg: np.dot(np.array([[0, 1.], [-1., 0]]),
    #                                       lambdify(dolfin.grad(arg))(x))

    #     elif arg.ufl_shape == (2, ):
    #         # vector <- R:grad(expr)            
    #         f= lambda x, arg=arg: np.trace(np.dot(np.array([[0, 1.], [-1., 0]]),
    #                                               lambdify(dolfin.grad(arg))(x)))

    #     else:
    #         assert arg.ufl_shape == (3, )
    #         # The usual stuff
    #         f = lambda x, arg=arg: (
    #             lambda G: np.array([G[1, 2]-G[2, 1],
    #                                 G[2, 0]-G[0, 2],
    #                                 G[0, 1]-G[1, 0]]))(lambdify(dolfin.grad(arg))(x))
    #     return f


    # Well that's it for now
    raise ValueError('Unsupported type %s', type(expression))


def extract_index(index):
    '''UFL index to int or list of int'''
    index = map(int, index.indices())
    return index.pop() if len(index) == 1 else tuple(index)


def extract_diff_function(arg, nderivs=1):
    '''
    Get the function to be differentatied along with the number of times 
    this is to be done.
    '''
    assert isinstance(arg, (ufl.differentiation.Grad, dolfin.Function))
    if isinstance(arg, dolfin.Function):
        return arg, nderivs

    return extract_diff_function(arg.ufl_operands[0], nderivs+1)
    

# ----------------------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    parameters['form_compiler']['no-evaluate_basis_derivatives'] = False

    u = Constant(3)

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'CG', 1)
    v = interpolate(Expression('x[0]+x[1]', degree=1), V)
    
    print lambdify(curl(v))((0.5, 0.5))

    V = VectorFunctionSpace(mesh, 'CG', 2)
    f = Expression(('x[0]*x[0]', '2*x[1]*x[0]'), degree=2)
    v = interpolate(f, V)

    print lambdify(curl(v))((0.5, 0.5))

    #print lambdify(Constant(((1, 2), (3, 4)))[0, 0])((0.5, 0.5))
    #print lambdify(inv(sym(grad(v)))+Constant(((1, 0), (0, 1))))((0.5, 0.5))

    A = Constant(((1, 0), (2, 3)))
    B = Constant(((2, 0), (2, 3)))
    v = Constant((1, 0))

    mesh = UnitCubeMesh(3, 3, 3)
    V = VectorFunctionSpace(mesh, 'CG', 2)
    f = interpolate(Expression(('-x[1]', 'x[0]', 'x[2]'), degree=1), V)

    print lambdify(curl(f))((0.5, 0.5, 0.5))

    # FIXME: things such as CellVolume

    # FIXME: differentiation
    # FIXME: wrapping as Expression, eval_cell?, cpp Expression

    # READ SICC: interpreters

    # ComponentTensor, Div, as_matrix, as_vector

    # HOWTO
    # In [4]: f = Expression('x[0]', element=V.ufl_element())

    # In [5]: import ufl

    # In [6]: ufl.algorithms.estimate_degrees.estimate_total_polynomial_degree(f)
    # Out[6]: 1

    # In [7]: ufl.algorithms.extract_unique_elements(f)
    # Out[7]: (FiniteElement('Lagrange', triangle, 1),)
