# UFL expression has a tree representation consisting of operators 
# and terminals. Terminals are leaves in this tree. I want to distinguish
# number because they can be easily evaluated.

from dolfin import Constant
from ufl.constantvalue import ConstantValue
from ufl.core.terminal import Terminal


def is_terminal(expr):
    '''Leaf'''
    return isinstance(expr, Terminal)


def is_compound(expr):
    '''Node'''
    return not is_terminal(expr)


def is_number(expr):
    '''A number in the system'''
    return isinstance(expr, (Constant, ConstantValue))


def is_variable(expr):
    '''A non number terminal'''
    return not is_number(expr) and is_terminal(expr)


def as_list(expr):
    '''Transform UFL expression to list(scheme) notation'''
    if is_compound(expr):
        return [type(expr)] + map(as_list, expr.ufl_operands)
    # Atoms
    return expr


def car(iterable):
    '''First'''
    return iterable[0]


def cdr(iterable):
    '''All but first'''
    return iterable[1:]


def collapse(expr):
    '''Turn list notation to UFL'''
    if is_terminal(expr): return expr

    return apply(car(expr), map(collapse, cdr(expr)))

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import FunctionSpace, Function, UnitSquareMesh

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'CG', 1)

    u = Function(V)
    v = Function(V)

    L = (u+v)-2*u*v
    print L

    L_list = as_list(L)
    print L_list

    print collapse(L_list)
