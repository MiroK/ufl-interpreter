import ufl, dolfin


def is_terminal(expr):
    '''An expression is a terminal if it has no operands'''
    return not bool(expr.ufl_operands)


def is_number(expr):
    '''UFL representation of number'''
    return isinstance(expr, (ufl.constantvalue.ConstantValue, dolfin.Constant))


def incr(expr):
    '''
    Increment a number. Numbers are defined by their values so this is 
    not an inplace operation
    '''
    assert is_number(expr)
    return type(expr)(expr(0) + 1) # Eval

        
def simplify_sum_once(expr):
    '''Simplify the sum '''
    print '\t', expr, type(expr)
    # The primitives cannot be simplified 
    if is_terminal(expr):
        return expr
    
    op = type(expr)

    if op == ufl.algebra.Sum:
        first, second = expr.ufl_operands        

        # Rule u + u
        if first == second:
            return dolfin.Constant(2)*first
        
        # Rule u + c*u, with c the constant
        for this, that in ((first, second), (second, first)):
            # u
            print this, is_terminal(this)
            if is_terminal(this):
                # (* c u)
                if isinstance(that, ufl.algebra.Product):
                    # c u
                    if is_number(that.ufl_operands[0]) and that.ufl_operands[1] == this:
                        return dolfin.Constant(incr(that.ufl_operands[0]))*this
                    # u c
                    if is_number(that.ufl_operands[1]) and that.ufl_operands[0] == this:
                        return dolfin.Constant(incr(that.ufl_operands[1]))*this

                
        # No rule worked

    return op(*map(simplify_sum_once, expr.ufl_operands))

# -------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import (UnitSquareMesh, FunctionSpace, Function, sin,
                        Constant, inner)

    
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'CG', 1)
    u = Function(V)
    v = Function(V)

    expr = inner(u, v) + inner(u, v)
    #expr = v + 2*v 

    print simplify_sum_once(expr)
    
    #print simplify_sum_once(simplify_sum_once(expr))

    
