import dolfin as df
import ufl


def is_number(expr):
    '''Number types'''
    is_df_number = isinstance(expr, (df.Constant, ufl.constantvalue.ConstantValue))
    return is_df_number or isinstance(expr, (int, float))

    
def is_compound(expr):
    '''A compound in the UFL graph is a node'''
    return bool(expr.ufl_operands)


def is_variable(expr):
    '''An UFL graph node which is not a number'''
    return not is_number(expr) and not is_compound(expr)


def car(iterable):
    '''Car'''
    return next(iter(iterable))


def rest(iterable):
    '''Cdr'''
    iterable = iter(iterable)
    next(iterable)
    return [item for item in iterable]


# --------------------------------------------------------------------

def simplifier(the_rules):
    '''Construct a procuderu for simplifying expression with given rules'''
    # What we want to make
    def simplify_expression(expr):
        '''Simplify expression. Return new expression'''
        return apply_rules(simplify_operands(expr) if is_compound(expr) else expr)

    def simplify_operands(expr):
        '''For a compound expression simplify its arguments. Return new expression.'''
        operands = expr.ufl_operands
        # Here we make a new expression with possibly simplified arguments
        return type(expr)(*[simplify_expression(first(operands)),
                            simplify_parts(rest(operands))])

    def apply_rules(expr):
        '''Return a procedure for making expression using the rules'''
        def scan(rules):
            '''Produce a new expression by applying rules to it'''
            if not rules: return expr
            # Get ingredients for instantiating the expression using the
            # first rule
            dictionary = match(pattern(first(rules)), expr)
            # If fails try the next rule
            if not dictionary: return scan(rest(rules))

            # A new expression
            return simplify_expression(initialize(skeleton(first(rules)), dictionary))
        # Scanner with the rules
        return scan(the_rules)

    # Tadaa
    return simplify_expression
    
# --------------------------------------------------------------------
        
class Simplifier(object):
    '''A simplifier which use the rules'''
    def __init__(self, rules):
        self.simplifier = simplifier(rules)
        self.rules = rules
                   
    def __call__(self, expr):
        return self.simplifier(expr)
        
    def __add__(self, other_simplifier):
        return Simplifier(self.rules + other_simplifier.rules)
