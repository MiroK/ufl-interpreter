from utils import *

# A rule is a tuple of two list where the first is the pattern against
# which we will match the expression, hile the second one is the skeleton
# i.e. a rule to make the simplified expression

#def pattern(rule): return rule[0]

#def skeleton(rule): return rule[1]


def match(pattern, expression, dictionary=None):
    '''
    A pattern is e.g. [ufl.algebra.Sum, ['?, 'v'], ['?', 'v']]. On 
    match we return {'v': expression}
    '''



    # The recursion
    
    return match(cdr(pattern), cdr(expression), match(car(pattern),
                                                      car(expression),
                                                      dictionary))
    
    
    if dictionary is None: dictionary = []

    if 

    
    if aribitrary_constant(pattern):
        if is_number(expression):
            return extend(pattern, expression, dictionary)

    if arbitrary_variable(expression):
        if is_variable(expression):
            return extend(pattern, expression, dictionary)

    if arbitrary_expression(expression):
        return extend(pattern, expression, dictionary)



    
