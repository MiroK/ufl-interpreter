from dolfin import Expression


def build_cexpr(element, shape, body):
    '''
    An Expression instance with element and which uses body to compute 
    it's eval method.
    '''

    def f(values, x, body=body): values[:] = body(x).flatten()
    
    return type('CExpr',
                (Expression, ),
                {'value_shape': lambda self, shape=shape: shape,
                 'eval': lambda self, values, x: f(values, x),
                 'is_CExpr': True})(element=element)
