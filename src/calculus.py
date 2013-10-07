from numpy import zeros, array

def gradient(f, point, epsilon=1e-6):
    '''
    Evaluates the gradient of a multidimensional function f(x1, x2, x3, ..., xn) at a given point
    (x1, x2, x3, ..., xn) using finite differences technique.
                                         _  _    
                                        | df |   
                                        | -- |   
                _  _                    | x1 |   
               | x1 |                   |    |   
        ->     |    |          /->\     | df |   
        x   =  | x2 |     grad |x |  =  | -- |   
               |    |          \  /     | x2 |   
               |_x3_|                   |    |   
                                        | df |   
                                        | -- |   
                                        |_x3_|   
    
    @param f: Callable 
        Function of n independent variables
    @param point: numpy.array()
        A vector of dimension n
    @param epsilon: float
        The increment used in finite difference technique
    '''
    grad = zeros(len(point))
    x0 = point
    for i in xrange(len(point)):
        x = array(x0)
        x[i] += epsilon
        grad[i] = (f(x) - f(x0)) / epsilon
    return grad

def hessian(f, point, epsilon=1e-6):
    hessian = zeros((len(point),len(point)))
    for i in xrange(len(point)):
        def ff(point):
            return gradient(f, point,epsilon)[i]
        hessian[i,:] = gradient(ff, point, epsilon)
    return hessian
