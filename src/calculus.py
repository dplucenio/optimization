from numpy import zeros, array

def gradient(f, point, epsilon=1e-9):
    '''
    Evaluates the gradient of a multidimensional function f(x1, x2, x3, ..., xn) at a given point
    (x1, x2, x3, ..., xn) using finite differences technique.
    
    @param f: Callable 
        Function of n independent variables
    @param point: numpy.array()
        A vector of dimension n
    @param epsilon: float
        The increment used in finite difference technique
    '''
    gradient = zeros(len(point))
    x0 = point
    for i in xrange(len(point)):
        x = array(x0)
        x[i] += epsilon
        gradient[i] = (f(x) - f(x0)) / epsilon
    return gradient