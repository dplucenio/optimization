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
    # Old way:
#     grad = zeros(len(point))
#     x0 = point
#     for i in xrange(len(point)):
#         x = array(x0)
#         x[i] += epsilon
#         grad[i] = (f(x) - f(x0)) / epsilon
#     return grad
    grad = zeros(len(point))
    for i in xrange(len(point)):
        x1 = array(point)
        x2 = array(point)
        x1[i] -= epsilon
        x2[i] += epsilon
        f1 = f(x1)
        f2 = f(x2)
        df = (f2 - f1) / (2.0 * epsilon)
        grad[i] = df
    return grad

def hessian(f, point, epsilon=1e-6):
    hessian = zeros((len(point),len(point)))
    for i in xrange(len(point)):
        def ff(point):
            return gradient(f, point,epsilon)[i]
        hessian[i,:] = gradient(ff, point, epsilon)
    return hessian

class AnalyticalGradientAndHessianFunction(object):
    def __init__(self, function, gradientFunction, hessianFunction):
        self.function = function
        self.gradientFunction = gradientFunction
        self.hessianFunction = hessianFunction
    
    def __call__(self, x):
        return self.function(x)
    
    def gradient(self, x):
        return self.gradientFunction(x)
    
    def hessian(self, x):
        return self.hessianFunction(x)