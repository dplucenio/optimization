from calculus import gradient
from numpy import zeros, empty, fabs


def firstOrderNecessaryCondition(f, point, epsilon=1e-4):
    '''
    @param f: Callable 
        Function of n independent variables
    @param point: numpy.array()
        A vector of dimension n
    @param epsilon: float
    @return: bool
        Returns true or false whether first order necessary condition is attended
    '''
    return (fabs(gradient(f, point)) < epsilon).all()

