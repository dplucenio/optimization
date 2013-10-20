from numpy.linalg import norm, solve
from numpy import identity

def steepestDescentDirection(x, gradient):
    return -gradient

def conjugateGradientDirection(x, gradient, latestgradient=None):
    if latestgradient is None:
        return steepestDescentDirection(x, gradient)
    else:
        c_0 = latestgradient
        c_1 = gradient
        betha = (norm(c_1) / norm(c_0))**2.0
        return -c_1 - betha*(c_0)
    
def newtonDirection(x, gradient, hessian): 
    direction = solve(hessian, -gradient)
    return direction

def modifiedNewtonDirection(x, rho, gradient, hessian):
    hessian = hessian + identity(len(gradient)) 
    direction = solve(hessian, -gradient)
    return direction
