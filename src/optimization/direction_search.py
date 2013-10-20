from numpy.linalg import norm, solve
from numpy import identity

def steepestDescentDirection(gradient):
    return -gradient

def conjugateGradientDirection(gradient, latestgradient=None):
    if latestgradient is None:
        return steepestDescentDirection(gradient)
    else:
        c_0 = latestgradient
        c_1 = gradient
        betha = (norm(c_1) / norm(c_0))**2.0
        return -c_1 - betha*(c_0)
    
def newtonDirection(gradient, hessian): 
    direction = solve(hessian, -gradient)
    return direction

def modifiedNewtonDirection(rho, gradient, hessian):
    hessian = hessian + rho * identity(len(gradient)) 
    direction = solve(hessian, -gradient)
    return direction
