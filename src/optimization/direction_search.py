from numpy.linalg import norm

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
    
def newton(x, gradientFunction, hessianFunction): 
    pass
