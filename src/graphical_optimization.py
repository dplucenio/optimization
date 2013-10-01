from matplotlib import pyplot as pyplot
from numpy import zeros, meshgrid

# todo: Replace these for a nice enum python implementation
OBJECTIVE_FUNCTION=0
EQUALITY_CONSTRAINT=1
INEQUALITY_CONSTRAINT=2

def contourTwoVariablefunction(x1_domain, x2_domain, f, functionType, levels=None):
    '''
    Draw contour lines of objective functions values and shades equality and inequality constraints
    revealing the feasible region. This allows for a graphic analysis and visualization of the
    optimization
    
    @param x1_domain: list(float) 
    @param x2_domain: list(float)
    @param f: function
        The objective function, equality or inequality constraint all function of two variables x1
        and x2
    @param functionType:
        OBJECTIVE_FUNCTION, EQUALITY_CONSTRAINT or INEQUALITY_CONSTRAINT
    @param levels:
        If desired the specific values for the objective function to be ploted
    '''
    
    f_image = zeros((len(x1_domain),len(x2_domain)))
    X1, X2 = meshgrid(x1_domain, x2_domain)
    for i,x1 in enumerate(x1_domain):
        for j,x2 in enumerate(x2_domain):
            f_image[i,j] = f(x1, x2)
    if functionType == OBJECTIVE_FUNCTION:
        if levels is not None:
            cs = pyplot.contour(X1, X2,f_image, levels)
        else:
            cs = pyplot.contour(X1, X2,f_image)
#         pyplot.quiver(X1,X2,(10,10),(10,10))
        pyplot.clabel(cs)
    elif functionType == EQUALITY_CONSTRAINT:
        cs = pyplot.contour(X1, X2,f_image, [0])
    elif functionType == INEQUALITY_CONSTRAINT:
        cs = pyplot.contourf(X1, X2,f_image, [0,max(max(x1_domain), max(x2_domain))])
    return cs


def quiverGradientOf2dFunction(x1_domain, x2_domain, f, functionType, levels=None):
    X1, X2 = meshgrid(x1_domain, x2_domain)
    epsilon = 1e-9
    
    f_image = zeros((len(x1_domain),len(x2_domain)))
    f_image = zeros((len(x1_domain),len(x2_domain)))