from matplotlib import pyplot as pyplot
from numpy import zeros, meshgrid, linspace
from calculus import gradient

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
    
    for i in xrange(len(X1)):
        for j in xrange(len(X2)):
            x1 = X1[i,j]
            x2 = X2[i,j]
            f_image[i,j] = f(x1, x2)
    
    if functionType == INEQUALITY_CONSTRAINT:
        cs = pyplot.contourf(X1, X2,f_image, [0,max(max(x1_domain), max(x2_domain))], colors=((0.85,0.8,0.95,),))
    elif functionType == EQUALITY_CONSTRAINT:
        cs = pyplot.contour(X1, X2,f_image, [0], linewidths=2.0)
    elif functionType == OBJECTIVE_FUNCTION:
        if levels is not None:
            cs = pyplot.contour(X1, X2,f_image, levels)
        else:
            cs = pyplot.contour(X1, X2,f_image)
        pyplot.clabel(cs, colors='k')
    pyplot.xlabel('x1')
    pyplot.ylabel('x2')
    return cs


def graphicalOptimization(
    x1_domain, 
    x2_domain, 
    objectiveFunction, 
    inequalityFunctions, 
    equalityFunctions,
    levels=None
    ): 
    for g in inequalityFunctions:
        contourTwoVariablefunction(x1_domain, x2_domain, g, INEQUALITY_CONSTRAINT)
    for h in equalityFunctions:
        contourTwoVariablefunction(x1_domain, x2_domain, h, EQUALITY_CONSTRAINT)
    contourTwoVariablefunction(x1_domain, x2_domain, objectiveFunction, OBJECTIVE_FUNCTION, levels)
    
    

def gradientOf2dFunction(x1_domain, x2_domain, f, eval_points=15):
    def ff(x):
        return f(x[0], x[1])
    
    X1 = linspace(x1_domain[0],x1_domain[-1], eval_points)
    X2 = linspace(x2_domain[0],x2_domain[-1], eval_points)
    X1, X2 = meshgrid(X1, X2)
    U = zeros((   len(X1), len(X2) ))
    V = zeros((   len(X1), len(X2) ))
    for i in xrange(len(X1)):
        for j in xrange(len(X2)):
            x = X1[i,j]
            y = X2[i,j]
            U[i,j] = gradient(ff, [x,y] )[0]
            V[i,j] = gradient(ff, [x,y] )[1]
    return pyplot.quiver(X1, X2, U,V, color=((0.4,0.75,0.6)))
    
