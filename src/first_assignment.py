from matplotlib import pyplot as pyplot
from numpy import linspace, zeros

# todo: Subtitute these for a nice enum python implementation
OBJECTIVE_FUNCTION=0
EQUALITY_CONSTRAINT=1
INEQUALITY_CONSTRAINT=2

# All objective function, equality and inequality constraints are defined here following the
# 'Standard Design Optimization Model described in 'Introduction to optimum design - Arora'
# f(x[]) = f(x1, x2, x3, ...)        : Objective functions
# h(x[]) = h(x1, x2, x3, ...) = 0    : Equality constraints
# g(x[]) = g(x1, x2, x3, ...) <= 0   : Inequality constraints


def contour2Dfunction(x1_domain, x2_domain, f, functionType, levels=None):
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
    for i,x1 in enumerate(x1_domain):
        for j,x2 in enumerate(x2_domain):
            f_image[i,j] = f(x1, x2)
    if functionType == OBJECTIVE_FUNCTION:
        if levels is not None:
            cs = pyplot.contour(x1_domain, x2_domain,f_image, levels)
        else:
            cs = pyplot.contour(x1_domain, x2_domain,f_image)
        pyplot.clabel(cs)
    elif functionType == EQUALITY_CONSTRAINT:
        cs = pyplot.contour(x1_domain, x2_domain,f_image, [0])
    elif functionType == INEQUALITY_CONSTRAINT:
        cs = pyplot.contourf(x1_domain, x2_domain,f_image, [0,max(max(x1_domain), max(x2_domain))])
    return cs
    

def exercise_3_1():
    x_vector = linspace(0.0, 4.0, 300)
    y_vector = linspace(0.0, 4.0, 300)
    
    def f(x1, x2):
        return (x1-3.)**2. +(x2-3.)**2.
    def g(x1, x2):
        return x1 + x2 -4 
    
    contour2Dfunction(x_vector, y_vector,f, OBJECTIVE_FUNCTION,levels=linspace(0.0, 15.0,16))
    contour2Dfunction(x_vector, y_vector,g, INEQUALITY_CONSTRAINT)
    pyplot.show()
    
def exercise_3_8():
    x1_domain = linspace(0.0, 6.0, 100)
    x2_domain = linspace(0.0, 6.0, 100)
    
    def f(x1, x2):
        return x1**2. - 2.*x2**2. - 4.*x1
    def g1(x1, x2):
        return x1 + x2 - 6
    def g2(x1, x2):
        return x2 - 3
            
    contour2Dfunction(x1_domain, x2_domain,f, OBJECTIVE_FUNCTION, levels=linspace(0,-22,6))
    contour2Dfunction(x1_domain, x2_domain,g1, INEQUALITY_CONSTRAINT)
    contour2Dfunction(x1_domain, x2_domain,g2, INEQUALITY_CONSTRAINT)
    pyplot.show()
    
    
def exercise_3_13():
    x1_domain = linspace(-6.0, 6.0, 100)
    x2_domain = linspace(-6.0, 6.0, 100)
    
    def f(x1, x2):
        return 9.*x1**2. + 13.*x2**2. + 18.*x1*x2 - 4.
    def g1(x1, x2):
        return x1**2. + x2**2. + 2.*x1 - 16
            
    contour2Dfunction(x1_domain, x2_domain,f, levels=linspace(210, 600, 6))
    contour2Dfunction(x1_domain, x2_domain,g1, EQUALITY_CONSTRAINT)
    pyplot.show()
    
exercise_3_8()