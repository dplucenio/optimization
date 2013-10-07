from matplotlib import pyplot as pyplot
from numpy import linspace

from graphical_optimization import contourTwoVariablefunction, EQUALITY_CONSTRAINT, \
    INEQUALITY_CONSTRAINT, OBJECTIVE_FUNCTION, gradientOf2dFunction

def exercise_3_1():
    x_vector = linspace(0.0, 4.0, 300)
    y_vector = linspace(0.0, 4.0, 300)
    
    def f(x1, x2):
        return (x1-3.)**2. +(x2-3.)**2.
    def g(x1, x2):
        return x1 + x2 -4 
    
    contourTwoVariablefunction(x_vector, y_vector,f, OBJECTIVE_FUNCTION,levels=linspace(0.0, 15.0,16))
    contourTwoVariablefunction(x_vector, y_vector,g, INEQUALITY_CONSTRAINT)
    gradientOf2dFunction(x_vector, y_vector, f)
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
    def g3(x1, x2):
        return -x1
    def g4(x1, x2):
        return -x2
            
    contourTwoVariablefunction(x1_domain, x2_domain,g1, INEQUALITY_CONSTRAINT)
    contourTwoVariablefunction(x1_domain, x2_domain,g2, INEQUALITY_CONSTRAINT)
    contourTwoVariablefunction(x1_domain, x2_domain,g3, INEQUALITY_CONSTRAINT)
    contourTwoVariablefunction(x1_domain, x2_domain,g4, INEQUALITY_CONSTRAINT)
    contourTwoVariablefunction(x1_domain, x2_domain,f, OBJECTIVE_FUNCTION, levels=linspace(0,-22,6))
    gradientOf2dFunction(x1_domain, x2_domain, f,21)
    
    pyplot.show()
    
    
def exercise_3_13():
    x1_domain = linspace(-6.0, 6.0, 100)
    x2_domain = linspace(-6.0, 6.0, 100)
    
    def f(x1, x2):
        return 9.*x1**2. + 13.*x2**2. + 18.*x1*x2 - 4.
    def g1(x1, x2):
        return x1**2. + x2**2. + 2.*x1 - 16
            
    contourTwoVariablefunction(x1_domain, x2_domain,f, OBJECTIVE_FUNCTION, levels=linspace(210, 600, 6))
    contourTwoVariablefunction(x1_domain, x2_domain,g1, EQUALITY_CONSTRAINT)
    gradientOf2dFunction(x1_domain, x2_domain, f)
    pyplot.show()
    
if __name__ == "__main__":
    exercise_3_8()