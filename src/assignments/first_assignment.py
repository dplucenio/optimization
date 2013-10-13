from matplotlib import pyplot as pyplot
from numpy import linspace

from optimization.graphical_optimization import gradientOf2dFunction, graphicalOptimization


def exercise_3_1():
    x1_domain = linspace(-0.2, 4.0, 300)
    x2_domain = linspace(-0.2, 4.0, 300)
    
    def f(x1, x2):
        return (x1-3.)**2. +(x2-3.)**2.

    def g1(x1, x2):
        return x1 + x2 -4
     
    def g2(x1, x2):
        return -x1
     
    def g3(x1, x2):
        return -x2 
    
    graphicalOptimization(x1_domain, x2_domain, f, [g1, g2, g3], [], levels=linspace(0,16,17))
    gradientOf2dFunction(x1_domain, x2_domain, f)
    pyplot.plot([2.0], [2.0], 'k.', markersize=15.0)
    pyplot.show()
    
def exercise_3_8():
    x1_domain = linspace(-0.2, 8.2, 100)
    x2_domain = linspace(-0.2, 3.4, 100)
    
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
            
    graphicalOptimization(x1_domain, x2_domain, f, [g1, g2, g3, g4], [], levels=linspace(6,-22,8))
    gradientOf2dFunction(x1_domain, x2_domain, f,21)
    pyplot.plot([2.0], [3.0], 'k.', markersize=15.0)
    pyplot.show()
    
    
def exercise_3_13():
    x1_domain = linspace(-6.0, 6.0, 100)
    x2_domain = linspace(-6.0, 6.0, 100)
    
    def f(x1, x2):
        return 9.*x1**2. + 13.*x2**2. + 18.*x1*x2 - 4.
    def h1(x1, x2):
        return x1**2. + x2**2. + 2.*x1 - 16
            
    graphicalOptimization(x1_domain, x2_domain, f, [], [h1], levels=[5,15,50,100,200,300,450,600])
    gradientOf2dFunction(x1_domain, x2_domain, f)
    pyplot.show()
    
    
    
if __name__ == "__main__":
    exercise_3_13()