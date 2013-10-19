from matplotlib import pyplot as pyplot
from numpy import linspace, array, zeros, dot
from scipy.optimize import fsolve

from calculus import gradient
from optimization.graphical_optimization import gradientOf2dFunction, graphicalOptimization, \
    plotOptimalityConditionAt


def exercise_3_1():
    x1_domain = linspace(-0.2, 4.2, 300)
    x2_domain = linspace(-0.2, 4.2, 300)
    
    def f(x):
        x1, x2 = x[0], x[1]
        return (x1-3.)**2. +(x2-3.)**2.

    def g1(x):
        x1, x2 = x[0], x[1]
        return x1 + x2 -4
     
    def g2(x):
        x1=x[0]
        return -x1
     
    def g3(x):
        x2=x[1]
        return -x2 
    
    graphicalOptimization(x1_domain, x2_domain, f, [g1, g2, g3], [], levels=linspace(0,16,17))
    gradientOf2dFunction(x1_domain, x2_domain, f, 21)
    pyplot.plot([2.0], [2.0], 'k.', markersize=15.0)
    pyplot.show()
    
def exercise_3_8():
    x1_domain = linspace(-0.2, 8.2, 100)
    x2_domain = linspace(-0.2, 3.4, 100)
    
    def f(x):
        x1, x2 = x[0], x[1]
        return x1**2. - 2.*x2**2. - 4.*x1
    def g1(x):
        x1, x2 = x[0], x[1]
        return x1 + x2 - 6
    def g2(x):
        x1, x2 = x[0], x[1]
        return x2 - 3
    def g3(x):
        x1, x2 = x[0], x[1]
        return -x1  
    def g4(x):
        x1, x2 = x[0], x[1]
        return -x2
            
    graphicalOptimization(x1_domain, x2_domain, f, [g1, g2, g3, g4], [], levels=linspace(6,-22,8))
    gradientOf2dFunction(x1_domain, x2_domain, f, 21)
    pyplot.plot([2.0], [3.0], 'k.', markersize=15.0)
    pyplot.show()
    
    
def exercise_3_13():
    x1_domain = linspace(-6.0, 6.0, 100)
    x2_domain = linspace(-6.0, 6.0, 100)
     
    def f(x):
        x1, x2 = x[0], x[1]
        return 9.*x1**2. + 13.*x2**2. + 18.*x1*x2 - 4.
    def h1(x):
        x1, x2 = x[0], x[1]
        return x1**2. + x2**2. + 2.*x1 - 16
             
    graphicalOptimization(x1_domain, x2_domain, f, [], [h1], levels=[5,15,50,100,200,300,450,600])
    gradientOf2dFunction(x1_domain, x2_domain, f)
    pyplot.show()

     
def exercise_3_21():
    x1_domain = linspace(-6.0, 50.0, 300)
    x2_domain = linspace(-6.0, 60.0, 300)
    
    M = 8000.0
    sigma_a = 0.8
    V = 150.0
    tau_a = 0.3
    
    
    def f(x):
        b, d = x[0], x[1]
        return b * d
    
    def g1(x):
        b, d = x[0], x[1]
        return 6.0 * M - (sigma_a * b * d**2.0)
    
    def g2(x):
        b, d = x[0], x[1]
        return 3.0 * V - (tau_a * 2.0 * b * d)
    
    def g3(x):
        b, d = x[0], x[1]
        return d - 2.0 * b
    
    def g4(x):
        b, d = x[0], x[1]
        return -b
    
    def g5(x):
        b, d = x[0], x[1]
        return -d
            
    graphicalOptimization(
        x1_domain,
        x2_domain,
        f,
        [g1,g2,g3,g4,g5],
        [g1,g2,g3,g4,g5],
        levels=[200.0, 400.0, 600.0, 800, 1000.0, 1216.0, 1500.0, 1800.0, 2100.0, 2400.0]
    )
    gradientOf2dFunction(x1_domain, x2_domain, f)
    pyplot.plot([24.66], [24.66*2], 'k.', markersize=15.0)
    pyplot.show()
    
if __name__ == "__main__":
    exercise_3_21()