from numpy import linspace

import matplotlib.pyplot as pyplot
from optimization.graphical_optimization import graphicalOptimization, gradientOf2dFunction, \
    plotOptimizationPath
from optimization.line_search import goldenLineSearch, armijoLineSearch, quadraticLineSearch
from optimization.unconstrained_optimization import UnconstrainedProblemSetup, \
    SteepestDescentOptimization, ConjugateGradientOptimization, NewtonOptimization,\
    ModifiedNewtonOptimization


def rosenBrockFunction(x):
    x, y = x[0], x[1]
    return (1.0 - x)**2.0 + 100.0*(y - x**2.0)**2.0

def rosenbrockUnconstrainedProblemSetup():
    p1 = UnconstrainedProblemSetup(
        f = rosenBrockFunction, 
        x0 = [-1.25, 1.75], 
        lineSearchMethod = goldenLineSearch, 
        absoluteEpsilon = 1.0e-4, 
        maxIterations = 500, 
        storePoints = True
    )
    return p1
    

def plotRosenbrockFunctionAndOptimizationPath(output):
    x1 = linspace(-2.0, 2.0, 200)
    x2 = linspace(-1.0, 3.0, 200)
    graphicalOptimization(x1, x2, rosenBrockFunction, [], [],linspace(5.0, 200.0,8))
    gradientOf2dFunction(x1, x2, rosenBrockFunction)
    plotOptimizationPath(output.optimizationPath)
    pyplot.show()

def steepestOptimization():
    p1 = rosenbrockUnconstrainedProblemSetup()
    opt = SteepestDescentOptimization(p1)
    output = opt.solve()
    print output.f, output.x, output.iterations
    plotRosenbrockFunctionAndOptimizationPath(output)
    
def conjugateGradientOptimization():
    p1 = rosenbrockUnconstrainedProblemSetup()
    opt = ConjugateGradientOptimization(p1)
    output = opt.solve()
    print output.f, output.x, output.iterations
    plotRosenbrockFunctionAndOptimizationPath(output)
    
def newtonOptimization():
    p1 = rosenbrockUnconstrainedProblemSetup()
    opt = NewtonOptimization(p1)
    output = opt.solve()
    print output.f, output.x, output.iterations
    plotRosenbrockFunctionAndOptimizationPath(output)
    
def modifiedNewtonOptimization():
    p1 = rosenbrockUnconstrainedProblemSetup()
    opt = ModifiedNewtonOptimization(p1)
    output = opt.solve()
    print output.f, output.x, output.iterations
    plotRosenbrockFunctionAndOptimizationPath(output)
    

steepestOptimization()
conjugateGradientOptimization()
newtonOptimization()
modifiedNewtonOptimization()
