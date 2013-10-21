from numpy import linspace, array

import matplotlib.pyplot as pyplot
from optimization.graphical_optimization import graphicalOptimization, gradientOf2dFunction, \
    plotOptimizationPath
from optimization.line_search import goldenLineSearch, armijoLineSearch, quadraticLineSearch,\
    equalLineSearch
from optimization.unconstrained_optimization import UnconstrainedProblemSetup, \
    SteepestDescentOptimization, ConjugateGradientOptimization, NewtonOptimization,\
    ModifiedNewtonOptimization, DfpOptimization, BfgsOptimization
from calculus import AnalyticalGradientAndHessianFunction
import time


def measureTime(optimizationSolve):
    start = time.clock() 
    output = optimizationSolve()
    elapsed = time.clock() - start
    return output, elapsed

def rosenBrockFunction(x):
    x, y = x[0], x[1]
    return (1.0 - x)**2.0 + 100.0*(y - x**2.0)**2.0

def rosenBrockFunctionGradient(x):
    x, y = x[0], x[1]
    return array([-2.0*(1.0 - x) - 400.0*x*(y-x**2.0), 200.0*(y-x**2.0)])

def rosenBrockFunctionHessian(x):
    x, y = x[0], x[1]
    return array( [
       [-400.0*(y - x**2.0) + 800.0*x**2.0 + 2, -400.0*x],
       [-400.0*x, 200.0]
    ])

analyticalRosenbrock = AnalyticalGradientAndHessianFunction(
    function = rosenBrockFunction,
    gradientFunction = rosenBrockFunctionGradient,
    hessianFunction = rosenBrockFunctionHessian
)

def rosenbrockUnconstrainedProblemSetup():
    p1 = UnconstrainedProblemSetup(
        f = rosenBrockFunction, 
#         x0 = [1., 3], # modified newton excels
        x0 = [-1.25, 1.75],  # classic newton gets lost
        lineSearchMethod = goldenLineSearch, 
        absoluteEpsilon = 1.0e-6, 
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
    print analyticalRosenbrock 
    p1 = rosenbrockUnconstrainedProblemSetup()
    opt = SteepestDescentOptimization(p1)
    output = opt.solve()
    print output.f, output.x, output.iterations
    plotRosenbrockFunctionAndOptimizationPath(output)
    
def steepestOptimizationAnalytical():
    p1 = rosenbrockUnconstrainedProblemSetup()
    p1.f = analyticalRosenbrock
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
    output , t = measureTime(opt.solve)
    print output.f, output.x, output.iterations, t
#     plotRosenbrockFunctionAndOptimizationPath(output)
    
def modifiedNewtonOptimizationAnalytical():
    p1 = rosenbrockUnconstrainedProblemSetup()
    p1.f = analyticalRosenbrock
    opt = ModifiedNewtonOptimization(p1)
    output , t = measureTime(opt.solve)
    print output.f, output.x, output.iterations, t
#     plotRosenbrockFunctionAndOptimizationPath(output)
    
def dfpOptimization():
    p1 = rosenbrockUnconstrainedProblemSetup()
    opt = DfpOptimization(p1)
    output = opt.solve()
    print output.f, output.x, output.iterations
    plotRosenbrockFunctionAndOptimizationPath(output)
    
def bfgsOptimization():
    p1 = rosenbrockUnconstrainedProblemSetup()
    opt = BfgsOptimization(p1)
    output = opt.solve()
    print output.f, output.x, output.iterations
    plotRosenbrockFunctionAndOptimizationPath(output)
    
# steepestOptimization()
# steepestOptimizationAnalytical()
# conjugateGradientOptimization()
# newtonOptimization()
modifiedNewtonOptimization()
modifiedNewtonOptimizationAnalytical()
# dfpOptimization()
# bfgsOptimization()
