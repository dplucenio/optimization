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


# 1 - Line search study---------------------------------------------------------------------------
lineSearchFunctionCalls = 0
def aroraExampleFunction(x):
    global lineSearchFunctionCalls
    lineSearchFunctionCalls += 1
    return 3.0*x[0]**2.0 + 2.0*x[0]*x[1] + 2.0*x[1]**2.0 + 7.0

def solveWithLineSearchAndOutputStatus(lineSearch):
    global lineSearchFunctionCalls
    d = array([-1.0, -1.0])
    x_k = array([1.0, 2.0])
    start = time.clock()
    alpha_k = lineSearch(aroraExampleFunction, x_k, d)
    elapsed = time.clock() - start
    print alpha_k, lineSearchFunctionCalls, elapsed
    lineSearchFunctionCalls = 0
    
def reportLineSearchPlot():
    d = array([-1.0, -1.0])
    x_k = array([1.0, 2.0])
    x = []
    y = []
    for i in linspace(0.0, 2.0, 200):
        print i
        x.append(i)
        y.append(aroraExampleFunction(x_k + i*d))
        
    pyplot.plot(x,y)
    pyplot.xlabel('alpha')
    pyplot.ylabel('f(x_k + alpha*d)')
    pyplot.show()

def lineSearchStudy():
    # reportLineSearchPlot()
    solveWithLineSearchAndOutputStatus(equalLineSearch)
    solveWithLineSearchAndOutputStatus(goldenLineSearch)
    solveWithLineSearchAndOutputStatus(quadraticLineSearch)
    solveWithLineSearchAndOutputStatus(armijoLineSearch)

# ------------------------------------------------------------------------------------------------

# Unconstrained optimization ---------------------------------------------------------------------
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

p1 = UnconstrainedProblemSetup(
#     f = rosenBrockFunction,
    f = analyticalRosenbrock, 
    x0 = [-1., 3.], # modified newton excels
#     x0 = [-1., -1.], # modified newton excels
#     x0 = [1., 3.], # normal
#     x0 = [-1.25, 1.75],  # classic newton gets lost
    lineSearchMethod = armijoLineSearch, 
    absoluteEpsilon = 1.0e-6, 
    maxIterations = 500, 
    storePoints = True
)

def plotRosenbrockFunctionAndOptimizationPath(output):
    x1 = linspace(-2.0, 2.0, 200)
    x2 = linspace(-1.0, 3.0, 200)
    graphicalOptimization(x1, x2, rosenBrockFunction, [], [],linspace(5.0, 200.0,8))
    gradientOf2dFunction(x1, x2, rosenBrockFunction)
    plotOptimizationPath(output.optimizationPath)
    pyplot.show()
    
def solveUnconstrainedOptimizationAndOutputStatus(optimizationSolver):
    opt = optimizationSolver(p1)
    start = time.clock() 
    output = opt.solve()
    elapsed = time.clock() - start
    outputStatus(output, elapsed)
    plotRosenbrockFunctionAndOptimizationPath(output)

def outputStatus(output, elapsed):
    stopCriterium = None
    if output.iterations >= 500:
        stopCriterium = "Maximum iteration reached"
    else:
        stopCriterium = "Absolute epsilon reached"
        
#     print '%.3f\t%.3f\t%.3f\t%.3f\t%f\t%s'%(output.f, output.x[0], output.x[1], output.residual, output.iterations, stopCriterium)
    print '%.2e\t%.2f\t%.2f\t%.2e\t%d\t%.4f'%(output.f, output.x[0], output.x[1], output.residual, output.iterations, elapsed)
    
solveUnconstrainedOptimizationAndOutputStatus(SteepestDescentOptimization)
solveUnconstrainedOptimizationAndOutputStatus(ConjugateGradientOptimization)
solveUnconstrainedOptimizationAndOutputStatus(NewtonOptimization)
solveUnconstrainedOptimizationAndOutputStatus(ModifiedNewtonOptimization)
solveUnconstrainedOptimizationAndOutputStatus(DfpOptimization)
solveUnconstrainedOptimizationAndOutputStatus(BfgsOptimization)

# ------------------------------------------------------------------------------------------------

