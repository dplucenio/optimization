import unittest

from numpy import linspace

import matplotlib.pyplot as pyplot
from optimization.graphical_optimization import graphicalOptimization, gradientOf2dFunction, \
    plotOptimizationPath
from optimization.unconstrained_optimization import unconstrainedProblemSetup, \
    steepestDescentOptimization, conjugateGradientOptimization


def aroraExample_8_4(x):
    x1, x2 = x[0], x[1] 
    return x1**2.0 + x2**2.0 -2.0*x1*x2

def rosenbrock(x):
    x1, x2 = x[0], x[1] 
    return 10.0*x1**4.0 -20.0*x1**2.0*x2 + 10.0*x2**2.0 + x1**2.0 - 2.0*x1 + 5

def buildRosebrockOptimizationSetup():
    p1 = unconstrainedProblemSetup(
        f = rosenbrock, 
        x0 = [-1.0 , 3.0], 
        absoluteEpsilon = 1e-4,
        maxIterations = 500,
        storePoints=True
    )
    return p1


def plotOptimizationPathFromOutput(output):
    x1 = linspace(-2.0, 2.0, 200)
    x2 = linspace(-1.0, 3.0, 200)
    graphicalOptimization(x1, x2, rosenbrock, [], [],linspace(5.0,30.0,8))
    gradientOf2dFunction(x1, x2, rosenbrock)
    plotOptimizationPath(output.optimizationPath)
    pyplot.show()

class Test(unittest.TestCase):

    def testSteepestDescentOptimization(self):
        p1 = buildRosebrockOptimizationSetup()
        opt = steepestDescentOptimization(p1)
        output = opt.solve()
        
        plotOptimizationPathFromOutput(output)
        
    def testConjugateGradientOptimization(self):
        p1 = buildRosebrockOptimizationSetup()
        opt = conjugateGradientOptimization(p1)
        output = opt.solve()
        
        plotOptimizationPathFromOutput(output)

if __name__ == "__main__":
    unittest.main()