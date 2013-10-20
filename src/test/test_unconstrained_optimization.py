from numpy import linspace, array
from numpy.testing import assert_almost_equal, assert_equal
from optimization.graphical_optimization import graphicalOptimization, gradientOf2dFunction, \
    plotOptimizationPath
from optimization.line_search import goldenLineSearch, quadraticLineSearch
from optimization.unconstrained_optimization import UnconstrainedProblemSetup, \
    SteepestDescentOptimization, ConjugateGradientOptimization, NewtonOptimization,\
    ModifiedNewtonOptimization, DfpOptimization, BfgsOptimization
import matplotlib.pyplot as pyplot
import unittest


def aroraExample_8_4(x):
    x1, x2 = x[0], x[1] 
    return x1**2.0 + x2**2.0 -2.0*x1*x2

def aroraExample_9_7(x):
    x1, x2 = x[0], x[1] 
    return 10.0*x1**4.0 -20.0*x1**2.0*x2 + 10.0*x2**2.0 + x1**2.0 - 2.0*x1 + 5.0

def aroraExample_9_9(x):
    x1, x2 = x[0], x[1] 
    return 5.0 * x1 ** 2.0 + 2.0 * x1 * x2 + x2 ** 2.0 + 7.0

def buildRosebrockOptimizationSetup():
    p1 = UnconstrainedProblemSetup(
        f = aroraExample_9_7, 
        x0 = [- 1.0 , 3.0], 
#         x0 = [- 1.0 , 1.5], # classic newton is awesome but it gets lost here, use in report! :D 
        lineSearchMethod = goldenLineSearch,
        absoluteEpsilon = 1e-6,
        maxIterations = 500,
        storePoints=True
    )
    return p1

def plotOptimizationPathFromOutput(output):
    x1 = linspace(-2.0, 2.0, 200)
    x2 = linspace(-1.0, 3.0, 200)
    graphicalOptimization(x1, x2, aroraExample_9_7, [], [],linspace(5.0,30.0,8))
    gradientOf2dFunction(x1, x2, aroraExample_9_7)
    plotOptimizationPath(output.optimizationPath)
    pyplot.show()

class Test(unittest.TestCase):

    def testSteepestDescentOptimization(self):
        p1 = buildRosebrockOptimizationSetup()
        opt = SteepestDescentOptimization(p1)
        output = opt.solve()
        
        assert_almost_equal(output.f, 4.0 , 4)
        assert_almost_equal(output.x, [1.0, 1.0] , 2)
        assert_equal(output.iterations, 501)
#         plotOptimizationPathFromOutput(output)
        
    def testConjugateGradientOptimization(self):
        p1 = buildRosebrockOptimizationSetup()
        opt = ConjugateGradientOptimization(p1)
        output = opt.solve()
        
        assert_almost_equal(output.f, 4.0 , 5)
        assert_almost_equal(output.x, [1.0, 1.0] , 5)
        assert_equal(output.iterations, 16)
#         plotOptimizationPathFromOutput(output)
        
    def testNewtonOptimization(self):
        p1 = buildRosebrockOptimizationSetup()
        opt = NewtonOptimization(p1)
        output = opt.solve()
        
        assert_almost_equal(output.f, 4.0 , 5)
        assert_almost_equal(output.x, [1.0, 1.0] , 5)
        assert_equal(output.iterations, 9)
#         plotOptimizationPathFromOutput(output)
        
    def testModifiedNewtonOptimization(self):
        p1 = buildRosebrockOptimizationSetup()
        opt = ModifiedNewtonOptimization(p1)
        output = opt.solve()

        assert_almost_equal(output.f, 4.0 , 4)
        assert_almost_equal(output.x, [1.0, 1.0] , 5)
        assert_equal(output.iterations, 9)        
#         plotOptimizationPathFromOutput(output)
        
    def testDfpOptimization(self):
        p1 = buildRosebrockOptimizationSetup()
        opt = DfpOptimization(p1)
        output = opt.solve()
        
        assert_almost_equal(output.f, 4.0 , 4)
        assert_almost_equal(output.x, [1.0, 1.0] , 5)
        assert_equal(output.iterations, 11)        
#         plotOptimizationPathFromOutput(output)
        
    def testBfgsOptimization(self):
        p1 = buildRosebrockOptimizationSetup()
        opt = BfgsOptimization(p1)
        output = opt.solve()
        
        assert_almost_equal(output.f, 4.0 , 4)
        assert_almost_equal(output.x, [1.0, 1.0] , 5)
        assert_equal(output.iterations, 11)        
#         plotOptimizationPathFromOutput(output)
        
    def testBfgsOptimizationAroraExample(self):
        p1 = UnconstrainedProblemSetup(
            f = aroraExample_9_9, 
            x0 = [1.0, 2.0], 
            lineSearchMethod = goldenLineSearch, 
            absoluteEpsilon = 1e-4, 
            maxIterations=500, 
            storePoints=True 
        )
        opt = BfgsOptimization(p1)
        output = opt.solve()
        
        assert_almost_equal(output.f, 7.0 , 4)
        assert_almost_equal(output.x, [0.0, 0.0] , 5)
        assert_equal(output.iterations, 2)        
        
    def testDfpOptimizationAroraExample(self):
        p1 = UnconstrainedProblemSetup(
            f = aroraExample_9_9, 
            x0 = [1.0, 2.0], 
            lineSearchMethod = goldenLineSearch, 
            absoluteEpsilon = 1e-4, 
            maxIterations=500, 
            storePoints=True 
        )
        
        opt = DfpOptimization(p1)
        output = opt.solve()
        
        assert_almost_equal(output.f, 7.0 , 4)
        assert_almost_equal(output.x, [0.0, 0.0] , 5)
        assert_equal(output.iterations, 2)
        
    def testFromReadme(self):
        def objectiveFunction(x):
            x1, x2 = x[0], x[1] 
            return 10.0*x1**4.0 -20.0*x1**2.0*x2 + 10.0*x2**2.0 + x1**2.0 - 2.0*x1 + 5.0
        
        p1 = UnconstrainedProblemSetup(
            f = objectiveFunction, 
            x0 = [- 1.0 , 3.0], 
            lineSearchMethod = goldenLineSearch, 
            absoluteEpsilon = 1e-6,
            maxIterations = 500,
            )
        
        newtonOpt = NewtonOptimization(p1)
        output = newtonOpt.solve()
        
        

if __name__ == "__main__":
    unittest.main()