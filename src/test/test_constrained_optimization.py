import unittest
from optimization.constrained_optimization import ConstrainedProblemSetup,\
    ExternalPenaltyOptimization, AugmentedLagrangianOptimization
from optimization.line_search import goldenLineSearch
from optimization.unconstrained_optimization import DfpOptimization, SteepestDescentOptimization,\
    ModifiedNewtonOptimization
from optimization.graphical_optimization import graphicalOptimization
from numpy import linspace
import matplotlib.pyplot as pyplot
import time


class Test(unittest.TestCase):

    def testExample(self):
        
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
        
        p1 = ConstrainedProblemSetup(
            f = f, 
            g = [g1, g2, g3, g4],
            h = [],
            x0 = [2.0, 0.5], 
            lineSearchMethod = goldenLineSearch,
            absoluteEpsilon = 1e-6,
            maxIterations = 500,
            storePoints=True
        )
        
        opt = ExternalPenaltyOptimization(p1, ModifiedNewtonOptimization,rho=100000.)
        start = time.clock() 
        output = opt.solve()
        elapsed = time.clock() - start
        print '%.2e\t%.2f\t%.2f\t%.4f'%(output.f, output.x[0], output.x[1],elapsed)
        
        graphicalOptimization(
            linspace(-0.2, 6.2, 100),
            x2_domain = linspace(-0.2, 3.4, 100),
            objectiveFunction = f, 
            inequalityFunctions = [g1, g2, g3, g4], 
            equalityFunctions = [], 
            levels=linspace(6,-22,8), 
            path = output.optimizationPath)
        pyplot.show()
        
    def testExampleWithAugmentedLagrangian(self):
        
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
        
        p1 = ConstrainedProblemSetup(
            f = f, 
            g = [g1, g2, g3, g4],
            h = [],
            x0 = [2.0, 0.5], 
            lineSearchMethod = goldenLineSearch,
            absoluteEpsilon = 1e-6,
            maxIterations = 500,
            storePoints=True
        )
        
        opt = AugmentedLagrangianOptimization(p1, ModifiedNewtonOptimization, rho=1000.0)
        start = time.clock() 
        output = opt.solve()
        elapsed = time.clock() - start
        print '%.2e\t%.2f\t%.2f\t%.4f'%(output.f, output.x[0], output.x[1],elapsed)
        
        graphicalOptimization(
            linspace(-0.2, 6.2, 100),
            x2_domain = linspace(-0.2, 3.4, 100),
            objectiveFunction = f, 
            inequalityFunctions = [g1, g2, g3, g4], 
            equalityFunctions = [], 
            levels=linspace(6,-22,8), 
            path = output.optimizationPath)
        pyplot.show()
        
    def testExample2DWithAugmentedLagrangian(self):
        
        def f(x):
            x1, x2 = x[0], x[1]
            return x1**2. + x2**2. - 3.*x1*x2
        def g(x):
            x1, x2 = x[0], x[1]
            return x1**2. + x2**2. - 6.
        
        p1 = ConstrainedProblemSetup(
            f = f, 
            g = [g],
            h = [],
            x0 = [0.1, 0.1], 
            lineSearchMethod = goldenLineSearch,
            absoluteEpsilon = 1e-6,
            maxIterations = 500,
            storePoints=True
        )
        
        opt = AugmentedLagrangianOptimization(p1, ModifiedNewtonOptimization, rho=1000.0)
        start = time.clock() 
        output = opt.solve()
        elapsed = time.clock() - start
        print '%.2e\t%.2f\t%.2f\t%.4f'%(output.f, output.x[0], output.x[1],elapsed)
        
        graphicalOptimization(
            linspace(-0.2, 6.2, 100),
            x2_domain = linspace(-0.2, 3.4, 100),
            objectiveFunction = f, 
            inequalityFunctions = [g], 
            equalityFunctions = [], 
            levels=linspace(6,-22,8), 
            path = output.optimizationPath)
        pyplot.show()
        
    def testExample2(self):
        def f(x):
            x1, x2 = x[0], x[1]
            return 9.*x1**2. + 13.*x2**2. + 18.*x1*x2 - 4.
        def h1(x):
            x1, x2 = x[0], x[1]
            return x1**2. + x2**2. + 2.*x1 - 16
    
        p1 = ConstrainedProblemSetup(
            f = f, 
            g = [],
            h = [h1],
            x0 = [2.0, 2.0], 
            lineSearchMethod = goldenLineSearch,
            absoluteEpsilon = 1e-6,
            maxIterations = 500,
            storePoints=True
        )
        
        opt = ExternalPenaltyOptimization(p1, ModifiedNewtonOptimization,rho=100000.)
        start = time.clock() 
        output = opt.solve()
        elapsed = time.clock() - start
        print '%.2e\t%.2f\t%.2f\t%.4f'%(output.f, output.x[0], output.x[1],elapsed)
                 
        graphicalOptimization(
            x1_domain = linspace(-6.0, 6.0, 100),
            x2_domain = linspace(-6.0, 6.0, 100),
            objectiveFunction = f, 
            inequalityFunctions = [], 
            equalityFunctions = [h1], 
            levels=[5,15,50,100,200,300,450,600], 
            path = output.optimizationPath)
        pyplot.show()

        
    def testExample3(self):
        
        def f(x):
            x, y = x[0], x[1]
            return (1.0 - x)**2.0 + 100.0*(y - x**2.0)**2.0
        
        def g1(x):
            x, y = x[0], x[1]
            return x - 0.5
        
        def g2(x):
            x, y = x[0], x[1]
            return y - 1.0
        
        p1 = ConstrainedProblemSetup(
            f = f, 
            g = [g1, g2],
            h = [],
            x0 = [-1.5, -1.5], 
#             x0 = [0.5, 0.25], 
            lineSearchMethod = goldenLineSearch,
            absoluteEpsilon = 1e-6,
            maxIterations = 500,
            storePoints=True
        )
        
        opt = ExternalPenaltyOptimization(p1, ModifiedNewtonOptimization,rho=1000.)
        start = time.clock() 
        output = opt.solve()
        elapsed = time.clock() - start
        print '%.2e\t%.2f\t%.2f\t%.4f'%(output.f, output.x[0], output.x[1],elapsed)
        
        graphicalOptimization(
            linspace(-2.0, 3.0, 100), 
            x2_domain = linspace(-2.0, 4.0, 100), 
            objectiveFunction = f, 
            inequalityFunctions = [g1, g2], 
            equalityFunctions = [], 
            levels=linspace(5.0, 200.0,8), 
            path = output.optimizationPath)
        pyplot.show()
    
if __name__ == "__main__":
    import sys;
    sys.argv = ['', 'Test.testExample3']
    unittest.main()