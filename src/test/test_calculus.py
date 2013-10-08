import unittest
from calculus import gradient, hessian
from numpy.testing import assert_almost_equal

class Test(unittest.TestCase):


    def testConstantFunction(self):
        def f(x):
            return 1.0
        assert_almost_equal(gradient(f, [0.0]), [0.0], 3)
        
        
    def testLinearFunction(self):
        def f(x):
            return x[0]
        assert_almost_equal(gradient(f, [0.0]), [1.0], 3)


    def testCubicFunction(self):
        def f(x):
            x1 = x[0]
            return x1**3.0
        assert_almost_equal(gradient(f, [2.0]), [12.0], 3)
        
        
    def testTwoVariableFunction1(self):
        def f(x):
            x1 = x[0]
            x2 = x[1]
            return x1**2.0 + x2**2.0
        assert_almost_equal(gradient(f, [2.0, 2.0]), [4.0, 4.0], 3)
        
        
    def testTwoVariableFunction2(self):
        def f(vector_x):
            x = vector_x[0]
            y = vector_x[1]
            return (x - 2.0)**2.0 + (2.0 * y - 4.0)**2.0
        
        assert_almost_equal(gradient(f, [2.0, 2.0]), [0.0, 0.0], 3)
        assert_almost_equal(gradient(f, [0.0, 0.0]), [-4.0, -16.0], 3)

        
    def testHessian1(self):
        def f(x):
            return x[0]**3.0 + x[1]**3.0
        
        x=[1.0, 1.0]
        assert_almost_equal(gradient(f,x), [3.0, 3.0], 3)
        assert_almost_equal(hessian(f,x), [[6.0, 0.0],[0.0, 6.0]], 3)
        
        
    def testHessian2(self):
        def f(x):
            return x[0]**3.0 + x[1]**3.0 + 2.0*x[0]**2.0 + 3.0*x[1]**2.0- x[0]*x[1] + 2.0*x[0] + 4.0*x[1]
        
        x=[1.0, 2.0]
        assert_almost_equal(gradient(f,x), [7.0, 27.0], 3)
        assert_almost_equal(hessian(f,x), [[10.0, -1.0],[-1.0, 18.0]], 2)
        
        
    def testHessian3(self):
        def f(x):
            return (2.0*x[1] - 4.0)**2.0 + 2.0
        
        x=[2.0, 2.0]
        assert_almost_equal(gradient(f,x), [0.0, 0.0], 3)
        assert_almost_equal(hessian(f,x), [[0.0, 0.0],[0.0, 8.0]], 3)
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testLinearFunction']
    unittest.main()