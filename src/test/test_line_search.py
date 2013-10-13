from math import exp
import unittest
from numpy import array
from calculus import gradient
from optimization.line_search import checkDescentDirection, goldenSearch, armijoSearch


class Test(unittest.TestCase):

    def testCheckDescentDirection(self):
        def f(x):
            x1=x[0]
            x2=x[1]
            return x1**2.0 -x1*x2 +2.0*x2**2.0 - 2.0*x1 + exp(x1+x2)
        
        x = [0.0, 0.0]
        d = [1.0, 2.0]
        self.assertFalse(checkDescentDirection(f, x, d))
        
        # The opposite direction of the gradient at a given point points to the maximum descent
        # direction
        d = -gradient(f, x)
        self.assertTrue(checkDescentDirection(f, x, d))
        
    def testGoldenSearch1(self):
        def f(x):
            return 3.0*x[0]**2.0 + 2.0*x[0]*x[1] + 2.0*x[1]**2.0 + 7.0
        x=array([1.0,2.0])
        d=array([-1.0,-1.0])
        alpha = goldenSearch(f, x, d, delta=0.01)
        self.assertAlmostEqual(alpha, 10.0/7.0, 6)
        
    def testGoldenSearch2(self):
        def f(x):
            return x[0]**2.0
        x=array([-1.0])
        d=array([2.0])
        self.assertAlmostEqual(goldenSearch(f, x, d), 0.5, 6)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testCheckDescentDirection']
    unittest.main()