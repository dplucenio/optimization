from math import exp
import unittest
from numpy import array
from calculus import gradient
from optimization.line_search import checkDescentDirection, goldenSearch, armijoSearch,\
    quadraticLineSearch
from scipy.constants.constants import alpha


def parabolaFunction(x):
    return x[0]**2.0
    
def aroraExample(x):
    return 3.0*x[0]**2.0 + 2.0*x[0]*x[1] + 2.0*x[1]**2.0 + 7.0

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
        x=array([1.0,2.0])
        d=array([-1.0,-1.0])
        alpha = goldenSearch(aroraExample, x, d, delta=0.01)
        self.assertAlmostEqual(alpha, 10.0/7.0, 6)
        
    def testGoldenSearch2(self):
        x=array([-1.0])
        d=array([2.0])
        self.assertAlmostEqual(goldenSearch(parabolaFunction, x, d), 0.5, 6)
        
    def testQuadraticLineSearch1(self):
        x=array([1.0,2.0])
        d=array([-1.0,-1.0])
        alpha = quadraticLineSearch(aroraExample, x, d, delta=0.01)
        self.assertAlmostEqual(alpha, 10.0/7.0, 6)
        
    def testQuadraticLineSearch2(self):
        x=array([-1.0])
        d=array([2.0])
        self.assertAlmostEqual(quadraticLineSearch(parabolaFunction, x, d), 0.5, 6)
        
    def testArmijoLineSearch1(self):
        x=array([1.0,2.0])
        d=array([-1.0,-1.0])
        alpha = armijoSearch(aroraExample, x, d)
        self.assertAlmostEqual(alpha, 1.96, 6)
        
    def testArmijoLineSearch2(self):
        x=array([-1.0])
        d=array([2.0])
        self.assertAlmostEqual(armijoSearch(parabolaFunction, x, d), 0.7, 6)
        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testCheckDescentDirection']
    unittest.main()