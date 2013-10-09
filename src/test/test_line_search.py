import unittest
from optimization.line_search import checkDescentDirection
from math import exp
from calculus import gradient


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


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testCheckDescentDirection']
    unittest.main()