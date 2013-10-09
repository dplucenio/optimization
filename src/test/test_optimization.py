import unittest
from optimization.optimality_conditions import firstOrderNecessaryCondition


class Test(unittest.TestCase):


    def testFirstOrderNecessaryCondition1(self):
        def f(x):
            return x[0]**2.0
        self.assertTrue(firstOrderNecessaryCondition(f, [0.0]))
        self.assertFalse(firstOrderNecessaryCondition(f, [0.001]))
        
        
    def testFirstOrderNecessaryCondition2(self):
        def f(x):
            x1=x[0]
            x2=x[1]
            return (x1 - 2.0)**2.0 + (2.0 * x2 - 4.0)**2.0
        
        self.assertTrue(firstOrderNecessaryCondition(f, [2.0, 2.0]))
        self.assertFalse(firstOrderNecessaryCondition(f, [2.0, 1.9]))


if __name__ == "__main__":
    unittest.main()