import unittest
from optimization.direction_search import steepestDescentDirection, conjugateGradientDirection,\
    newtonDirection, modifiedNewtonDirection
from optimization.line_search import goldenLineSearch
from calculus import gradient, hessian
from numpy import array
from numpy.testing import assert_almost_equal

def aroraExample_8_4(x):
    x1, x2 = x[0], x[1]
    return x1**2.0 + x2**2.0 -2.0*x1*x2
    
def aroraExample_8_5(x):
    x1, x2, x3 = x[0], x[1], x[2]
    return x1**2.0 + 2.0*x2**2.0 + 2.0*x3**2.0 + 2.0*x1*x2 + 2.0*x2*x3

def aroraExample_9_6(x):
    x1, x2 = x[0], x[1]
    return 3.0*x1**2.0 + 2.0*x1*x2 + 2.0*x2**2.0 + 7.0

class Test(unittest.TestCase):


    def teststeepestDescentDirection_8_4(self):
        f = aroraExample_8_4
        def grad_f(x):
            return gradient(f, x)
        
        x_0 = [1.0, 0.0]
        grad_0 = grad_f(x_0)
        direction_0 = steepestDescentDirection(grad_0)
        alpha = goldenLineSearch(f, x_0, direction_0)
        assert_almost_equal(direction_0, [-2.0, 2.0], 5)
        assert_almost_equal(alpha, 0.25, 6)
        
        x1 = x_0 + alpha*direction_0
        assert_almost_equal(grad_f(x1), [0.0, 0.0] , 5)
        
    def teststeepestDescentDirection_8_5_numeric(self):
        f = aroraExample_8_5
        def grad_f(x):
            return gradient(f, x)
        
        x_0 = [2.0, 4.0, 10.0]
        grad_0 = grad_f(x_0)
        direction_0 = steepestDescentDirection(grad_0)
        alpha = goldenLineSearch(f, x_0, direction_0)
        assert_almost_equal(direction_0, [-12.0, -40.0, -48.0], 5)
        assert_almost_equal(alpha, 0.15872, 5)
        
        x_1 = x_0 + alpha*direction_0
        assert_almost_equal(grad_f(x_1), [-4.50688457, -4.4416123, 4.82814806] , 5)
        assert_almost_equal(x_1, [0.09535947, -2.34880176, 2.38143789] , 5)
        
    def teststeepestDescentDirection_8_5_analytical(self):
        f = aroraExample_8_5
        def grad_f(x):
            x_1, x2, x3 = x[0], x[1], x[2]
            return array([2.0*x_1 + 2.0*x2, 4.0*x2 + 2.0*x_1 + 2.0*x3, 4.0*x3 + 2.0*x2])
        
        x_0 = [2.0, 4.0, 10.0]
        grad_0 = grad_f(x_0)
        direction_0 = steepestDescentDirection(grad_0)
        alpha = goldenLineSearch(f, x_0, direction_0)
        assert_almost_equal(direction_0, [-12.0, -40.0, -48.0], 5)
        assert_almost_equal(alpha, 0.15872, 5)
         
        x_1 = x_0 + alpha*direction_0
        assert_almost_equal(grad_f(x_1), [-4.50688457, -4.4416123, 4.82814806] , 5)
        assert_almost_equal(x_1, [0.09535947, -2.34880176, 2.38143789] , 5)

    def testconjugateGradientDirection_8_5_numeric(self):
        f = aroraExample_8_5
        def grad_f(x):
            return gradient(f, x)
        
        x_0 = [2.0, 4.0, 10.0]
        grad_0 = grad_f(x_0)
        direction_0 = conjugateGradientDirection(grad_0)
        alpha = goldenLineSearch(f, x_0, direction_0)
        assert_almost_equal(direction_0, [-12.0, -40.0, -48.0], 5)
        assert_almost_equal(alpha, 0.15872, 5)
        
        x_1 = x_0 + alpha*direction_0
        grad_1 = grad_f(x_1)
        direction_1 = conjugateGradientDirection(grad_1, grad_0)
        alpha = goldenLineSearch(f, x_1, direction_1)
        assert_almost_equal(direction_1, [4.31908533,  3.81561485, -5.579345], 5)
        assert_almost_equal(alpha, 0.31545152, 5)
        

    def testNewtonDirection_9_6_analytical(self):
        f = aroraExample_9_6
        def grad_f(x):
            x1, x2 = x[0], x[1]
            return array([6.0*x1 + 2.0*x2, 2.0*x1 + 4.0*x2])
        def hessian_f(x):
            return array([[6,2],[2,4]])
            
        x_0 = [5.0, 10.0]
        grad_0 = grad_f(x_0)
        hess_0 = hessian_f(x_0)
        direction_0 = newtonDirection(grad_0, hess_0)
        alpha_0 = goldenLineSearch(f, x_0, direction_0)
        x_1 = x_0 + alpha_0*direction_0
        assert_almost_equal(direction_0, [-5.0, -10.0], 5)
        assert_almost_equal(alpha_0, 1.0, 5)
        assert_almost_equal(x_1, [0.0, 0.0], 5)
        
    def testNewtonDirection_9_6_numeric(self):
        f = aroraExample_9_6
        def grad_f(x):
            return gradient(f, x)
        def hessian_f(x):
            return hessian(f, x)
            
        x_0 = [5.0, 10.0]
        grad_0 = grad_f(x_0)
        hess_0 = hessian_f(x_0)
        direction_0 = newtonDirection(grad_0, hess_0)
        alpha_0 = goldenLineSearch(f, x_0, direction_0)
        x_1 = x_0 + alpha_0*direction_0
        assert_almost_equal(direction_0, [-5.0, -10.0], 1)
        assert_almost_equal(alpha_0, 1.0, 2)
        assert_almost_equal(x_1, [0.0, 0.0], 5)
        
    def testModifiedNewtonDirection(self):
        f = aroraExample_9_6
        def grad_f(x):
            return gradient(f, x)
        def hessian_f(x):
            return hessian(f, x)
        x_0 = [5.0, 10.0]
        grad_0 = grad_f(x_0)
        hess_0 = hessian_f(x_0)
        
        steepestDir = steepestDescentDirection(grad_0)
        alpha = goldenLineSearch(f, x_0, steepestDir)
        x_1_steepest = x_0 + alpha*steepestDir
        
        newtonDir = newtonDirection(grad_0, hess_0)
        alpha = goldenLineSearch(f, x_0, newtonDir)
        x_1_newton = x_0 + alpha*newtonDir
        
        # Modified newton direction with rho = 0.0 after alpha is calculated should give equivalent
        # new position as classic newton method
        modifiedNewtonDirLow = modifiedNewtonDirection(0.0, grad_0, hess_0)
        alpha = goldenLineSearch(f, x_0, modifiedNewtonDirLow)
        x_1_modNewton = x_0 + alpha*modifiedNewtonDirLow
        assert_almost_equal(x_1_modNewton, x_1_newton, 2)
        
        # Modified newton direction with high rho after alpha is calculated should give equivalent
        # new position as steepest descent method
        modifiedNewtonDirHigh = modifiedNewtonDirection(1.0e3, grad_0, hess_0)
        alpha = goldenLineSearch(f, x_0, modifiedNewtonDirHigh)
        x_1_modNewton = x_0 + alpha*modifiedNewtonDirHigh
        assert_almost_equal(x_1_modNewton, x_1_steepest, 2)
        
        
    
    
    
if __name__ == "__main__":
    unittest.main()