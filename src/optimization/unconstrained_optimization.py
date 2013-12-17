from numpy import dot, identity, mat, array
from numpy.linalg import norm

from calculus import gradient, hessian
from optimization.direction_search import steepestDescentDirection, conjugateGradientDirection, \
    newtonDirection, modifiedNewtonDirection
from optimization.line_search import goldenLineSearch, assertDescentDirection,\
    assertDescentDirectionWithGradient


class UnconstrainedProblemSetup(object):
    def __init__(
        self, 
        f, 
        x0,
        lineSearchMethod=goldenLineSearch, 
        absoluteEpsilon=1e-4, 
        relativeEpsilon=0.0, 
        maxIterations=200,
        storePoints=False
        ):
        self.f = f
        self.x0 = x0
        self.lineSearchMethod=lineSearchMethod
        self.absoluteEpsilon = absoluteEpsilon
        self.relativeEpsilon = relativeEpsilon
        self.maxIterations = maxIterations
        self.storePoints = storePoints
        
class OptimizationOutput(object):
    def __init__(self):
        self.f = None
        self.x = None
        self.iterations = None
        self.convergeceStatus = None
        self.convergeceReason = None
        self.optimizationPath = None
        self.residual = None
        

class BaseUnconstrainedOptimization(object):
    def __init__(self, unconstrainedProblemSetup):
        self.x0 = unconstrainedProblemSetup.x0 
        self.objectiveFunction = unconstrainedProblemSetup.f
        
        # Assigning numeric gradient and hessian calculation callbacks if the objective function
        # does not have analytical functions for them
        if (hasattr(self.objectiveFunction, 'gradient')):
            self.gradient = self.objectiveFunction.gradient
        else:
            def grad(x):
                return gradient(self.objectiveFunction, x)
            self.gradient = grad
        if (hasattr(self.objectiveFunction, 'hessian')):
            self.hessian = self.objectiveFunction.hessian
        else:
            def hess(x):
                return hessian(self.objectiveFunction, x)
            self.hessian = hess
        
        self.lineSearch = unconstrainedProblemSetup.lineSearchMethod
            
        self.current_grad = None
        self.current_x = None
        self.current_f = None
        self.status = None
        self.iterationCount = 0
        self.absoluteEpsilon = unconstrainedProblemSetup.absoluteEpsilon
        self.maxIterations = unconstrainedProblemSetup.maxIterations
        self.points=[]
        self.storePoints = unconstrainedProblemSetup.storePoints
        if self.storePoints:
            self.points.append(self.x0)
            
        
    def solve(self):
        self.doInitialize()
        while not self.stopCriteriumAttended():
            self.doSolve()
            if self.storePoints:
                self.points.append(self.current_x)
            self.iterationCount += 1
            
        output = OptimizationOutput()
        output.iterations = self.iterationCount
        output.x = self.current_x
        output.f = self.objectiveFunction(self.current_x)
        output.optimizationPath = self.points
        output.residual = norm(self.current_grad)
        return output
            
    def stopCriteriumAttended(self):
        return norm(self.current_grad) <= self.absoluteEpsilon or self.iterationCount > self.maxIterations
    
    def doInitialize(self):
        pass
    
    def doSolve(self):
        pass
        
    
class SteepestDescentOptimization(BaseUnconstrainedOptimization):
    
    def doInitialize(self):
        self.current_x = self.x0
        self.current_grad = self.gradient(self.current_x)
        
    def doSolve(self):
        d = steepestDescentDirection(self.current_grad)
        alpha = self.lineSearch(self.objectiveFunction, self.current_x, d)
        self.current_x = self.current_x + alpha * d
        self.current_grad = self.gradient(self.current_x)
        
class ConjugateGradientOptimization(BaseUnconstrainedOptimization):
    
    def doInitialize(self):
        self.current_x = self.x0
        self.current_grad = self.gradient(self.current_x)
        self.old_grad = None
        
    def doSolve(self):
        d = conjugateGradientDirection(self.current_grad, self.old_grad)
        alpha = self.lineSearch(self.objectiveFunction, self.current_x, d)
        self.current_x = self.current_x + alpha * d
        self.old_grad = self.current_grad
        self.current_grad = self.gradient(self.current_x)
        
class NewtonOptimization(BaseUnconstrainedOptimization):
    def doInitialize(self):
        self.current_x = self.x0
        self.current_grad = self.gradient(self.current_x)
        self.current_hess = self.hessian(self.current_x)

    def doSolve(self):
        d = newtonDirection(self.current_grad, self.current_hess)
        assertDescentDirectionWithGradient(self.current_grad, d)
        alpha = self.lineSearch(self.objectiveFunction, self.current_x, d)
        self.current_x = self.current_x + alpha * d
        self.current_grad = self.gradient(self.current_x)
        self.current_hess = self.hessian(self.current_x)
        
class ModifiedNewtonOptimization(BaseUnconstrainedOptimization):
    def doInitialize(self):
        self.current_x = self.x0
        self.current_grad = self.gradient(self.current_x)
        self.current_hess = self.hessian(self.current_x)
        
        # Marquardt modification: rho * I will be added to the hessian in case neton method does
        # not point to a descent direction
        self.rho = 0.001 # Set initially as a high constant
        
        # Additional parameters for modified newton method:
        # theta sets how far from a descent direction this methods allows. If its set to 0.0 it checks
        # strictly for descent directions
        self.theta = 0.1
        # betha is a parameter to not allow to small directions (observation, I think this is only 
        # usefull when inaccurate line search methods are used such as armijo)
        self.betha = 0.0

    def doSolve(self):
        d = modifiedNewtonDirection(self.rho, self.current_grad, self.current_hess)

        # First modification to newton method:
        while not dot(self.current_grad, d) < self.theta * norm(self.current_grad) * norm(d):
            self.rho = 2.0 * self.rho
            d = modifiedNewtonDirection(self.rho, self.current_grad, self.current_hess)
        
        # Second modification to newthon method
        if norm(d) < self.betha * norm(self.current_grad):
            d = d * self.betha *  norm(self.current_grad)/norm(d)
            
        alpha = self.lineSearch(self.objectiveFunction, self.current_x, d)
        self.current_x = self.current_x + alpha * d
        self.current_grad = self.gradient(self.current_x)
        self.current_hess = self.hessian(self.current_x)
        self.rho = 0.5 * self.rho
        
class DfpOptimization(BaseUnconstrainedOptimization):
    
    def doInitialize(self):
        self.current_x = self.x0
        self.current_grad = self.gradient(self.current_x)
        self.current_H = mat(identity(len(self.x0)))
        
    def doSolve(self):
        # Calculate search direction
        d =  - dot(array(self.current_H), self.current_grad)
        alpha = self.lineSearch(self.objectiveFunction, self.current_x, d)
        self.current_x = self.current_x + alpha * d
        self.old_grad = self.current_grad
        self.current_grad = self.gradient(self.current_x)
        
        # Update matrix A
        y = mat(self.current_grad - self.old_grad).T # Change in gradient
        s = mat(alpha * d).T # Change in design
        z = self.current_H * y
        
        B = (s * s.T) / (s.T * y)
        C = -(z * z.T) / (y.T * z)
        self.current_H = self.current_H + B + C
        
class BfgsOptimization(BaseUnconstrainedOptimization):
    
    def doInitialize(self):
        self.current_x = self.x0
        self.current_grad = self.gradient(self.current_x)
        self.current_H = mat(identity(len(self.x0)))
        
    def doSolve(self):
        # Calculate search direction
        d = newtonDirection(self.current_grad, array(self.current_H))
        alpha = self.lineSearch(self.objectiveFunction, self.current_x, d)
        print self.current_x, d, alpha
        self.current_x = self.current_x + alpha * d
        self.old_grad = self.current_grad
        self.current_grad = self.gradient(self.current_x)
        
        # Update matrix A
        y = mat(self.current_grad - self.old_grad).T # Change in gradient
        s = mat(alpha * d).T # Change in design
        c = mat(self.old_grad).T
        
        D = (y * y.T) / (y.T * s)
        E = (c * c.T) / dot(self.old_grad, d)
        self.current_H = self.current_H + D + E
        