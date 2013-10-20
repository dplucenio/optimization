from calculus import gradient, hessian
from optimization.direction_search import steepestDescentDirection, conjugateGradientDirection,\
    newtonDirection, modifiedNewtonDirection
from optimization.line_search import goldenLineSearch, checkDescentDirection, armijoLineSearch,\
    quadraticLineSearch, equalLineSearch
from numpy.linalg import norm

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
            self.gradient = self.objectiveFunction.gradient
        else:
            def hess(x):
                return hessian(self.objectiveFunction, x)
            self.hessian = hess
            
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
        alpha = goldenLineSearch(self.objectiveFunction, self.current_x, d)
        self.current_x = self.current_x + alpha * d
        self.current_grad = self.gradient(self.current_x)
        
class ConjugateGradientOptimization(BaseUnconstrainedOptimization):
    
    def doInitialize(self):
        self.current_x = self.x0
        self.current_grad = self.gradient(self.current_x)
        self.old_grad = None
        
    def doSolve(self):
        d = conjugateGradientDirection(self.current_grad, self.old_grad)
        alpha = goldenLineSearch(self.objectiveFunction, self.current_x, d)
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
        alpha = goldenLineSearch(self.objectiveFunction, self.current_x, d)
        self.current_x = self.current_x + alpha * d
        self.current_grad = self.gradient(self.current_x)
        self.current_hess = self.hessian(self.current_x)
        
class ModifiedNewtonOptimization(BaseUnconstrainedOptimization):
    def doInitialize(self):
        self.current_x = self.x0
        self.current_grad = self.gradient(self.current_x)
        self.current_hess = self.hessian(self.current_x)
        self.rho = 1.0 # Set initially as a high constant

    def doSolve(self):
        d = modifiedNewtonDirection(self.rho, self.current_grad, self.current_hess)
        while not checkDescentDirection(self.objectiveFunction, self.current_x, d):
            print 'increasing rho, awesome. Science! :D'
            self.rho = 2.0 * self.rho
            d = modifiedNewtonDirection(self.rho, self.current_grad, self.current_hess)
        alpha = goldenLineSearch(self.objectiveFunction, self.current_x, d)
        self.current_x = self.current_x + alpha * d
        self.current_grad = self.gradient(self.current_x)
        self.current_hess = self.hessian(self.current_x)
        self.rho = 0.5 * self.rho