from optimization.line_search import goldenLineSearch
from optimization.unconstrained_optimization import BaseUnconstrainedOptimization,\
    UnconstrainedProblemSetup
import numpy
from calculus import gradient


class ConstrainedProblemSetup(object):
    def __init__(
        self, 
        f,
        g,
        h, 
        x0,
        lineSearchMethod=goldenLineSearch, 
        absoluteEpsilon=1e-4, 
        relativeEpsilon=0.0, 
        maxIterations=200,
        storePoints=False
        ):
        self.f = f
        self.g = g
        self.h = h
        self.x0 = x0
        self.lineSearchMethod=lineSearchMethod
        self.absoluteEpsilon = absoluteEpsilon
        self.relativeEpsilon = relativeEpsilon
        self.maxIterations = maxIterations
        self.storePoints = storePoints
        

class ExternalPenaltyOptimization(object):
    
    def __init__(self, constrainedProblemSetup, baseUnconstrainedOptimization, rho=0.1):
        self.objectiveFunction = constrainedProblemSetup.f
        self.equalityConstraints = constrainedProblemSetup.h
        self.inequalityConstraints = constrainedProblemSetup.g
        self.x0 = constrainedProblemSetup.x0
        self.lineSearch = constrainedProblemSetup.lineSearchMethod
        self.unconstrainedOptimization = baseUnconstrainedOptimization
        self.rho = rho
        
        def transformedObjectiveFunction(x):
            result = self.objectiveFunction(x)
            for i in xrange(len(self.inequalityConstraints)):
                result += self.rho * (max(0.0, self.inequalityConstraints[i](x))) ** 2.0
            for i in xrange(len(self.equalityConstraints)):
                result += self.rho * (self.equalityConstraints[i](x)) ** 2.0
            return result
        self.transformedObjectiveFunction = transformedObjectiveFunction
        
    
    def solve(self):
        residual = None
        latestOptimization = None
        output = None
        iteration = 0
        completePath = [self.x0]
        while residual > 1e-6 or residual is None:
            print self.rho
            iteration += 1
            p1 = UnconstrainedProblemSetup(
                f = self.transformedObjectiveFunction, 
                x0 = self.x0, 
                lineSearchMethod = self.lineSearch,
                absoluteEpsilon = 1e-6,
                maxIterations = 500,
                storePoints=True
            )
            opt = self.unconstrainedOptimization(p1)
            output = opt.solve()
            if iteration > 1:
                residual = abs(output.f - latestOptimization)
            latestOptimization = output.f
            self.rho = self.rho * 2.0
            self.x0 = output.x
            completePath.extend(output.optimizationPath[1:])
            print output.f, residual
        output.optimizationPath = completePath
        return output
            
        
class AugmentedLagrangianOptimization(object):
    
    def __init__(self, constrainedProblemSetup, baseUnconstrainedOptimization, rho=0.1):
        self.objectiveFunction = constrainedProblemSetup.f
        self.equalityConstraints = constrainedProblemSetup.h
        self.inequalityConstraints = constrainedProblemSetup.g
        self.x0 = constrainedProblemSetup.x0
        self.lineSearch = constrainedProblemSetup.lineSearchMethod
        self.unconstrainedOptimization = baseUnconstrainedOptimization
        self.rho = rho
        self.mi = numpy.zeros(len(self.inequalityConstraints))
        self.mi[3] = 2511.59595516
#         for i in xrange(len(self.inequalityConstraints)):
#             self.mi[i] = 0.3
        #TODO: Implement lagrangian multipliers initialization for equality constraints
        #TODO: Improve this residuals (extract them to setup)
        self.absoluteResidualCriterium = 1.0e-6
        self.relativeResidualCriterium = 1.0e-10
        
        def transformedObjectiveFunction(x):
            result = self.objectiveFunction(x)
            rho = self.rho
            for i in xrange(len(self.inequalityConstraints)):
                mi = self.mi[i]
                g = self.inequalityConstraints[i]
                result += (
                    mi * max(-mi/rho, g(x)) +
                    rho * max(-mi/rho, g(x)) ** 2.0
                )
            for i in xrange(len(self.equalityConstraints)):
                #TODO: Implement equality constraints in Augmented Lagrangian
                assert False
            return result
        self.transformedObjectiveFunction = transformedObjectiveFunction    
        
    
    def solve(self):
        relativeResidual = None
        latestOptimization = None
        output = None
        completePath = [self.x0]
        
        iteration = 0
        converged = self._checkConvergenceCriteria(relativeResidual)
        while not converged:
            print 'rho', self.rho, 'mi', self.mi
            iteration += 1
            p1 = UnconstrainedProblemSetup(
                f = self.transformedObjectiveFunction, 
                x0 = self.x0, 
                lineSearchMethod = self.lineSearch,
                absoluteEpsilon = 1e-6,
                maxIterations = 500,
                storePoints=True
            )
            opt = self.unconstrainedOptimization(p1)
            print 'gradient of transformed:', gradient(self.transformedObjectiveFunction, self.x0)
            output = opt.solve()
            print output.x
            if iteration > 1:
                relativeResidual = abs(output.f - latestOptimization)
                
            self.rho = self.rho * 2.0
            latestOptimization = output.f
            self.x0 = output.x
            self._updateLagrangianMultipliers(self.x0)
            
            completePath.extend(output.optimizationPath[1:])
            converged = self._checkConvergenceCriteria(relativeResidual)
            
        output.optimizationPath = completePath
        print output.f
        return output
        
    def _updateLagrangianMultipliers(self, x):
        for i in xrange(len(self.inequalityConstraints)):
            self.mi[i] = max(0.0, self.mi[i]+ self.rho * self.inequalityConstraints[i](x))
#             self.mi[i] = self.mi[i]+ self.rho * self.inequalityConstraints[i](x)
        for i in xrange(len(self.equalityConstraints)):
            #TODO: Implement equality constraints in Augmented Lagrangian
            assert False
            
    
    def _updatePenaltyParameter(self, x0, x1):
        #TODO: Evaluate if this is worth doing
        h0 = numpy.zeros(len(self.inequalityConstraints))
        h1 = numpy.zeros(len(self.inequalityConstraints))
        for i in xrange(len(self.inequalityConstraints)):
            pass
            
    
    def _checkConvergenceCriteria(self, relativeResidual):
        product = numpy.zeros(len(self.inequalityConstraints))
        for i in xrange(len(self.inequalityConstraints)):
            product[i] = self.mi[i] * self.inequalityConstraints[i](self.x0)
        absoluteResidual = numpy.linalg.norm(product)
        print 'abs:', absoluteResidual, 'rel:', relativeResidual
        if absoluteResidual < self.absoluteResidualCriterium:
            return True
        if relativeResidual is None:
            return False
        if relativeResidual < self.relativeResidualCriterium:
            return True
        return False
