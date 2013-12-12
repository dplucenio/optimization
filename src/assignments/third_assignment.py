from numpy import zeros, transpose, mat, array, linspace
from numpy.linalg import solve
from optimization.constrained_optimization import ConstrainedProblemSetup,\
    AugmentedLagrangianOptimization, ExternalPenaltyOptimization
from optimization.line_search import goldenLineSearch
from optimization.unconstrained_optimization import ModifiedNewtonOptimization, BfgsOptimization
from optimization.graphical_optimization import graphicalOptimization
import matplotlib.pyplot as pyplot
from calculus import gradient

E = 2.0e11
L = 0.3
b = 0.05
P = 5.0e3
V_max = b * L *(0.1 + 0.1 + 0.1)

h1_min = 0.02
h2_min = 0.02
h3_min = 0.02

def f(x):
#     print x
    h1, h2, h3 = x[0], x[1], x[2]
    
    k1 = 12.0 * E / L ** 3.0
    k2 = 6.0 * E / L ** 2.0
    k3 = 4.0 * E / L
    k4 = 2.0 * E / L
    
    I1 = b * h1**3.0 / 12.0
    I2 = b * h2**3.0 / 12.0
    I3 = b * h3**3.0 / 12.0
    
    A = zeros((6,6))
    B = zeros(6)
    
    A[0][0] = k1 * (I1 + I2)
    A[0][1] = k2 * (I2 - I1)
    A[0][2] = -k1 * I2
    A[0][3] = k2 * I2
    A[0][4] = 0.0
    A[0][5] = 0.0
    
    A[1][0] = k2 * (I2 - I1)
    A[1][1] = k3 * (I1 + I2)
    A[1][2] = -k2 * I2
    A[1][3] = k4 * I2
    A[1][4] = 0.0
    A[1][5] = 0.0
    
    A[2][0] = -k1 * I2
    A[2][1] = -k2 * I2
    A[2][2] = k1 * (I2 + I3)
    A[2][3] = k2 * (I3 - I2)
    A[2][4] = -k1 * I3
    A[2][5] = k2 * I3
    
    
    A[3][0] = k2 * I2
    A[3][1] = k4 * I2
    A[3][2] = k2 * (I3 - I2)
    A[3][3] = k3 * (I2 + I3)
    A[3][4] = -k2 * I3
    A[3][5] = k4 * I3
    
    A[4][0] = 0.0
    A[4][1] = 0.0
    A[4][2] = -k1 * I3
    A[4][3] = -k2 * I3
    A[4][4] = k1 * I3
    A[4][5] = -k2 * I3
    
    A[5][0] = 0.0
    A[5][1] = 0.0
    A[5][2] = k2 * I3
    A[5][3] = k4 * I3
    A[5][4] = -k2 * I3
    A[5][5] = k3 * I3
    
    B[4] = -P

    x = solve(A,B)
    
    x = mat(x)
    A = mat(A)

#     print A
#     print b
#     print x
#     print x.T 
#     print x * A * x.T
    return x * A * x.T

def g1(x):
    h1, h2, h3 = x[0], x[1], x[2]
    return h1_min - h1 

def g2(x):
    h1, h2, h3 = x[0], x[1], x[2]
    return h2_min - h2

def g3(x):
    h1, h2, h3 = x[0], x[1], x[2]
    return h3_min - h3

def g4(x):
    h1, h2, h3 = x[0], x[1], x[2]
    V = b * L *(h1 + h2 + h3)
    return V - V_max
    
# x0 = [0.03, 0.03, 0.03]
x0 = [0.14216274, 0.11075718, 0.0680922]

print f(x0)
print f([0.030001, 0.03, 0.03])
print f([0.03, 0.030001, 0.03])
print f([0.03, 0.03, 0.030001])
print gradient(f, x0)
# raw_input()

p1 = ConstrainedProblemSetup(
    f = f, 
    g = [g1, g2, g3, g4],
    h = [],
    x0 = x0, 
    lineSearchMethod = goldenLineSearch,
    absoluteEpsilon = 1e-3,
    maxIterations = 50,
)

opt = AugmentedLagrangianOptimization(p1, ModifiedNewtonOptimization, rho=1000.0)
# opt = ExternalPenaltyOptimization(p1, ModifiedNewtonOptimization, rho=1000.0)
opt.solve()

# graphicalOptimization(
#     linspace(0.01, 0.05, 100),
#     x2_domain = (0.01, 0.05, 100),
#     objectiveFunction = f, 
#     inequalityFunctions = [g1,g2,g4], 
#     equalityFunctions = [], 
#     levels=linspace(0.0, 100.0 ,8) 
#     )
# pyplot.show()







    