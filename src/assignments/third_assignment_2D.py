from numpy import zeros, transpose, mat, array, linspace
from numpy.linalg import solve
from optimization.constrained_optimization import ConstrainedProblemSetup,\
    AugmentedLagrangianOptimization, ExternalPenaltyOptimization
from optimization.line_search import goldenLineSearch, armijoLineSearch
from optimization.unconstrained_optimization import ModifiedNewtonOptimization, BfgsOptimization
from optimization.graphical_optimization import graphicalOptimization, gradientOf2dFunction
import matplotlib.pyplot as pyplot
from calculus import gradient

E = 2.0e11
L = 0.3
b = 0.05
P = 5.0e3
V_max = b * L *(0.1 + 0.1 + 0.1)

h1_min = 0.02
h2_min = 0.02
h3_min = 0.07

fixed_h3 = 0.07

def f(x):
#     print x
    h1, h2, h3 = x[0], x[1], fixed_h3
    
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

    X=None
    try:
        X = solve(A,B)
    except:
        print x
    
    X = mat(X)
    A = mat(A)

#     print A
#     print b
#     print x
#     print x.T 
#     print x * A * x.T
    return X * A * X.T

def g1(x):
    h1, h2, h3 = x[0], x[1], fixed_h3
    return h1_min - h1 

def g2(x):
    h1, h2, h3 = x[0], x[1], fixed_h3
    return h2_min - h2

def g3(x):
    h1, h2, h3 = x[0], x[1], fixed_h3
    return h3_min - h3

def g4(x):
    h1, h2, h3 = x[0], x[1], fixed_h3
    V = b * L *(h1 + h2 + h3)
    return V - V_max
    
x0 = [0.03, 0.075]

p1 = ConstrainedProblemSetup(
    f = f, 
    g = [g1, g2, g4],
    h = [],
    x0 = x0, 
    lineSearchMethod = goldenLineSearch,
    absoluteEpsilon = 1e-6,
    maxIterations = 10,
    storePoints=True
)

opt = AugmentedLagrangianOptimization(p1, ModifiedNewtonOptimization, rho=100000000.0)
output = opt.solve()
print output.x, output.f

graphicalOptimization(
    x1_domain = linspace(0.01, 0.21, 21),
    x2_domain = linspace(0.01, 0.21, 21),
    objectiveFunction = f, 
    inequalityFunctions = [g1,g2,g4], 
    equalityFunctions = [], 
    levels=[5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 150.0, 300.0],
    path=output.optimizationPath
    )
pyplot.show()







    