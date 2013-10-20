optimization-teapm
==================
Repository to store all my studies and experiments from my masters Optimization discipline
Unconstrained optimization algorithms
-------------------------------------
Simple usage: Define objective function, define unconstrained problem and solve with any UnconstrainedOptimization solver available so far (SteepestDescent, ConjugateGradient and Newton Methods. Soon: ModifiedNewton and QuasiNewton).

Four different types of line search methods are available: equal intervals, golden section, quadratic interpolation and armijo method.
```python
def rosenbrockFunction(x):
    x1, x2 = x[0], x[1] 
    return 10.0*x1**4.0 -20.0*x1**2.0*x2 + 10.0*x2**2.0 + x1**2.0 - 2.0*x1 + 5.0

p1 = UnconstrainedProblemSetup(
    f = rosenbrockFunction, 
    x0 = [- 1.0 , 3.0], 
    lineSearchMethod = goldenLineSearch, 
    absoluteEpsilon = 1e-6,
    maxIterations = 500,
    )

newtonOpt = NewtonOptimization(p1)
output = newtonOpt.solve()
```
Graphical optimization
----------------------
Tools to ouput two variables (for proof of concept, teaching) objective functions, equality and inequality constraints. Shows feasible region and also optimization path for the existing algorithms implemented so far.
![Figure3_21](/docs/figure_3_21.png)
![Figure_1](/docs/figure_1.png)
