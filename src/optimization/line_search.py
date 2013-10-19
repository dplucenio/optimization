from numpy import dot
from calculus import gradient

# TODO: Add documentation explaining techniques, meaning of a_l, a_u, d ...

def checkDescentDirection(function, point, direction):
    '''
    '''
    # TODO: Write documentation explaining this property of descent direction
    return dot(gradient(function, point), direction) < 0

def alphaOneVariableFunction(f, x, d, alpha):
    return f(x + alpha * d)

def constantLineSearch(f, x, d):
    '''
    This constant line search is only used with classical Newton method where the step size is not 
    calculated, in other words:

     /->    \    /->\   ->  
    f|x     | = f|x | + d    
     \ k + 1/    \ k/
    
    alpha = 1.0
    
    tex: f\left(x_{k+1}^{>}\right)=f\left(x_{k}^{>}\right)+d^{>}
    '''
    return 1.0

def equalLineSearch(f, x, d, delta=0.1, epsilon = 1e-6, max_iterations=5000):
    a_l, a_u = _baseIntervalUncertaintyRegion(f, x, d, delta, epsilon, max_iterations, r=1.0)
    I = a_u - a_l
    while I > epsilon:
        delta=delta*0.5
        a_l, a_u = _baseIntervalUncertaintyRegion(f, 
            x, 
            d, 
            delta, 
            epsilon, 
            max_iterations, 
            r=1.0, 
            a_1=a_l,
            a_2=a_l,
            a_3=a_l
        )
        I = a_u - a_l
    return (a_u + a_l) * 0.5
        
def goldenLineSearch(f, x, d, delta=0.1, epsilon = 1e-6, max_iterations=5000):
    a_l, a_u = __goldenIntervalUncertaintyRegion(f, x, d, delta, epsilon, max_iterations)
    return __goldenIntervalUncertaintydSearch(f, x, d, a_l, a_u, epsilon)

def quadraticLineSearch(f, x, d, delta=0.1, epsilon = 1e-6, max_iterations=5000):
    a_l, a_u = __goldenIntervalUncertaintyRegion(f, x, d, delta, epsilon, max_iterations)
    return quadraticIntervalUncertaintydSearch(f, x, d, a_l, a_u, epsilon)

def armijoLineSearch(f, x, d, a=1.0, rho=0.2, eta=2.0, safe_mode=False):
    if safe_mode:
        assert checkDescentDirection(f, x, d) , 'Direction passed is not a descent direction'
         
    def f_a(alpha):
        return alphaOneVariableFunction(f, x, d, alpha)
    grad_f_a = gradient(f_a, [0.0])
    
    def armijoCriterium(alpha):
        return f_a(alpha) <= f_a(0) + alpha * rho * grad_f_a
        
    while armijoCriterium(a):
        a = a * 2.0
    while not armijoCriterium(a):
        a = a * 0.7
    return a

def _baseIntervalUncertaintyRegion(
    f, 
    x, 
    d, 
    delta=0.1, 
    epsilon = 1e-6, 
    max_iterations=5000, 
    r=1.0, 
    a_1=0.0, 
    a_2=0.0, 
    a_3=0.0
    ): 
    
    i = 0
    while i <= max_iterations:
        f_2 = f(x + a_2 * d)
        f_3 = f(x + a_3 * d)
        if f_3 > f_2:
            return (a_1, a_3)
        else:
            a_1 = a_2
            a_2 = a_3
            a_3 = a_3 + delta
        i += 1
        delta = delta * r
        # If reached maximum number of iterations, will return the last alpha found
    return a_1, a_3

def __goldenIntervalUncertaintyRegion(f, x, d, delta=0.1, epsilon = 1e-6, max_iterations=5000):
    return _baseIntervalUncertaintyRegion(f, x, d, delta, epsilon, max_iterations, r=1.618)
            
def __goldenIntervalUncertaintydSearch(f, x_k, d, a_l, a_u, epsilon=1e-6):
    I = a_u - a_l
    while I > epsilon:
        a_a = a_l + 0.382*I
        a_b = a_l + 0.618*I
        f_a = f(x_k + a_a * d)
        f_b = f(x_k + a_b * d)
        if f_a < f_b:
            a_u = a_b
        elif f_a > f_b:
            a_l = a_a
        elif f_a == f_b:
            a_l = a_a
            a_u = a_b
        I = a_u - a_l
    return (a_l + a_u) * 0.5

def quadraticIntervalUncertaintydSearch(f, x_k, d, a_l, a_u, epsilon=1e-6):
    '''
                   _                                 _  
             1    | f(a_u) - f(a_l)   f(a_i) - f(a_l) | 
    a2 = ---------| --------------- - --------------- | 
         a_u - a_i|_   a_u - a_l         a_i - a_l   _| 
         
         
    
         f(a_i) - f(a_l)              
    a1 = --------------- - a2(a_l + a_i)
            a_i - a_l                

                               2
    a0 = f(a_l) - a1a_l - a2a_l 

    '''
    def fa(alpha):
        return alphaOneVariableFunction(f, x_k, d, alpha)
    I = a_u - a_l
    while I > epsilon:
        a_i = (a_l + a_u)*0.5
        a2 = 1.0 / (a_u - a_i) * ( (fa(a_u) - fa(a_l))/(a_u - a_l) - (fa(a_i) - fa(a_l))/(a_i - a_l) )
        a1 = (fa(a_i) - fa(a_l))/(a_i - a_l) - a2*(a_l + a_i)
        amin = - 1.0 / (2.0*a2) * a1
        if a_i < amin:
            if fa(a_i) < fa(amin):
                a_u = amin
            else:
                a_l = a_i
                a_i = amin
        else:
            if fa(a_i) < fa(amin):
                a_l = amin
                a_u = a_u
            else:
                a_i = amin
                a_u = a_i
        I = a_u - a_l
    return (a_l + a_u) * 0.5
