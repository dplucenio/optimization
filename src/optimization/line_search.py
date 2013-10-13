from numpy import dot
from calculus import gradient

# TODO: Add documentation explaining techniques, meaning of a_l, a_u, d ...

def checkDescentDirection(function, point, direction):
    '''
    '''
    # TODO: Write documentation explaining this property of descent direction
    return dot(gradient(function, point), direction) < 0

def goldenSearch(f, x, d, delta=0.1, epsilon = 1e-9, max_iterations=5000):
    a_1 = 0.0
    a_2 = 0.0
    a_3 = 0.0
    i = 0
    while i <= max_iterations:
        f_2 = f(x + a_2 * d)
        f_3 = f(x + a_3 * d)
        if f_3 > f_2:
            return goldenBracketedSearch(f, x, d, a_1, a_3, epsilon)
        else:
            a_1 = a_2
            a_2 = a_3
            a_3 = a_3 + delta
        i += 1
        delta = delta*1.618
        # If reached maximum number of iterations, will return the last alpha found
    return a_3

            
def goldenBracketedSearch(f, x_k, d, a_l, a_u, epsilon=1e-9):
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
        return goldenBracketedSearch(f, x_k, d, a_l, a_u)
    return (a_l + a_u) * 0.5


def armijoSearch(f, x, d, a=1.0, rho=0.1, eta=1.4, safe_mode=False):
    if safe_mode:
        assert checkDescentDirection(f, x, d) , 'Direction passed is not a descent direction' 
    def f_a(alpha):
        return f(x + alpha * d)
    grad_f_a = gradient(f_a, [0.0])
    def armijoCriterium(alpha):
        return f_a(alpha) <= f_a(0) + rho * grad_f_a * alpha
        
    if armijoCriterium(a):
        while armijoCriterium(a):
            a = a * eta
        return a
    else:
        while not armijoCriterium(a):
            a = a / eta
        return a
    