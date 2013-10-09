from numpy import dot
from calculus import gradient

def checkDescentDirection(function, point, direction):
    '''
    '''
    # TODO: Write documentation explaining this property of descent direction
    return dot(gradient(function, point), direction) < 0

def equalIntervalSearch(function, point, direction, delta=0.1, convergence_criterium=1e-6, safe_mode=False):
    '''
    '''
    if safe_mode:
        assert checkDescentDirection(function, point, direction)
    
        
