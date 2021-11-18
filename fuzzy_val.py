import math as m
import pickle as p

from numpy import divide 

class Term:
    '''
    Describe term of fuzzy variable
    '''
    def __init__(self, name, func, *arg):
        self.name = name
        self.func = func
        self.arg = arg

    def fuzzification(self, x):
        return self.func(x, self.arg)

class FuzzyVar:
    '''
    Fuzzy value
    '''
    def __init__(self) -> None:
        self.terms = list()
        pass

    def add_term(self, term:Term):
        self.terms.append(term)
    
    def print_val_memberships(self, x) -> None:
        for term in self.terms:
            print("{} : {}".format(term.name, term.fuzzification(x)))
    
    def get_val_memberships(self, x) -> dict:
        return [[term.name , term.fuzzification(x)] for term in self.terms]

class FIS:

    def __init__(self) -> None:
        self.val_input = dict()
        self.rules = dict()
        pass

    def save_fis_to_file(path:str, fis):
        with open(path, "wb") as f:
            p.dump(fis, f)

    def load_fis_from_file(path:str):
        with open(path, "rb") as f:
            return p.load(f)            

    def add_input_value(self, f_var:FuzzyVar):
        self.val_input[len(self.val_input) + 1] = f_var

    def calc_centroid(self, *arg):
        '''
        *args is argument for inputs
        '''
        area = 0; divider = 0
        for i in self.val_input:
            divider += max(term.fuzzification(arg[i - 1]) for term in self.val_input[i].terms)
            area += (divider * arg[i - 1])
        return area/divider
  

def triangle_func(x,arg):
    '''
    Get 3 coef a,b,c
    '''
    if x <= arg[0]:
        return 0
    elif arg[0] <= x and x <= arg[1]:
        return (x - arg[0])/(arg[1] - arg[0])
    elif arg[1] <= x and x <= arg[2]:
        return (arg[2]-x)/(arg[2]-arg[1])
    elif arg[2] <= x:
        return 0

def trapezoid_func(x, arg):
    '''
    Get 4 coef a,b,c,d
    '''
    if x <= arg[0]:
        return 0
    elif arg[0] <= x and x <= arg[1]:
        return (x-arg[0])/(arg[1]-arg[0])
    elif arg[1] <= x and x <= arg[2]:
        return 1
    elif arg[2] <= x and x <= arg[3]:
        return (arg[3]-x)/(arg[3]-arg[2])
    elif arg[3] <= x:
        return 0

def sigmoid_func(x, arg):
    '''
    x - value
    d1 - coeficient of up curve
    d2 - coeficient of down curve
    '''
    return 1/(1+(m.e**(- arg[1]*(x-arg[0]))))

def s_func(x,arg):
    '''
    Get 2 coef a,b
    '''
    if x < arg[0]:
        return 0
    elif arg[0] <= x and x <= arg[1]:
        return (1/2)+(1/2)* m.cos((x-arg[1])/(arg[1]-arg[0])* m.pi)
    elif x > arg[1]:
        return 1

def z_func(x,arg):
    '''
    Get 2 coef a,b
    '''
    if x < arg[0]:
        return 1
    elif arg[0] <= x and x <= arg[1]:
        return (1/2)+(1/2)* m.cos((x-arg[0])/(arg[1]-arg[0])* m.pi)
    elif x > arg[1]:
        return 0

def s_linear_func(x,arg):
    '''
    Get 2 coef a,b
    '''
    if x <= arg[0]:
        return 0
    elif arg[0] < x and x < arg[1]:
        return (x-arg[0])/(arg[1]-arg[0])
    elif arg[1] <= x:
        return 1

def z_linear_func(x,arg):
    '''
    Get 2 coef a,b
    '''
    if x <= arg[0]:
        return 1
    elif arg[0] < x and x < arg[1]:
        return (arg[1]-x)/(arg[1]-arg[0])
    elif arg[1] <= x:
        return 0

def P_sz_func(x,arg):
    '''
    Get 4 coef a,b,c,d
    '''
    return min(s_func(x,arg[0],arg[1]), z_func(x,arg[2],arg[3]))

def bell_func(x,arg):
    '''
    Get 3 coef a,b,c
    '''
    return 1/(1+((x-arg[2])/arg[0])**2*arg[1])

def gaussian_func(x,arg):
    '''
    Get 2 coef a,c
    '''
    return m.e**(-((x-arg[1])**2)/(2*arg[0]**2))
