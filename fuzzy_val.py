import math as m
import pickle as p
import numpy as np
from numpy.core.fromnumeric import var

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

def sigmoid_function_NN(x):
    return 1/(1+np.e**(-x))
        
class ANFIS_HL(FIS):

    def __init__(self, hidden_neurals, fis=None) -> None:
        super().__init__()
        if fis != None:
            self.fis = fis
        else:
            self.fis = super()
        np.random.seed(1)
        self.hidden_neurals = hidden_neurals
        self.s_hidden = None
        self.s_output = None
    
    def train_system(self, epoch, input_set, train_set):
        #Get terms and func from variables
        terms = [self.fis.val_input[i].terms for i in self.fis.val_input]
        func = [[j.func for j in i] for i in terms]
        args = [[j.arg for j in i] for i in terms]
        set_train = []
        #Make layers for NN inside FIS
        for var_num in input_set:
            for i in range(len(var_num)):
                set_train.append([func[i][j](var_num[i], args[i][j]) for j in range(len(func[i]))])
        l0_s = np.array(set_train)
        output = np.array(train_set)
        self.s_hidden = 2 * np.random.random((len(l0_s.T), self.hidden_neurals))
        self.s_output = 2 * np.random.random((self.hidden_neurals, len(output.T)))
        print("Hidden")
        print(self.s_hidden)
        print("out")
        print(self.s_output)
        for j in range(epoch):
            l0 = l0_s
            #Synopsys
            l1 = sigmoid_function_NN(l0.dot(self.s_hidden))
            l2 = sigmoid_function_NN(l1.dot(self.s_output))
            #deltas
            l2_delta = (output - l2) * (l2 * (1 - l2))
            l1_delta = l2_delta.dot(self.s_output.T) * (l1 * (1 - l1))
            self.s_output += l1.T.dot(l2_delta)
            self.s_hidden += l0.T.dot(l1_delta)
        print("Hidden")
        print(self.s_hidden)
        print("out")
        print(self.s_output)

    def calc_after_train(self, input):
        terms = [self.fis.val_input[i].terms for i in self.fis.val_input]
        func = [[j.func for j in i] for i in terms]
        args = [[j.arg for j in i] for i in terms]
        input_set = []
        #Make layers for NN inside FIS
        for var_num in input:
            for i in range(len(var_num)):
                input_set.append([func[i][j](var_num[i], args[i][j]) for j in range(len(func[i]))])
        input_array = np.array(input_set)
        r = (sigmoid_function_NN(input_array.dot(self.s_hidden)).dot(self.s_output))
        print("result")
        print(r)

  

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
