import math as m
import pickle as p
import numpy as np
from typing import Iterable

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
        '''
        Print terms anf fuzification value
        '''
        for term in self.terms:
            print("{} : {}".format(term.name, term.fuzzification(x)))
    
    def get_val_memberships(self, x) -> dict:
        '''
        return list with lists of terms and fuzzification values for each one
        '''
        return [[term.name , term.fuzzification(x)] for term in self.terms]

class FIS:

    def __init__(self) -> None:
        self.val_input = dict()
        pass

    def save_fis_to_file(path:str, fis):
        '''
        save FIS object to binary file
        '''
        with open(path, "wb") as f:
            p.dump(fis, f)

    def load_fis_from_file(path:str):
        '''
        Load FIS object from file
        '''
        with open(path, "rb") as f:
            return p.load(f)            

    def add_input_value(self, f_var:FuzzyVar):
        self.val_input[len(self.val_input) + 1] = f_var

    def calc_centroid(self, *arg):
        '''
        *args is argument for inputs
        '''
        area = 0; divider = 1
        for i in self.val_input:
            divider += max(term.fuzzification(arg[i - 1]) for term in self.val_input[i].terms)
            area += (divider * arg[i - 1])
        return area/divider

def sigmoid_function_NN(x):
    '''
    Sigmoid function for NN
    '''
    return 1/(1+np.e**(-x))

def make_combinations_calc(func, s:Iterable[Iterable]):
    '''
    Combinatoric function
    Used for make multiplicator layers with rules
    '''
    set_s = []
    buff = []
    for i in range(len(s) - 1, -1, -1):  
        if set_s:
            buff = set_s.copy()
            set_s.clear()
            for j in s[i]:
                for k in buff:
                    set_s.append(func(j,k))
        else:
            for j in s[i]:
                set_s.append(j)
    return set_s


class FIONS(FIS):

    def __init__(self, hidden_neurals, output_var:FuzzyVar, fis=None ) -> None:
        super().__init__()
        if fis != None:
            self.fis = fis
        else:
            self.fis = super()
        np.random.seed(1)
        self.output_var = output_var
        self.hidden_neurals = hidden_neurals
        self.rules = dict()
        self.s_hidden = None
        self.s_output = None
    
    def init_system(self):
        '''
        Initialize net architecture
        after that, if you change architect of network
        you need init system again
        '''
        terms = [self.fis.val_input[i].terms for i in self.fis.val_input]
        args = [[j.arg for j in i] for i in terms]
        num_of_multiplicator_neurons = 1
        for j in [len(i) for i in args]:
            num_of_multiplicator_neurons *= j
        #Create hidden layers
        self.s_hidden =  2 * np.random.random((num_of_multiplicator_neurons, self.hidden_neurals))
        self.s_output = 2 * np.random.random((self.hidden_neurals, 1))

    def _train_system(self, input, train):
        '''
        Train system on one set
        '''
        terms = [self.fis.val_input[i].terms for i in self.fis.val_input]
        func = [[j.func for j in i] for i in terms]
        args = [[j.arg for j in i] for i in terms]
        assert(len(train) == len(self.output_var.terms))
        out = np.array([self.output_var.terms[i].func(train[i],self.output_var.terms[i].arg ) for i in range(len(train))])
        input_fuzzy = []
        for i in range(len(input)):
            input_fuzzy.append([func[i][j](input[i], args[i][j]) for j in range(len(func[i]))])
        rules_input_multiplication = np.array([[i] for i in make_combinations_calc(lambda a,b : min(a,b), input_fuzzy )])
        l0 = rules_input_multiplication
        l1 = np.array(sigmoid_function_NN(l0.T.dot(self.s_hidden)))
        l2 = sigmoid_function_NN(l1.dot(self.s_output))
        l2_delta = (out - l2) * (l2 * (1 - l2))
        l1_delta = l2_delta.dot(self.s_output.T) * (l1 * (1 - l1))
        self.s_output += l1.T.dot(l2_delta)
        self.s_hidden += l0.dot(l1_delta)

    def train_fions(self, epoch, input_dataset, answer_dataset):
        '''
        Train anfis system using trainset and num of epoch
        '''
        assert(len(input_dataset) == len(answer_dataset))
        for j in range(epoch):
            for i in range(len(input_dataset) - 1):
                self._train_system(input_dataset[i], answer_dataset[i])

    def calc_after_train(self, input):
        '''
        Calculation output of system
        use it after initialization or training
        '''
        terms = [self.fis.val_input[i].terms for i in self.fis.val_input]
        func = [[j.func for j in i] for i in terms]
        args = [[j.arg for j in i] for i in terms]
        input_fuzzy = []
        for i in range(len(input)):
            input_fuzzy.append([func[i][j](input[i], args[i][j]) for j in range(len(func[i]))])
        rules_input_multiplication = np.array([[i] for i in make_combinations_calc(lambda a,b : a*b, input_fuzzy )])
        l0 = rules_input_multiplication
        l1 = np.array(sigmoid_function_NN(l0.T.dot(self.s_hidden)))
        l2 = sigmoid_function_NN(l1.dot(self.s_output))
        return l2

    def save_fions_to_file(path:str, model):
        '''
        save FIONS object to binary file
        '''
        with open(path, "wb") as f:
            p.dump(model, f)

    def load_fions_from_file(path:str):
        '''
        Load FIONS object from file
        '''
        with open(path, "rb") as f:
            return p.load(f)        


  

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
