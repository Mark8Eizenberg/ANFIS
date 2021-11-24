import itertools as it
from typing import Iterable

s = [['a','b','c'], ['s','f'], ['x','c','v']]
d = [[10,1],[100,1000,10000,0.01],[0.1]]
print(s)
def make_combinations_calc(func, s:Iterable[Iterable]):
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

print(d)
print(make_combinations_calc(lambda a,b : a+b, s)) 
