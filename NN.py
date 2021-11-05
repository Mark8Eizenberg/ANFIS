import matplotlib.pyplot as plt
import numpy as nm
import math as m
import fuzzy_val as fv


x = nm.arange(0,10,0.01)

A = fv.Term('Warm', fv.bell_func, 2,5,5)
B = fv.Term('Cold', fv.z_func, 2,4)
C = fv.Term('Hot', fv.s_func, 6,8)

ff = fv.FuzzyVar()

ff.add_term(A)
ff.add_term(B)
ff.add_term(C)
print([i for i in ff.get_val_memberships(4.2)])

plt.plot(x,[A.fuzzification(i) for i in x], 'tab:green')
plt.plot(x,[B.fuzzification(i) for i in x], 'tab:blue')
plt.plot(x,[C.fuzzification(i) for i in x], 'tab:red')
plt.grid(True)
plt.show()
