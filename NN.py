import matplotlib.pyplot as plt
import numpy as nm
import math as m
import fuzzy_val as fv


x = nm.arange(-20,50,0.5)

A = fv.Term('Warm', fv.bell_func, 10,20,20)
B = fv.Term('Cold', fv.z_func, 10,15)
C = fv.Term('Hot', fv.s_func, 26,30)

ff = fv.FuzzyVar()
ff_2 = fv.FuzzyVar()
ff_3 = fv.FuzzyVar()

input = [20,50, 10]

ff.add_term(A)
ff.add_term(B)
ff.add_term(C)
print([i for i in ff.get_val_memberships(input[0])])


ff_2.add_term(fv.Term("dry", fv.z_func, 20,55))
ff_2.add_term(fv.Term("wet", fv.s_func, 50, 90))
ff_2.add_term(fv.Term("normal", fv.bell_func, 10,20,50))
print([i for i in ff_2.get_val_memberships(input[1])])

ff_3.add_term(fv.Term("dry1", fv.z_func, 20,55))
ff_3.add_term(fv.Term("wet1", fv.s_func, 50, 90))
ff_3.add_term(fv.Term("normal1", fv.bell_func, 10,20,50))
print([i for i in ff_3.get_val_memberships(input[2])])

fs = fv.FIS()
fs.add_input_value(ff)
fs.add_input_value(ff_2)
fs.add_input_value(ff_3)
a = fs.calc_centroid(*input)
print("centroid: {}".format(a))

s = fs
#s = fv.FIS.load_fis_from_file("FIS.fis")
print(f"centroid for fis:{s.calc_centroid(10,20,40)}")

in_dataset = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]
out_data = [0, 1, 1, 0]


anfis = fv.ANFIS_HL(6, s)
anfis.init_system()
anfis.train_anfis(10, in_dataset, out_data)
anfis.calc_after_train([0, 0, 0])
anfis.train_anfis(100000, in_dataset, out_data)
anfis.calc_after_train([0, 0, 0])
# anfis.train_system(10,[[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]],[[0, 1, 1, 0]])
# anfis.calc_after_train([[1, 1, 0]])
# anfis.train_system(100,[[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]],[[0, 1, 1, 0]])
# anfis.calc_after_train([[1,1,0]])

# plt.plot(x,[A.fuzzification(i) for i in x], 'tab:green')
# plt.plot(x,[B.fuzzification(i) for i in x], 'tab:blue')
# plt.plot(x,[C.fuzzification(i) for i in x], 'tab:red')
# # plt.plot(x, [fv.Term("dry", fv.z_func, 20,55).fuzzification(i) for i in x])
# # plt.plot(x, [fv.Term("wet", fv.s_func, 50, 90).fuzzification(i) for i in x])
# plt.grid(True)
# plt.show()
