import matplotlib.pyplot as plt
import numpy as nm
import math as m
import fuzzy_val as fv

x = nm.arange(-20,120,0.5)

A = fv.Term('Warm', fv.bell_func, 10,20,20)
B = fv.Term('Cold', fv.z_func, 10,15)
C = fv.Term('Hot', fv.s_func, 26,30)

ff = fv.FuzzyVar()
ff_2 = fv.FuzzyVar()

input = [20,50,10]

ff.add_term(A)
ff.add_term(B)
ff.add_term(C)
print([i for i in ff.get_val_memberships(input[0])])


ff_2.add_term(fv.Term("dry", fv.z_func, 20,55))
ff_2.add_term(fv.Term("wet", fv.s_func, 50, 90))
ff_2.add_term(fv.Term("normal", fv.bell_func, 10,20,50))
print([i for i in ff_2.get_val_memberships(input[1])])


fs = fv.FIS()
fs.add_input_value(ff)
fs.add_input_value(ff_2)
a = fs.calc_centroid(*input)
print("centroid: {}".format(a))

#s = fv.FIS.load_fis_from_file("FIS.fis")

s = fs
print(f"centroid for fis:{s.calc_centroid(10,20,40)}")

in_dataset = [[20, 50], [22, 54], [80,100], [15, 60], [10, 10]]
out_data = [[85], [92], [0], [24], [10]]
answer = [80,50]

out_var = fv.FuzzyVar()
out_var.add_term(fv.Term("comfort", fv.s_func, 0, 100))

anfis = fv.FIONS(13, out_var, fs)
anfis.init_system()

trainres = [out_var.get_val_memberships(i[0]) for i in out_data]
delta = []
i_delta = []
epoch = 500
for k in range(epoch):
    anfis.train_fions(1, in_dataset, out_data)
    for i in range(len(in_dataset)):
        l1 = anfis.calc_after_train(in_dataset[i])
        i_delta.append(abs(trainres[i][0][1] - l1[0][0])/(l1[0][0])*100)
    delta.append(i_delta.copy())
    i_delta.clear()

x_error = nm.arange(0, epoch, 1)
colors=['red','black','blue','orange','green']
plt.plot(x_error, [ sum(j)/float(len(j)) for j in delta], '-')

for i in range(len(in_dataset)):
    plt.plot(x_error, [j[i] for j in delta], colors[i], label=f'line {i}')
plt.grid(True)
plt.show()

l1 = anfis.calc_after_train(answer)
print("res {} ; delta {}".format(l1, trainres))
#anfis.train_fions(90, in_dataset, out_data)
l2 = anfis.calc_after_train(answer)
print(l2)
l3 = anfis.calc_after_train(answer)
print(l3)

# plt.plot(x, [fv.Term("comfort", fv.s_func, 0, 100).fuzzification(i) for i in x])
# plt.plot(x, [l1[0][0] for i in x], 'tab:green')
# plt.plot(x, [l2[0][0] for i in x], 'tab:orange')
# plt.plot(x, [l3[0][0] for i in x], 'tab:red')
