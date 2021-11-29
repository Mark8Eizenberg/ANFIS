import pandas as pnd
import datetime as dt
import fuzzy_val as fv
import numpy as np
import matplotlib.pyplot as plt

#read dataset from CSV file
covid = pnd.read_csv("covid_data.csv")


#make date depence list
dates = {}
for i in covid['date']:
    if not dt.date.fromisoformat(i) in dates and dt.date.fromisoformat(i) < dt.date(2021,10,1):
        dates[dt.date.fromisoformat(i)] = {}

covid.fillna(0, inplace=True) #fill all nan by zero

for i in range(1, len(covid)):
    #print(covid['date'][i])
    if covid['location'][i] in ['Ukraine', 'Japan', 'Italy', 'France', 'China', 'United States', 'United Kingdom', 'Sweden'] and dt.date.fromisoformat(covid['date'][i]) <  dt.date(2021,10,1):
        dates[dt.date.fromisoformat(covid['date'][i])][covid['location'][i]] = [
            covid['total_cases_per_million'][i], #0
            covid['new_cases_per_million'][i], #1
            covid['total_deaths_per_million'][i], #2
            covid['new_deaths_per_million'][i], #3
            covid['reproduction_rate'][i], #4
            covid['total_vaccinations_per_hundred'][i], #5
            covid['population'][i], #6
            covid['population_density'][i], #7
            covid['median_age'][i], #8
            covid['cardiovasc_death_rate'][i], #9
            covid['diabetes_prevalence'][i], #10
            covid['female_smokers'][i], #11
            covid['male_smokers'][i], #12
            covid['hospital_beds_per_thousand'][i] #13
        ]

# ncpm_max, ndpm, rep_min, rep_max, tvph_max, pop_dens_min, pop_dens_max = 0,0,100000000,0,0,100000000,0
# for i in dates:
#     for j in dates[i]:
#         if dates[i][j][1] > ncpm_max: ncpm_max = dates[i][j][1]
#         if dates[i][j][3] > ndpm: ndpm = dates[i][j][3]
#         if dates[i][j][4] > rep_max: rep_max = dates[i][j][4]
#         if dates[i][j][5] > tvph_max: tvph_max = dates[i][j][5]
#         if dates[i][j][7] < pop_dens_min: pop_dens_min = dates[i][j][7]
#         if dates[i][j][7] > pop_dens_max: pop_dens_max = dates[i][j][7]

# print([ncpm_max, ndpm, rep_min, rep_max, tvph_max, pop_dens_min, pop_dens_max])

train_set = []
for i in dates:
    if i < dt.date(2021,9,30):
        for j in dates[i]:
            train_set.append([i, dates[i][j], dates[i + dt.timedelta(days=1)][j]])

cases_x = np.arange(0, 60, 1)

cases_small = fv.Term("Small cases", fv.z_func, 1, 15)
cases_medium = fv.Term("Medium cases", fv.bell_func , 25, 15, 25)
cases_most = fv.Term("Most cases", fv.s_func, 32, 50 )

cases = fv.FuzzyVar()
cases.add_term(cases_small)
cases.add_term(cases_medium)
cases.add_term(cases_small)

population_x = np.arange(0, 1000000000, 100)

population_small = fv.Term("Small population", fv.z_func, 1, 10000000)
population_medium = fv.Term("Medium population", fv.bell_func , 500000000, 1600, 400000000)
population_most = fv.Term("Large population", fv.s_func, 700000000, 2000000000 )

population = fv.FuzzyVar()
population.add_term(population_small)
population.add_term(population_medium)
population.add_term(population_most)
# plt.plot(cases_x, [cases_small.fuzzification(x) for x in cases_x], 'tab:green')
# plt.plot(cases_x, [cases_medium.fuzzification(x) for x in cases_x], 'tab:orange')
# plt.plot(cases_x, [cases_most.fuzzification(x) for x in cases_x], 'tab:red')
# plt.grid(True)
# plt.show()

cases_answer_term = fv.Term("Cases", fv.s_func, 1, 60)

corelation_answer = [cases_answer_term.fuzzification(x) for x in cases_x]

plt.plot(cases_x, corelation_answer, 'tab:blue')

cases_answer = fv.FuzzyVar()
cases_answer.add_term(cases_answer_term)

fis = fv.FIS()
fis.add_input_value(cases)
fis.add_input_value(population)

n = fv.FIONS(15, cases_answer, fis)

input_data = [[y[1][1], y[1][6] ] for y in train_set]
train_data = [[y[2][1]] for y in train_set]

n.init_system()
n.train_fions(10, input_data, train_data)
n1 = n.calc_after_train([15, 40000000])
n.train_fions(100, input_data, train_data)
n2 = n.calc_after_train([15, 40000000])
n.train_fions(100, input_data, train_data)
n3 = n.calc_after_train([15, 40000000])
print([n1, n2, n3])

tmp = n3
for i in range(10):
    tmp =  n.calc_after_train([tmp, 40000000])
    print("Answer #{} = {}".format(i, tmp[0][0]))
plt.grid(True)
plt.show()


