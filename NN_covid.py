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
        ]

train_set = []
for i in dates:
    if i < dt.date(2021,9,30):
        for j in dates[i]:
            train_set.append([i, dates[i][j], dates[i + dt.timedelta(days=1)][j]])

#X oordinates for plots
cases_x = np.arange(0, 60, 1)
death_cases_x = np.arange(0, 10, 0.1)
reproduction_x = np.arange(-11, 40, 0.5)
population_x = np.arange(0, 600000000, 10000)
vaccine_x = np.arange(0, 100, 1)
pop_density_x = np.arange(0, 400, 1)
medium_age_x = np.arange(0,100,1)
cv_death_x = np.arange(0, 600, 1)

#Cases fuzzy variable
cases_small = fv.Term("Small cases", fv.z_func, 1, 15)
cases_medium = fv.Term("Medium cases", fv.bell_func , 25, 15, 25)
cases_most = fv.Term("Most cases", fv.s_func, 32, 50 )

cases = fv.FuzzyVar()
cases.add_term(cases_small)
cases.add_term(cases_medium)
cases.add_term(cases_small)

#Death cases variable
death_cases_small = fv.Term("Small death cases", fv.z_func, 0, 2)
death_cases_medium = fv.Term("Medium death cases", fv.bell_func , 1.75, 2.5, 4)
death_cases_most = fv.Term("Most death cases", fv.s_func, 5, 10 )

death_cases = fv.FuzzyVar()
death_cases.add_term(death_cases_small)
death_cases.add_term(death_cases_medium)
death_cases.add_term(death_cases_small)

#Reproduction rate variable
reproduction_small = fv.Term("Small reproduction rate", fv.z_func, -10, 8)
reproduction_high = fv.Term("Big reproduction rate", fv.s_func, 5, 30)

reproduction = fv.FuzzyVar()
reproduction.add_term(reproduction_small)
reproduction.add_term(reproduction_high)

#Vaccinated per hundred
vaccine_small = fv.Term("Small vaccinations rate", fv.z_func, 0, 20)
vaccine_medium = fv.Term("Medium vaccinations rate", fv.bell_func, 34, 4.75, 50)
vaccine_large = fv.Term("Large vaccinations rate", fv.s_func, 55, 100)

vaccinations = fv.FuzzyVar()
vaccinations.add_term(vaccine_small)
vaccinations.add_term(vaccine_medium)
vaccinations.add_term(vaccine_large)

#Population variable
population_small = fv.Term("Small population", fv.z_func, 1, 400000000)
population_most = fv.Term("Large population", fv.s_func, 300000000, 500000000 )

population = fv.FuzzyVar()
population.add_term(population_small)
population.add_term(population_most)

#Population density variable
population_density_var = fv.Term("Population density", fv.s_func, 0, 250)

pop_density = fv.FuzzyVar()
pop_density.add_term(population_density_var)

#medium age
medium_age_var = fv.Term("Medium age", fv.s_func, 20, 60)

medium_age = fv.FuzzyVar()
medium_age.add_term(medium_age_var)

#Cardio-vascular death rate
cv_death_var = fv.Term("Cardio-vascular death rate", fv.s_func, 0, 400)

cv_death = fv.FuzzyVar()
cv_death.add_term(cv_death_var)

#Cases output data
cases_answer_term = fv.Term("Cases", fv.s_func, 1, 60)
corelation_answer = [cases_answer_term.fuzzification(x) for x in cases_x]
# plt.plot(cases_x, corelation_answer, 'tab:blue')
cases_answer = fv.FuzzyVar()
cases_answer.add_term(cases_answer_term)

#Make FIS system for input
fis_input = fv.FIS()
fis_input.add_input_value(cases)
fis_input.add_input_value(death_cases)
fis_input.add_input_value(reproduction)
fis_input.add_input_value(vaccinations)
fis_input.add_input_value(population)
fis_input.add_input_value(pop_density)
fis_input.add_input_value(medium_age)
fis_input.add_input_value(cv_death)

a = fis_input.calc_centroid(10,0,3,10,2,5,1,4)
nn_system = fv.FIONS(23, cases_answer, fis_input )
nn_system.init_system()
nn_system.train_fions(1,)
