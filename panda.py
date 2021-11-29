import pandas as pnd
import datetime as dt
import fuzzy_val as fv

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

ncpm_max, ndpm, rep_min, rep_max, tvph_max, pop_dens_min, pop_dens_max = 0,0,100000000,0,0,100000000,0
for i in dates:
    for j in dates[i]:
        if dates[i][j][1] > ncpm_max: ncpm_max = dates[i][j][1]
        if dates[i][j][3] > ndpm: ndpm = dates[i][j][3]
        if dates[i][j][4] > rep_max: rep_max = dates[i][j][4]
        if dates[i][j][5] > tvph_max: tvph_max = dates[i][j][5]
        if dates[i][j][7] < pop_dens_min: pop_dens_min = dates[i][j][7]
        if dates[i][j][7] > pop_dens_max: pop_dens_max = dates[i][j][7]

print([ncpm_max, ndpm, rep_min, rep_max, tvph_max, pop_dens_min, pop_dens_max])

train_set = []
for i in dates:
    if i < dt.date(2021,9,30):
        for j in dates[i]:
            train_set.append([i, dates[i][j], dates[i + dt.timedelta(days=1)][j]])



