import math as m
import matplotlib.pyplot as plt
import numpy as nump
 
#Константа
E = nump.e
 
#Функція тотожності
def eqFunc(x):
    return x
#Сигмоїдальна функція
def sigmoidFunc(x):
    return 1/(1+E**(-x))
#"Двійковий крок"
def binStepFunc(x):
    if(len(x) > 1):
        x[x > 0] = 1
        x[x != 1] = 0
        return x
    else: 
        return 0 if x < 0 else 1
#Гаусіанська функція
def gaussianFunc(x):
    return E**(-x**2)
#Гіперболічний тангенс
def tanHFunc(x):
    return (E**x-E**(-x))/(E**x+E**(-x))
 
x = nump.arange(-10, 10, 0.01) 
# plt.plot(x, eqFunc(x), 'tab:blue') #Синій
# plt.plot(x, [sigmoidFunc(z) for z in x], 'tab:green') #Зелений
# plt.plot(x, [binStepFunc(z) for z in x], 'tab:red') #Червоний
# plt.plot(x,[gaussianFunc(z) for z in x], 'tab:orange') #Помаранчовий
plt.plot(x, [tanHFunc(z) for z in x], 'tab:cyan') #Блакитний
plt.grid(True)
plt.show()
 
# Навчальна вибірка
X = nump.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]) #Вхідні дані
y = nump.array([[0, 1, 1, 0]]).T #Відповіді
 
nump.random.seed(1)
 
#Кількість нейронів прихованого шару
hidden_neural = 5
 
#Матриці синапсів
s_hidden = 2 * nump.random.random((len(X.T),hidden_neural))
s_output = 2 * nump.random.random((hidden_neural,len(y.T)))
 
# Кількіст епох
epoch = 200
 
#Змінна для перегляду навчання
output = [[],[],[]]
 
#Навчання системи
for j in range(epoch):
    #Вхідний шар
    l0 = X 
    #Синапсиси
    l1 = tanHFunc(l0.dot(s_hidden))
    l2 = tanHFunc(l1.dot(s_output))
    #Вираховування помилки
    l2_delta = (y - l2) * (l2 * (1 - l2))
    l1_delta = l2_delta.dot(s_output.T) * (l1 * (1 - l1))
    # корегування ваг для шарів
    s_output += l1.T.dot(l2_delta)
    s_hidden += l0.T.dot(l1_delta)
    #Заповнення графіку
    output[0].append(tanHFunc(tanHFunc(l0.dot(s_hidden)).dot(s_output))[0])
    output[1].append(tanHFunc(tanHFunc(l0.dot(s_hidden)).dot(s_output))[1])
    output[2].append(tanHFunc(tanHFunc(l0.dot(s_hidden)).dot(s_output))[2])
 
print('HIDDEN LAYER')
print(l1)
print('OUT LAYER')
print(l2)
print('HIDDEN LAYER WEIGHT')
print(s_hidden)
print('OUT LAYER WEIGHT')
print(s_output)
 
#Перевірка роботи нейронної мережі
#Вхідні дані
question = nump.array([[1,0,0],
                        [0,1,0]])
 
#Ідеальні дані
ideal = nump.array([[1, 1]]).T
 
print('QUESTION')
print(question)
#Розрахунок відповіді системою
answer = sigmoidFunc(tanHFunc(question.dot(s_hidden)).dot(s_output))
print("ANSWER:")
print(answer)
print('IDEAL:')
print(ideal)
#Розрахунок похибки
print('ERROR:')
print((nump.divide(ideal - answer, ideal) * 100))
 
#Рисуєм графік
r = range(epoch)
plt.plot(r, output[0], 'tab:red')
plt.plot(r, output[1], 'tab:cyan')
plt.plot(r, output[2], 'tab:green')
plt.grid(True)
plt.show()
