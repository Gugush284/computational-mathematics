import numpy as np                                                              # Импорт бибилиотеки numpy
import matplotlib.pyplot as plt                                                 # Импорт модуля matplotlib.pyplot

def gorner(x):                                                                  # Функция подсчета методом Горнера
    coeff = np.array([-18, 144, -672, 2016, -4032, 5376, -4608, 2304, -512])
    p = 1                                                                       # Присваеваем первый коэффициент полиному

    for coeffi in coeff:                                                        # Алгоритм Горнера
        p = x*p + coeffi                                                    

    return p


coeff = np.array([1, -18, 144, -672, 2016, -4032, 5376, -4608, 2304, -512])     # Коэффициенты уравнения (x-2)^9

x = 2                                                                           # Начальное значение на отрезке                    
inc_list = []                                                                   # Список погрешности по формуле из отчета
x_list = []                                                                     # Список значений отрезка [1.92, 2.08] с шагом 10^(-4)
gorner_list = []                                                                # Список, содержащий значения по Горнеру

while x <= 2.08:                                                                 # Идем по отрезку с шагом
    inac = 0
    for i in range(0, 10):                                                      # Делаем оценку погрешности
        inac += abs(coeff[i] * x ** (9-i))
    inac = inac * 9 * np.finfo(np.float64).eps                                  # np.finfo(np.float64).eps - эпсилон машинное для типа float64 из numpy

    g = gorner(x)                                                               # Считаем по Горнеру

    print(g, inac)                                                              # Выводим значения
    inc_list.append(inac)                                                       # и заполняем списки
    x_list.append(x)
    gorner_list.append(g)


    x += 10 ** -4                                                               # Делаем шаг


plt.figure(figsize=[12, 4])                                                     # Создаем полотно
plt.plot(x_list, inc_list)                                                      # и строим графики
plt.plot(x_list, gorner_list)                                                                                                 
plt.show()                                                                  