import numpy as np                                                          # Импорт бибилиотеки numpy
import matplotlib.pyplot as plt                                             # Импорт модуля matplotlib.pyplot

def gorner(x, coeff):                                                       # Функция подсчета методом Горнера
    p = 1                                                                   # Присваеваем первый коэффициент полиному

    for coeffi in coeff:                                                    # Алгоритм Горнера
        p = x*p + coeffi                                                    

    return p                                                                    



x = 1.92                                                                    # Начальное значение на отрезке
coeff = np.array([-18, 144, -672, 2016, -4032, 5376, -4608, 2304, -512])    # Коэффициенты уравнения (x-2)^9, исключая первый
gorner_list = []                                                            # Список, содержащий значения по Горнеру
eq_list = []                                                                # Список значений, полученных решением (x-2)^9
x_list = []                                                                 # Список значений отрезка [1.92, 2.08] с шагом 10^(-4)

while x < 2.08:                                                             # Пробегаем по отрезку
    p = gorner(x, coeff)                                                    # Вызываем подсчет Горнером
    eq = (x-2)**9                                                           # Считаем значение подстановкой

    print(x, p, eq)                                                         # Вывод

    eq_list.append(eq)                                                      # Добавляем все посчитанные значения в списки
    gorner_list.append(p)                                                   
    x_list.append(x)                                                        

    x += 10 ** -4                                                           # Делаем шаг 

plt.figure(figsize=[12, 4])                                                 # Создаем полотно
plt.plot(x_list, gorner_list)                                               # Построение графика для Горнера
plt.plot(x_list, eq_list)                                                   # Построение точного графика
plt.show()                                                                  # Вывод графика