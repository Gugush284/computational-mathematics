import numpy as np                  # импорт бибилиотеки numpy
import matplotlib.pyplot as plt     # импорт модуля matplotlib.pyplot

def gorner(x, coeff):
    p = 1

    for coeffi in coeff:
        p = x*p + coeffi

    return p



x = 1.92
coeff = np.array([-18, 144, -672, 2016, -4032, 5376, -4608, 2304, -512])
gorner_list = []
eq_list = []
x_list = []

while x < 2.08:
    p  = gorner(x, coeff)
    eq = (x-2)**9

    print(x, p, eq)

    eq_list.append(eq)
    gorner_list.append(p)
    x_list.append(x)

    x += 10 ** -4

plt.figure(figsize=[12, 4]) 
plt.plot(x_list, gorner_list)
plt.plot(x_list, eq_list)
plt.show()