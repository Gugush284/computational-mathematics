# x^2 - exp(x)/5 = 0

import numpy as np


def simp_iter(start):
    converge = True
    x = np.copy(start)

    while converge:
        x = np.sqrt(np.exp([x])[0]/5)

        converge = (np.abs(x ** 2 - np.exp([x])[0]/5) > 0.001)

    return x

def newton(start):
    x  = np.copy(start)
    converge = True

    while converge:
        x = x - (x ** 2 - np.exp([x])[0]/5)/(2 * x - np.exp([x])[0]/5)

        converge = (np.abs(x ** 2 - np.exp([x])[0]/5) > 0.001)

    return x

def main():
    x = 4.1

    x_simp_iter = simp_iter(x)

    x_newton = newton(x)

    print(x_simp_iter)
    print(x_newton)

main()