import numpy as np    # Импорт бибилиотеки numpy
import matplotlib.pyplot as plt   # Импорт модуля matplotlib.pyplot
import copy

def printMatrix(matrix):
    for elem in matrix:
        print(elem)

def staffing(size):
    A = np.eye(size)  # Создание единичной матрицы

    for i in range(0, size):
        A[0][i] = 1.0

    for i in range(1,size-1):
        A[i][i] = 10.0
        A[i][i-1] = 1.0
        A[i][i+1] = 1.0

    A[size-1][size-2] = 1.0

    b = np.full(size, 1.0)

    k = size
    for i in range(0, size):
        b[i] = k
        k -= 1

    return A, b

def gauss(Ar, br, size):
    A = copy.deepcopy(Ar)
    b = copy.deepcopy(br)
    x = np.full(size, 0.0)

    for k in range(0, size-1):
        denominator = A[k][k]

        for i in range(k+1, size):
            frac = A[i][k] / denominator 

            for j in range(k, size):
                A[i][j] -= A[k][j] * frac
            b[i] -= frac * b[k]

    for k in range(size-1, -1, -1):
        x[k] = b[k]

        for i in range(k+1, size):
            x[k] -= A[k][i] * x[i]

        x[k] = x[k]/A[k][k]

    return x
      
def main():
    size = 4

    A, b = staffing(size)

    x = gauss(A, b, size)

    print("Матрица коэффициентов")
    printMatrix(A)
    print("Вектор свободных членов")
    printMatrix(b)
    print("Вектор-решение")
    printMatrix(x) 

main()