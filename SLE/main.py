import numpy as np    # Импорт бибилиотеки numpy
import matplotlib.pyplot as plt   # Импорт модуля matplotlib.pyplot

def printMatrix(matrix):
    for elem in matrix:
        print(elem)

def staffing(size):
    A = np.eye(size)  # Создание единичной матрицы

    for i in range(0, size):
        A[0][i] = 1

    for i in range(1,size-1):
        A[i][i] = 10
        A[i][i-1] = 1
        A[i][i+1] = 1

    A[size-1][size-2] = 1

    b = np.full((size,1), 1)

    i =  0
    for str in b:
        i += 1
        str[0] = i

    x = np.full((size,1), 0)

    return A, b, x

def gauss(A, b, x):
    return A, b, x
      
def main():
    size = 5

    A, b, x = staffing(size)

    gauss(A, b, x)

    print("Матрица коэффициентов")
    printMatrix(A)
    print("Вектор свободных членов")
    printMatrix(b)
    print("Вектор-решение")
    printMatrix(x) 

main()