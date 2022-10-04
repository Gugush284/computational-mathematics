import numpy as np    # Импорт бибилиотеки numpy
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

def LDUseparation(A, size):
    U = np.full((size, size), 0.0)
    LD = A

    for i in range(0, size):
        for j in range(i+1, size):
            U[i][j] = LD[i][j]
            LD[i][j] = 0

    return LD, U

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

def seidel(Ar, br, size):
    A = copy.deepcopy(Ar)
    b = copy.deepcopy(br)
    x = np.full(size, 0.0)

    LD, U = LDUseparation(A, size)

    printMatrix(LD)
    print("U")
    printMatrix(U)

    return x
      
def main():
    size = 3

    A, b = staffing(size)

    x_gauss = gauss(A, b, size)
    x_seidl = seidel(A, b, size)

    print("Матрица коэффициентов")
    printMatrix(A)
    print("Вектор свободных членов")
    printMatrix(b)
    print("Вектор-решение по Гауссу")
    printMatrix(x_gauss)
    print("Вектор-решение по Зейделю")
    printMatrix(x_seidl)

main()