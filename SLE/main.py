import numpy as np    # Импорт бибилиотеки numpy
import copy

def printMatrix(matrix):
    for elem in matrix:
        print(elem)

def printResult(ref, gau, seidl):
    for i in range(len(ref)):
        print(
            ref[i],
            "\t",
            gau[i],
            "\t",
            seidl[i]
        )

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

def gauss(Ar, br):
    A = copy.deepcopy(Ar)
    b = copy.deepcopy(br)
    size = len(Ar)
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

def seidel(A, b):
    size = len(A)
    x = np.zeros(size)

    converge = True
    while converge:
        x_new = np.copy(x)
        for i in range(size):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, size))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        converge = (np.sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(size))) > np.finfo(np.float64).eps)
        x = x_new

    return x
      
def main():
    size = 100

    A, b = staffing(size)

    print(len(A))

    x_gauss = gauss(A, b)
    x_seid = seidel(A, b)

    print("Матрица коэффициентов")
    printMatrix(A)
    print("Вектор свободных членов")
    printMatrix(b)
    print("Эталонное решение\tВектор-решение по Гауссу\tВектор-решение по Зейделю")
    printResult(np.linalg.solve(A, b), x_gauss, x_seid)

main()