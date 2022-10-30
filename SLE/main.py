import copy

import numpy as np


def printMatrix(matrix): # Печать 
    for elem in matrix:
        print(elem)

def printResult(*nums):
    for i in range(len(nums[0])):
        for n in nums:
            print(n[i], end="\t")
        print(end="\n")

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

def gauss(Ar, br, solve):
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

    r = br - Ar.dot(x)

    return x, r

def normV(v):
    n = abs(v[0])
    for i in v:
        if abs(i) > n:
            n = abs(i)

    return n

def seidel(A, b, solve):
    size = len(A)
    x = np.zeros(size)

    print("Зейдель - итерации")
    converge = True
    amIter = 0
    while converge:
        x_new = np.copy(x)
        for i in range(size):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, size))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        converge = (np.sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(size))) > np.finfo(np.float64).eps)

        print(normV(x_new-solve))

        x = x_new
        amIter += 1
        
    r = b - A.dot(x)

    return x, r, amIter

def eigenvalue(A):
    eigenvalues, _ = np.linalg.eig(A)
    if len(eigenvalues) == 1:
        return eigenvalues, eigenvalues
    else:
        max = eigenvalues[0]
        min = eigenvalues[0]
        for value in eigenvalues:
            if value > max:
                max = value
            if value < min:
                min = value
        return min, max

def normM(A):
    n = 0
    for i in range(len(A)):
        s = sum(abs(A[i][k]) for k in range(len(A)))
        if n < s:
            n = s

    return n

def conditionality(A):
    invA = np.linalg.inv(A)
    return (normM(A)*normM(invA))

def LDUseparation(A):

    size = len(A)

    U = np.full((size, size), 0.0)
    LD = np.copy(A)

    for i in range(0, size):
        for j in range(i+1, size):
            U[i][j] = LD[i][j]
            LD[i][j] = 0

    return LD, U

def inaccuracy(Matrix, solve, seid, amIter):
    actual = normV(seid-solve)

    LD, U = LDUseparation(Matrix)

    print(normM((np.linalg.inv(LD)).dot(U)))
    print(normV(np.zeros(len(solve))-solve))

    

      
def main():
    size = 100

    A, b = staffing(size)

    print("Матрица коэффициентов")
    printMatrix(A)
    print("\nВектор свободных членов")
    printMatrix(b)

    x = np.linalg.solve(A, b)
    x_gauss, r_gauss = gauss(A, b, x)
    x_seid, r_seid, amIter = seidel(A, b, x)
    lambda_min, lambda_max = eigenvalue(A)

    print("\nЭталонное решение\tВектор-решение по Гауссу\tВектор-решение по Зейделю")
    printResult(x, x_gauss, x_seid)

    print("\nНевязка по Гауссу\tНевязка по Зейделю")
    printResult(r_gauss, r_seid)

    print("\nМинимальное собственное значение: ", lambda_min)
    print("Максимальное собственное значение: ", lambda_max)
    print("Эталонное число обусловленности:", np.linalg.cond(A, np.inf))
    print("Число обусловленности", conditionality(A), "\n")

    inaccuracy(A, x, x_seid, amIter)

if __name__ == "__main__":
    main()