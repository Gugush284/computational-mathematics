import numpy as np    # Импорт бибилиотеки numpy
import copy

def printMatrix(matrix):
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

    r = br - Ar.dot(x)

    return x, r

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
        
    r = b - A.dot(x)

    return x, r

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

def norm(A):
    n = 0
    for i in range(len(A)):
        s = sum(abs(A[i][k]) for k in range(len(A)))
        if n < s:
            n = s

    return n

def conditionality(A):
    invA = np.linalg.inv(A)
    return (norm(A)*norm(invA))
      
def main():
    size = 100

    A, b = staffing(size)

    print("Матрица коэффициентов")
    printMatrix(A)
    print("\nВектор свободных членов")
    printMatrix(b)

    x_gauss, r_gauss = gauss(A, b)
    x_seid, r_seid = seidel(A, b)
    lambda_min, lambda_max = eigenvalue(A)

    print("\nЭталонное решение\tВектор-решение по Гауссу\tВектор-решение по Зейделю")
    printResult(np.linalg.solve(A, b), x_gauss, x_seid)

    print("\nНевязка по Гауссу\tНевязка по Зейделю")
    printResult(r_gauss, r_seid)

    print("\nМинимальное собственное значение: ", lambda_min)
    print("Максимальное собственное значение: ", lambda_max)
    print("Эталонное число обусловленности:", np.linalg.cond(A, np.inf))
    print("Число обусловленности", conditionality(A), "\n")

main()