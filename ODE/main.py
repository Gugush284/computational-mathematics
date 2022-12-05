import numpy as np

def rightFunction(x, y):
    return -100*y

def function(x):
    return np.exp(-100*x)

def directEuler(n, h, x, y):
    for i in range(n):
        # print(x, y)
        y += h * rightFunction(x, y)
        x += h
    return x, y # решение

def main(n, start, stop):
    h = (stop - start)/n
    print(h)

    xSDE, ySDE = directEuler(n, h, start, function(start))
    print("Solve of direct Euler")
    print(xSDE)
    print(ySDE)
    print(np.abs(ySDE-function(1)))


    
main(8000000, 0, 1)
