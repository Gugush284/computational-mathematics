import numpy as np

def explicitMethod():
    a = 1
    print(a)

def main():
    T = 10 - 0 # range
    N = 10000 # amount of steps
    t = T/N # step
    explicitMethod() # explicit Euler method
    #implicitMethod()

main()