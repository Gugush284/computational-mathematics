import struct
import numpy as np

def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

x = np.float64(0.15625)
print(binary(x), type(x))