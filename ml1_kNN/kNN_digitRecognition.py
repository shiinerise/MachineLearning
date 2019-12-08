import os
import numpy as np
def img2vector(filename):
    vector = np.zeros((1, 1024))
    file = open(filename)
    for i in range(32):
        line = file.readline()
        for j in range(32):
            vector[0, 32*i+j] = int(line[j])
    return vector

if __name__ == '__main__':
    data = img2vector('testDigits/0_0.txt')
    print(data)