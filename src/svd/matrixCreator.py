import numpy as np


def create_matrix(m, n):
    return np.random.random_sample((m, n))*10


np.save("matrix.npy", create_matrix(10000, 100))

x = np.load("matrix.npy")

print(x)
