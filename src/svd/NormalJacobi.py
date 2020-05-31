import time
'''
    Author : Shashwat Sanket
    Description: This Script is the implementation of the QR Decomposition using (Householder decompositon method)
    Data of Code : 8 oct 2019
    PC: Sanket
'''

import os
import sys
import numpy as np
from QRDecomposition import QRDecomposition as QR

# Reading the filename as CLI argument
argvs = sys.argv
print(argvs)
if(len(argvs) != 2):
    sys.exit("Insufficient Arguments ")
# Get the current working directory
cwd = os.getcwd()
# Reading matrix A(filename: argv[0])
matA_file = os.path.join(cwd, sys.argv[1])
# Processing the input file as each line(, as  deliminator) of the file reperesents the row of the matrix
# Using the numpy library
# matrix = np.loadtxt(matA_file, delimiter=",")
# Getting the rows and columns

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def NormalJacobi(matrix):
    ep = 1.e-1
    iterations = 0
    U = matrix.copy()
    n = matrix.shape[1]
    V = np.identity(n)
    converge = ep+1
    while converge > ep:
        converge = 0
        for j in range(1, n):
            for i in range(j):
                # Computing alpha,beta,gamma
                #print(U[:, i])
                alpha = np.dot(U[:, i], U[:, i])
                beta = np.dot(U[:, j], U[:, j])
                # print(beta)
                gamma = np.dot(U[:, j], U[:, i])
                #print(alpha, beta, gamma)
                converge = max(converge, abs(gamma)/np.sqrt(alpha*beta))
                # print(converge)
                eta = (beta-alpha)/(2*gamma)
                # print(eta)
                t = np.sign(eta)/(abs(eta)+np.sqrt(1+eta**2))
                # print(t)
                c = 1/np.sqrt(1 + t*t)
                # print(c)
                s = c*t
                # print(s)
                # Updating the columns i and j of U
                t = U[:, i].copy()
                U[:, i] = c*t - s*U[:, j]
                # print(U)
                U[:, j] = s*t + c*U[:, j]
                # print(U)
                # Updating the columns i and j of V
                t = V[:, i].copy()
                V[:, i] = c*t - s*V[:, j]
                V[:, j] = s*t + c*V[:, j]
                iterations += 1
    ans = np.zeros(n)
    for i in range(n):
        norm = np.linalg.norm(U[:, i])
        ans[i] = norm
        U[:, i] = U[:, i]/norm
    #S = ans
    return [U, ans, V, iterations]


# st = time.time()
# t = NormalJacobi(matrix)
# stop = time.time()
# print(stop-st)
# print(NormalJacobi(matrix))
