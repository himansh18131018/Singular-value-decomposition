'''
    Author : Shashwat Sanket
    Description: This Script is the implementation of the HouseHolder Decomposition
    Data of Code : 7 oct 2019
    PC: Sanket
'''

import os
import sys
import numpy as np
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
matrix = np.loadtxt(matA_file, delimiter=",")
# Getting the rows and columns
rows, cols = matrix.shape

# Initialisations
U = None  # Left Reflection Matrix mxn
V = None  # Right Reflection of the matrix
B = matrix
init = 0  # Discarding the first row
###################

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# Decomposition to Tridiagonal Form (B)
for i in range(cols):
    u = B[:, i]
    s = np.linalg.norm(u[i:])
    if(i >= rows):
        break
    y = np.concatenate((u[:i], np.array([s])), axis=None)
    # Boundry condition in the case of mxn n>m
    y = np.concatenate((y, np.zeros(u.size-y.size)))
    W = u - y
    norm_W = np.linalg.norm(W) if np.linalg.norm(W) != 0 else 1
    w = W/norm_W
    # offset = 2*W*W'
    size = w.size
    offset = 2*np.matmul(w.reshape(size, 1), w.reshape(1, size))
    Q = np.identity(size) - offset
    B = np.matmul(Q, B)
    U = np.matmul(U, Q) if init != 0 else Q
    if(i <= (cols-2)):
        u = B[i]
        s = np.linalg.norm(u[i+1:])
        y = np.concatenate((u[:i+1], np.array([s])), axis=None)
        y = np.concatenate((y, np.zeros(abs(u.size-y.size))))
        W = u - y
        norm_W = np.linalg.norm(W) if np.linalg.norm(W) != 0 else 1
        w = W/norm_W
        size = w.size
        offset = 2*np.matmul(w.reshape(size, 1), w.reshape(1, size))
        P_T = np.identity(size) - offset
        P = P_T.transpose()
        B = np.matmul(B, P)
        V = np.matmul(P, V) if init != 0 else P
    init = 1
    ###########
print(B)
print(U)
print(V)

# print(np.linalg.det(B))
# print(np.linalg.det(matrix))
###########
