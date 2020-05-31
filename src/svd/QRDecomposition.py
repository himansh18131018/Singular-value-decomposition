'''
    Author : Shashwat Sanket
    Description: This Script is the implementation of the QR Decomposition using (Householder decompositon method)
    Data of Code : 8 oct 2019
    PC: Sanket
'''

import os
import sys
import numpy as np

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


def QRDecomposition(matrix):
    rows, cols = matrix.shape
    A = matrix.copy()
    red = True
    Q = np.identity(rows)
    elmtCount = 0
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    while(red):
        A = A[elmtCount:, elmtCount:]
        # Stop the reduction process
        if(A.shape[0] <= 0 or A.shape[1] <= 0 or A.size <= 1):
            break
        # Check if the matrix((1,1)) here vector is upper diagonal or not
        if(A.shape[0] == 1 or A.shape[1] == 1):
            # Checking if upper matrix
            temp = A.tolist()
            if temp[1][0] == 0:
                break
        a = A[:, 0]
        norm_a = np.linalg.norm(a)
        sign = 1 if a[0] >= 0 else -1  # Sign of first element of the vector
        e = np.concatenate((1, np.zeros(a.size - 1)), axis=None)
        v = a - sign*norm_a*e
        row_v = np.reshape(v, (v.size, 1))
        offset = np.matmul(row_v, row_v.transpose())
        dotProduct = 1 if np.dot(v, v) == 0 else np.dot(v, v)
        I = np.identity(v.size)
        H = I - (2/dotProduct)*offset
        A = np.matmul(H, A)
        # Shaping H to shape as that of A
        shapeMatrix = np.identity(rows)
        shapeMatrix[(rows - H.shape[0]):, (rows - H.shape[0]):] = H
        H = shapeMatrix
        Q = np.matmul(H, Q)
        # Will be replaced by current status of A here
        # Checking whether A is Upper diagonal with the case when A(1,1)
        elmtCount = 1

    R = np.matmul(Q, matrix)
    return [Q, R]


# print(QRDecomposition(matrix))
