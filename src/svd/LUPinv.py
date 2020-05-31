import time
import os
import sys
import numpy as np
import scipy.linalg as la

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

matrix = np.load(matA_file)


# def recur(matrix):

#     return

def LU(matrix):
    (P, L, U) = la.lu(matrix)
    return [L, U]


def LUmxn(matrix, r, k):
    if k == 0:
        return matrix
    A00 = matrix[:r, :r]
    A01 = matrix[:r, r:]
    A10 = matrix[r:, :r]
    A11 = matrix[r:, r:]
    L00, U00 = LU(A00)
    retMat = np.zeros(matrix.shape)
    L10 = np.dot(A10, np.linalg.inv(U00))
    if((A01.shape[0] != 0) and (A01.shape[1] != 0)):
        U01 = np.dot(A01, np.linalg.inv(L00))
        off = np.dot(L10, U01)
        temp = A11 - off
        retMat[:r, r:] = U01
        retMat[r:, r:] = LUmxn(temp, r, k-1)
    else:
        A01 = 0
        U01 = 0
    block1 = U00 + L00
    #block2 = L10
    #block3 = U01
    retMat[:r, :r] = block1
    retMat[r:, :r] = L10

    return retMat


# print(np.dot(l, u))
start = time.time()

m, n = matrix.shape

r = 100
k = int(min(m/r, n/r))

retMat = LUmxn(matrix, r, k)
L, U = np.tril(retMat), np.triu(retMat)

#l, u = np.tril(m), np.triu(m)

pinv = np.dot(np.transpose(np.linalg.pinv(U)), np.linalg.pinv(L))


end = time.time()

print("Total time taken ", (end-start))
