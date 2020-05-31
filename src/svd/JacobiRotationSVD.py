'''
    Author : Shashwat Sanket
    Description: This Script is the implementation of the QR Decomposition using (Householder decompositon method)
    Data of Code : 8 oct 2019
    PC: Sanket
'''
import time
import os
import sys
import numpy as np
# from QRDecomposition import QRDecomposition as QR
from NormalJacobi import NormalJacobi as jacobi_svd
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
# Loading numpy data

matrix = np.load(matA_file)


# Getting the rows and columns

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def MoorePenrose(U, svd, V):
    n = len(svd)
    sigma = np.zeros((n, n))
    np.fill_diagonal(sigma, 1/svd)
    At = np.dot(V, sigma)
    At = np.dot(At, np.transpose(U))
    return At


def getMatrix(U, svd, V):
    n = len(svd)
    sigma = np.zeros((n, n))
    np.fill_diagonal(sigma, svd)
    A = np.dot(U, sigma)
    print(np.dot(A, np.transpose(V)))


def JacobianRotationSVD(matrix):
    # Input mxn matrix
    # Output [U S V]
    # Finding the housholder decomposition of the Matrix
    # start = time.time()
    # numpy_svd = np.linalg.svd(matrix)
    # print(numpy_svd)
    # k = time.time()-start
    start = time.time()
    U, normal_svd, V, normal_itr = jacobi_svd(matrix)
    # m = time.time()-start
    # SVD after QR decomposition
    #_, matrix_r = QR(matrix)
    #_, qr_svd, _, qr_itr = jacobi_svd(matrix_r)
    end = time.time()
    print("Total Iteration taken for Normal Matrix", normal_itr)
    print("Total time take to calculate svd : ", (end-start))
    start = time.time()
    a = np.linalg.svd(matrix)
    print(a)
    end = time.time()
    #At = MoorePenrose(U, normal_svd, V)
    getMatrix(U, normal_svd, V)
    #print("Total time taken to calculate MoorePenrose: ", time.time()-start)
    print("Through inbuilt numpy algorithm :", (end-start))


JacobianRotationSVD(matrix)
