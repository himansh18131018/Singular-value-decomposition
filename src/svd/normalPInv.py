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

start = time.time()
pinv = np.linalg.pinv(matrix)
end = time.time()

print("Total time taken ", (end-start))
