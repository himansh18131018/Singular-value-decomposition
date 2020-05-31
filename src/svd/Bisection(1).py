import os
import sys
import time
import numpy as np

np.set_printoptions(suppress=True, precision=4)

B = np.array([[ -9.8489,14.6164,0,0,0],[0,3.5309,5.3093,0,0],
              [  0,0,-3.2272,3.2113,0],
             [0,0,0,3.224,0.5598],
             [0,0,0,-2.3999,1.2292]]);
matrix = B
st = time.time()
B = np.linalg.norm(matrix)

def Negcount(n,B,u):
    t = -u
    count=0
    for k in range(1,n-1):
        d = (B[k][k]**2) + t
        if d<0:
            count = count + 1
        else:
            t = t*(((B[k][k+1])**2)/d) - u

    d = (B[n-1][n-1])**2 + t
    if d<0:
        count = count + 1

    return count

#n = int(input("Enter number of columns and rows:"))
#for i in range(4):
 #   for j in range(4):
  #      x = int(input("Enter the elements"))
   #     matrix.append(x)
n=5
a = 0
b = 20
n_a = Negcount(n,matrix,a)
n_b = Negcount(n,matrix,b)
l = []
w = []
tol = 0.0000001
if n_a==n_b:
    exit(0)
else:
    l.append(a)
    l.append(n_a)
    l.append(b)
    l.append(n_b)
    w=B
    while len(l)!=0:
        n_up = l.pop()
        up = l.pop()
        n_low = l.pop()
        low = l.pop()
        mid = (low+up)/2
        if(up-low<tol):
            for i in range(n_low+1,n_up):
                w[i-n_a] = mid
        else:
            n_mid = Negcount(n,matrix,mid)
            if n_mid>n_low:
                l.append(low)
                l.append(n_low)
                l.append(mid)
                l.append(n_mid)
            if n_up>n_mid:
                l.append(mid)
                l.append(n_mid)
                l.append(up)
                l.append(n_up)
sp = time.time()
#
print("The time for bisection is:\n",sp-st)
print(w)
