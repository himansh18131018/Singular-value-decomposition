import math
import numpy as np
import time
st = time.time()
matrix = [[-3., -2., 13., -4., 4.], [9., 5., 1., -2., 4.],
          [9., 1., -7.,  5., -1.], [2., -6., 6.,  5., 1.], [4., 5., 0., -2., 2.]]
#matrix = [[1,2],[3,4]]
#matrix = [[1.,2.],[1.,2.],[1.,2.],[1.,2.],[3.,4.]]
mat = np.load("matrix.npy")
mat1 = mat.astype(np.float32)


def svd(mat1):
    '''Compute the singular value decomposition of array.'''

    # Golub and Reinsch state that eps should not be smaller than the
    # machine precision, ie the smallest number
    # for which 1+e>1.  tol should be beta/e where beta is the smallest
    # positive number representable in the computer.
    eps = 1.e-15  # assume double precision
    tol = 1.e-64/eps
    assert 1.0+eps > 1.0  # if this fails, make eps bigger
    assert tol > 0.0     # if this fails, make tol bigger
    itmax = 50
    matU = mat1
    m = len(mat1)
    n = len(mat1[0])
    # if __debug__: print 'a is ',m,' by ',n
    # print(matrixU,m,n)
    if m < n:
        if __debug__:
            print('Error: m is less than n')
        raise (ValueError, 'SVD Error: m is less than n.')

    e = [0.0]*n  # allocate arrays
    eMat = np.zeros(n)
    qMat = np.zeros(n)
    matV = np.ndarray(shape=(n, n))
    matV.fill(0.0)

    # Householder's reduction to bidiagonal form
    g = 0.0
    x = 0.0
    for i in range(n):
        eMat[i] = g
        s = 0.0
        l = i+1
        #COLi * COLi
        s += np.dot(matU[i:, i], matU[i:, i])

        if s <= tol:
            g = 0.0
        else:
            f = matU[i][i]
            if f < 0.0:
                g = np.sqrt(s)
            else:
                g = -np.sqrt(s)
            h = f*g-s
            matU[i][i] = f-g
            for j in range(l, n):
                s = 0.0
                # multiplying col i with other cols of the matrix
                s += np.dot(matU[i:, j], matU[i:, i])
                f = s/h
                # colj = coli*f + colj; (j=i+1 to m)
                matU[i:, j] = matU[i:, j]+(f*matU[i:, i])
                # print(matU)
        qMat[i] = g
        s = 0.0
        s += np.dot(matU[i, l:], matU[i, l:])
        # print(s)
        if s <= tol:
            g = 0.0
        else:
            f = matU[i][i+1]
            if f < 0.0:
                g = math.sqrt(s)
            else:
                g = -math.sqrt(s)
            h = f*g - s
            matU[i][i+1] = f-g
            eMat[l:] = matU[i, l:]/h
            for j in range(l, m):
                s = 0.0
                s += np.dot(matU[j, l:], matU[i, l:])
                matU[j, l:] = matU[j, l:] + (s*eMat[l:])
        y = abs(qMat[i]) + abs(eMat[i])
        if y > x:
            x = y

    # print(matU)
    # print(matV)
    # print(eMat)

    # accumulation of right hand gtransformations
    for i in range(n-1, -1, -1):
        if g != 0.0:
            h = g*matU[i][i+1]
            matV[l:, i] = matU[i, l:]/h
            for j in range(l, n):
                s = 0.0
                s += np.dot(matU[i, l:], matV[l:, j])
                # print(s)
                matV[l:, j] += s*matV[l:, i]
        matV[i, l:] = 0.0
        matV[l:, i] = 0.0
        matV[i][i] = 1.0
        g = eMat[i]
        l = i
    # print(matU)
    # accumulation of left hand transformations
    for i in range(n-1, -1, -1):
        l = i+1
        g = qMat[i]
        #for j in range(l,n): matU[i][j] = 0.0
        matU[i, l:] = 0.0
        if g != 0.0:
            h = matU[i][i]*g
            for j in range(l, n):
                s = 0.0
                s = np.sum(np.dot(matU[l:, i], matU[l:, j]))
                # print(i,j,matU[l:,i]*matU[l:,j]
                # print(s)
                f = s/h
                # print(f)
                # print(f,matU[i:,i])
                matU[i:, j] += (f*matU[i:, i])
            matU[i:, i] = matU[i:, i]/g
        else:
            matU[i:, i] = 0.0
        matU[i][i] += 1.0
    # print(matU)

    # diagonalization of the bidiagonal form
    eps = eps*x
    for k in range(n-1, -1, -1):
        for iteration in range(itmax):
            # test f splitting
            for l in range(k, -1, -1):
                goto_test_f_convergence = False
                if abs(eMat[l]) <= eps:
                    # goto test f convergence
                    goto_test_f_convergence = True
                    break  # break out of l loop
                if abs(qMat[l-1]) <= eps:
                    # goto cancellation
                    break  # break out of l loop
            if not goto_test_f_convergence:
                # cancellation of e[l] if l>0
                c = 0.0
                s = 1.0
                l1 = l-1
                for i in range(l, k+1):
                    f = s*eMat[i]
                    eMat[i] = c*eMat[i]
                    if abs(f) <= eps:
                        # goto test f convergence
                        break
                    g = qMat[i]
                    h = pythag(f, g)
                    qMat[i] = h
                    c = g/h
                    s = -f/h
                    for j in range(m):
                        y = matU[j][l1]
                        z = matU[j][i]
                        matU[j][l1] = y*c+z*s
                        matU[j][i] = -y*s+z*c
            # test f convergence
            z = qMat[k]
            if l == k:
                # convergence
                if z < 0.0:
                    # q[k] is made non-negative
                    qMat[k] = -z
                    matV[:, k] = -matV[:, k]
                break  # break out of iteration loop and move on to next k value
            if iteration >= itmax-1:
                if __debug__:
                    print('Error: no convergence.')
                # should this move on the the next k or exit with error??
                # raise ValueError,'SVD Error: No convergence.'  # exit the program with error
                break  # break out of iteration loop and move on to next k
            # shift from bottom 2x2 minor
            x = qMat[l]
            y = qMat[k-1]
            g = eMat[k-1]
            h = eMat[k]
            f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
            g = pythag(f, 1.0)
            if f < 0:
                f = ((x-z)*(x+z)+h*(y/(f-g)-h))/x
            else:
                f = ((x-z)*(x+z)+h*(y/(f+g)-h))/x
            # next QR transformation
            c = 1.0
            s = 1.0
            for i in range(l+1, k+1):
                g = eMat[i]
                y = qMat[i]
                h = s*g
                g = c*g
                z = pythag(f, h)
                eMat[i-1] = z
                c = f/z
                s = h/z
                f = x*c+g*s
                g = -x*s+g*c
                h = y*s
                y = y*c
                for j in range(n):
                    x = matV[j][i-1]
                    z = matV[j][i]
                    matV[j][i-1] = x*c+z*s
                    matV[j][i] = -x*s+z*c
                z = pythag(f, h)
                qMat[i-1] = z
                c = f/z
                s = h/z
                f = c*g+s*y
                x = -s*g+c*y
                for j in range(m):
                    y = matU[j][i-1]
                    z = matU[j][i]
                    matU[j][i-1] = y*c+z*s
                    matU[j][i] = -y*s+z*c
            eMat[l] = 0.0
            eMat[k] = f
            qMat[k] = x
            # goto test f splitting

    #vt = transpose(v)
    # return (u,q,vt)
    return (matU, qMat, matV)


def pythag(a, b):
    absa = abs(a)
    absb = abs(b)
    if absa > absb:
        return absa*math.sqrt(1.0+(absb/absa)**2)
    else:
        if absb == 0.0:
            return 0.0
        else:
            return absb*math.sqrt(1.0+(absa/absb)**2)


def transpose(a):
    '''Compute the transpose of a matrix.
    m = len(a)
    n = len(a[0])
    at = []
    for i in range(n): at.append([0.0]*m)
    for i in range(m):
        for j in range(n):
            at[j][i]=a[i][j]'''
    at = np.transpose(a)
    return at


def matrixmultiply(a, b):
    '''Multiply two matrices.
    a must be two dimensional
    b can be one or two dimensional.'''

    am = len(a)
    bm = len(b)
    # print(len(a))
    if(len(a) != 1):
        an = len(a[0])
    else:
        an = 1
    try:
        bn = len(b[0])
    except TypeError:
        bn = 1
    if an != bm:
        raise (ValueError, 'matrixmultiply error: array sizes do not match.')
    cm = am
    cn = bn
    if bn == 1:
        c = [0.0]*cm
    else:
        c = []
        for k in range(cm):
            c.append([0.0]*cn)
    for i in range(cm):
        for j in range(cn):
            for k in range(an):
                if bn == 1:
                    c[i] += a[i][k]*b[k]
                    # print(c[i])
                else:
                    c[i][j] += a[i][k]*b[k][j]

    return c


'''
#matrixmultiply(a,a)
u1,w1,v1=svd(a)
v2=transpose(v1)
#print(u1)
#print(" ")
#print(w1)
#print(" ")
#print(v1)
#print(" ")
#t=(matrixmultiply(u1,w1))
#print(t)
#print(" ")
#print(v2)
#print(" ")
#print(matrixmultiply(v2,t))
w=[]
w3=[]
for i in range(len(w1)):
    for j in range(len(w1)):
        if(i==j):
            w.append(w1[i])
        else:
            w.append(0)
    w3.append(w)
    w=[]
t=(matrixmultiply(u1,w3))
#print(t)
h1=(matrixmultiply(t,v2))
#print(w3)
h=[]
h2=[]
for i in range(len(w1)):
    for j in range(len(w1)):
        h.append(round(h1[i][j]))
    h2.append(h)
    h=[]
stop = time.time()
print(h2)
print(stop-st)
'''

start = time.time()
u1, w1, v1 = svd(mat1)
end = time.time()

print(end-start)
print(w1)
# # print(u1)
# # print(v1)
# # print(w1)

# v2 = transpose(v1)
# # print(v2)
# w = []
# w3 = []
# for i in range(len(w1)):
#     for j in range(len(w1)):
#         if(i == j):
#             w.append(w1[i])
#         else:
#             w.append(0)
#     w3.append(w)
#     w = []
# wnp = np.array(w3)
# # t=(matrixmultiply(u1,w3))
# t = np.matmul(u1, w3)
# # print(t)
# h1 = np.matmul(t, v2)
# # h1=(matrixmultiply(t,v2))
# # print(w3)
# h = []
# h2 = []
# for i in range(len(w1)):
#     for j in range(len(w1)):
#         h.append(round(h1[i][j]))
#     h2.append(h)
#     h = []
# stop = time.time()
# print(h2)
# print(stop - st)
