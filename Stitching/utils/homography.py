import numpy as np
def computeH(im1Points, im2Points):
    A = np.zeros((im1Points.shape[0]*2, 9))
    for i in range(im1Points.shape[0]):
        x1 = im1Points[i, 0]
        y1 = im1Points[i, 1]
        x2 = im2Points[i, 0]
        y2 = im2Points[i, 1]
        A[2*i,0] = x1
        A[2*i,1] = y1
        A[2*i,2] = 1
        A[2*i,6] = -x1*x2
        A[2*i,7] = -y1*x2
        A[2*i,8] = -x2
        A[2*i+1,3] = x1
        A[2*i+1,4] = y1
        A[2*i+1,5] = 1
        A[2*i+1,6] = -x1*y2
        A[2*i+1,7] = -y1*y2
        A[2*i+1,8] = -y2
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    H = vh[-1].reshape((3,3))
    return H / H[2,2]
    