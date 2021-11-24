import numpy as np
def computeH(im1Points, im2Points, normalization=True):
    if normalization:
        std1 = np.std(im1Points, axis=0)
        std2 = np.std(im2Points, axis=0)
        mean1 = np.mean(im1Points, axis=0)
        mean2= np.mean(im2Points, axis=0)
        T1 = np.array([[1/std1[0], 0, -mean1[0]/std1[0]],
                    [0, 1/std1[1], -mean1[1]/std1[1]],
                    [0, 0, 1]])
        T2 = np.array([[1/std2[0], 0, -mean2[0]/std2[0]],
                    [0, 1/std2[1], -mean2[1]/std2[1]],
                    [0, 0, 1]])
        im1Points = np.array([np.matmul(T1, np.append(p,1))[:-1] for p in im1Points])
        im2Points = np.array([np.matmul(T2, np.append(p,1))[:-1] for p in im2Points])
    
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
    
    if normalization:
        T2inv = np.matrix(np.linalg.inv(T2))
        H= np.matrix(H)
        T1 = np.matrix(T1)
        H = T2inv * H * T1
    H = H / H[2,2]
    return H
    