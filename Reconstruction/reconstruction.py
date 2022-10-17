import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
from scipy.sparse.linalg import lsqr as eqsol

if __name__ == '__main__':
    img_path='./lena.bmp'

    img=cv2.imread(img_path)

    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.max())
    img22=img.copy()
    ny,nx=img.shape

    cv2.imshow('img',img)
    minY, minX = 0, 0
    maxY, maxX = img.shape
    ranY = maxY - minY
    ranX = maxX - minX
    A = scipy.sparse.lil_matrix((ranX, ranX))
    A.setdiag(-1, -1)
    A.setdiag([0,4,0])
    A.setdiag(-1, 1)
    A = scipy.sparse.block_diag([A] * ranY).tolil()

    b=img.flatten()
    b=A.dot(b)
    x=eqsol(A,b)[0]
    x_c=x.copy()
    x=x.reshape((ranY,ranX))
    x[x>255]=255
    x[x<0]=0
    x=x.astype('uint8')
    cv2.imshow('result',x)

    cv2.waitKey(0)
    lse=np.linalg.norm((A*x_c)-b)
    print("LSE:", lse)