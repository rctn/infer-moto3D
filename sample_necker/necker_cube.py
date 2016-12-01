'''
Script to generated a Necker Cube through a 3D projection
Mayur Mudigonda
Nov 30th, 2016
'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def proj_3d_2d(X):
    W = np.ones((2,3))
    W[0,2] = 0.01
    W[1,2] = 0.01
    W[0,1] = 0.
    W[1,0] = 0.
   # W[0,0] = 0.25
   # W[1,1] = 0.25
    projected = np.dot(W,X.T)
    return projected

def rot_3D(X,Rand=None):
    if Rand is not None:
        Rot = np.random.randn(3,3)
    else:
        print("Fixed Rotation")
        Rot = np.ones((3,3))
        Rot[0,1] = 0.1
        Rot[1,0] = 0.1
    rotated = np.dot(Rot,X.T)
    return rotated.T

def make_3D():
    X = np.zeros((8,3))
    X[0,:] = np.array([0.,0.,0.])
    X[1,:] = np.array([0.,0.,1.])
    X[2,:] = np.array([1.,0.,0.])
    X[3,:] = np.array([1.,0.,1.])
    X[4,:] = np.array([0.,1.,0.])
    X[5,:] = np.array([1.,1.,0.])
    X[6,:] = np.array([0.,1.,1.])
    X[7,:] = np.array([1.,1.,1.])
    return X

if __name__ == "__main__":
    print("hello")
    X = make_3D()
    X = rot_3D(X)
    print X
    '''
    fig = plt.figure()
    ax  = fig.gca(projection='3d')
    ax.plot(X[:,0],X[:,1],X[:,2],'r+')
    ax.hold(True)
    #ax.plot(X[:,0],X[:,1],X[:,2],'g')
    ax.view_init(elev=18,azim=-27)
    ax.dist = 9
    plt.show()
    '''
    projected = proj_3d_2d(X)
    print(projected.T)
    plt.plot(projected.T,'r+')
    plt.hold(True)
    plt.plot(X[:,0],X[:,1],'g*')
    plt.axis([-2.5,2.5,-2.5,2.5])
    plt.show()
