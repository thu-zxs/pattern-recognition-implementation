import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
np.random.seed(2018)

def rotate_matrix(direction, theta):

    x, y, z = direction/(np.linalg.norm(direction))
    cos, sin = np.cos, np.sin
    t = theta
    return np.array([
            [cos(t)+(1-cos(t))*x**2, (1-cos(t))*x*y-sin(t)*z, (1-cos(t))*x*z+sin(t)*y],
            [(1-cos(t))*y*x+sin(t)*z, cos(t)+(1-cos(t))*y**2, (1-cos(t))*y*z-sin(t)*x],
            [(1-cos(t))*z*x-sin(t)*y, (1-cos(t))*z*y+sin(t)*x, cos(t)+(1-cos(t))*z**2]
           ])

def draw3(N):
    """ random generate 4 fold of data and form the `3` shape
    """
    x = np.linspace(-100, 100, N)
    
    delta = np.random.uniform(-10, 10, size=(N,))
    y1 = np.sqrt(50**2-(x+50)**2)
    y1[np.where(x>0)] = 0
    y2 = np.sqrt(50**2-(x-50)**2)
    y2[np.where(x<=0)] = 0
    # y3 = -np.sqrt(50**2-(x+50)**2)
    # y3[np.where(x>-80)] = 0
    # y4 = -np.sqrt(50**2-(x-50)**2)
    # y4[np.where(x<80)] = 0
    # y = y1 + y2 + y3 + y4 + delta
    y = y1 + y2 + delta
    y -= 25

    z = np.random.uniform(-10, 10, size=(N,))

    fig = plt.figure(1)
    ax = Axes3D(fig)
    
    ## ratate generated data along direction by theta
    direction = np.array([1,1,0])
    # theta = np.pi/2
    theta = 0
    rmat = rotate_matrix(direction, theta)
    x, y, z = np.dot(rmat, np.array([x,y,z]))
    ax.scatter(x, y, z)

    ## set axis range
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([-100, 100])
    # plt.show()
    plt.savefig('3.png')

    return np.array([x,y,z])

def drawN(N):
    """ random generate 3 fold of data and form the `N` shape 
        by transformation (rotation or translation).
    """
    x = np.linspace(-100, 100, N)

    delta = np.random.uniform(-10, 10, size=(N,))
    y1 = 4*x+300
    y1[np.where(x>-50)] = 0
    y3 = 4*x-300
    y3[np.where(x<50)] = 0
    y2 = -2*x
    y2[np.where((x<-50)|(x>50))] = 0
    y = y1 + y2 + y3 + delta

    z = np.random.uniform(-10, 10, size=(N,))

    fig = plt.figure()
    ax = Axes3D(fig)
    
    ## ratate generated data along direction by theta
    direction = np.array([1,1,0])
    # theta = np.pi/2
    theta = 0
    rmat = rotate_matrix(direction, theta)
    x, y, z = np.dot(rmat, np.array([x,y,z]))
    ax.scatter(x, y, z)

    ## set axis range
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([-100, 100])
    # plt.show()
    plt.savefig('N.png')

    return np.array([x,y,z])


def eculidean(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def ISOMAP(data, epsilon=30):

    N = data.shape[1]
    A = np.inf*(np.ones((N, N)))
    for i in xrange(N):
        for j in xrange(N):
            d = eculidean(data[:,i], data[:,j])
            if d < epsilon:
                A[i, j] = d

    for i in xrange(N):
        for j in xrange(N):
            for k in xrange(N):
                A[i,j] = min(A[i,j], A[i,k]+A[j,k])

    print(np.any(A==np.inf))
    # print(np.any(A<0))

    A = -0.5*A**2
    H = np.eye(N)-1.0/N*np.ones((N,N))
    B = (H.dot(A)).dot(H)
    # B = data.T.dot(data)
    V,E = np.linalg.eig(B)
    V = np.real(V)
    E = np.real(E)
    reduced = E[:,:2].dot(np.diag(V[:2]))
    # print(reduced)

    return reduced

def LLE(data, k=5):

    N = data.shape[1]
    p = data.shape[0]
    D = np.zeros((N,N))
    for i in xrange(N):
        for j in xrange(N):
            if i == j:
                D[i,j] = np.inf
            else:
                D[i,j] = eculidean(data[:,i], data[:,j])
    Neighbor = np.zeros((N, k), dtype=np.int)
    for i in xrange(N):
        Neighbor[i,:] = np.argpartition(D[i,:], kth=k-1)[:k]

    # Step 1: Solve W
    Q = np.zeros((N, k, k))
    w = np.zeros((N, k))
    for i in xrange(N):
        diff = data[:,i][:, np.newaxis] - data[:,Neighbor[i,:]]
        Q[i] = diff.T.dot(diff) 
        for j in xrange(k):
            Q_inv = np.linalg.inv(Q[i])
            w[i, j] = np.sum(Q_inv[j, :]) / np.sum(Q_inv)

    # Step 2: Solve M
    W = np.zeros((N, N))
    for i in xrange(N):
        W[i, Neighbor[i, :]] = w[i, :]

    M = (np.eye(N)-W).T.dot(np.eye(N)-W)

    V, E = np.linalg.eig(M)
    V = np.real(V)

    reduced = np.real(E[:, -2:])
    return reduced






if __name__ == "__main__":

    n_samples = 200

    dataN = drawN(n_samples)
    dataN_reduced = ISOMAP(dataN)
    fig = plt.figure(2)
    plt.scatter(dataN_reduced[:, 0], dataN_reduced[:, 1])
    plt.savefig("N_reduced.png")
    
    data3 = draw3(n_samples)
    data3_reduced = LLE(data3)
    fig = plt.figure(3)
    plt.scatter(data3_reduced[:, 0], data3_reduced[:, 1])
    plt.savefig("3_reduced.png")

    
