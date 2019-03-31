import numpy as np
import struct, sys
from scipy.misc import imsave
from scipy.signal import convolve2d


win_size = 9
down_size = 28 - win_size + 1

def dataLoader(mode='train'):

    dataFile = 'train-images.idx3-ubyte' if mode=='train' else 't10k-images.idx3-ubyte'
    labelFile = 'train-labels.idx1-ubyte' if mode=='train' else 't10k-labels.idx1-ubyte'

    """ Read Images
    """
    index = 0
    buf = open(dataFile, 'rb')
    buf = buf.read()
    _, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')
    
    X = np.empty((numImages, 784))
    for i in xrange(numImages):
        im = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        im = np.array(im)
        X[i, :] = im

    """ Read Labels
    """
    index = 0
    buf = open(labelFile, 'rb')
    buf = buf.read()
    _, numLabels = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')

    Y = np.empty(numLabels)
    for i in xrange(numLabels):
        num = struct.unpack_from('1B', buf, index)
        Y[i] = num[0]
        index += struct.calcsize('1B')
    
    return X, Y.astype(int)
    # return (X>127).astype(float), Y.astype(int)


def KNN(XTrain, YTrain, XTest, YTest, disFunc, k=1, tangent=False):

    if tangent:
        dis = disFunc(XTrain, XTest, rotTangent(XTrain, down_size, down_size))
    else:
        dis = disFunc(XTrain, XTest)
    numTrain = XTrain.shape[0]
    numTest = XTest.shape[0]
    minMatch = np.argpartition(dis, kth=k-1, axis=1)[:, :k]
    matchedClasses = YTrain[minMatch.reshape(-1)].reshape(numTest, k)
    binCount = np.array(map(lambda x:np.bincount(x,minlength=10), matchedClasses))
    matchedClass = np.argmax(binCount, axis=1)

    return (matchedClass==YTest).astype(np.float).sum()/XTest.shape[0]


def Euclidean(XTrain, XTest):
    
    testNorm = np.sum(XTest*XTest, axis=1)[:, np.newaxis]
    trainNorm = np.sum(XTrain*XTrain, axis=1)[np.newaxis, :]
    cross = XTest.dot(XTrain.T)
    return np.sqrt(testNorm + trainNorm - 2*cross)

def Mahalanobis(XTrain, XTest):
    
    X = np.vstack((XTrain, XTest))
    mu = np.mean(X, axis=0)[np.newaxis, :]
    Sigma = (X-mu).T.dot(X-mu) 
    L = np.linalg.cholesky(Sigma)
    L_inv = L**(-1)
    
    newXTrain = (L_inv.dot(XTrain.T)).T 
    newXTest = (L_inv.dot(XTest.T)).T 

    testNorm = np.sum(newXTest*newXTest, axis=1)[:, np.newaxis]
    trainNorm = np.sum(newXTrain*newXTrain, axis=1)[np.newaxis, :]
    cross = newXTest.dot(newXTrain.T)
    return np.sqrt(testNorm + trainNorm - 2*cross)

def L4(XTrain, XTest):

    testQuad = np.sum(XTest**4, axis=1)[:, np.newaxis]
    trainQuad = np.sum(XTrain**4, axis=1)[np.newaxis, :]

    cross = -4*(XTest**3).dot(XTrain.T)
    cross += -4*(XTest).dot((XTrain**3).T)
    cross += 6*(XTest**2).dot((XTrain**2).T)

    return (testQuad + trainQuad + cross)**(1.0/4)

def rotTangent(XTrain, h=28, w=28):

    Xtemp = XTrain.reshape((-1, h, w))
    partial_x = np.zeros(Xtemp.shape)
    partial_y = np.zeros(Xtemp.shape)
    partial_x[:, 0:h-1, :] = Xtemp[:, 1:h, :] - Xtemp[:, 0:h-1, :]
    partial_y[:, :, 0:w-1] = Xtemp[:, :, 1:w] - Xtemp[:, :, 0:w-1] 
    
    factor = 1.0/(w*0.5)
    offset = 0.5 - 1.0/factor 
    mesh_y, mesh_x = np.meshgrid(range(h), range(w))
    mesh_x, mesh_y = offset+mesh_x, offset+mesh_y
    mesh_x, mesh_y = mesh_x[np.newaxis, :], mesh_y[np.newaxis, :]

    tangent = (mesh_y*partial_x - mesh_x*partial_y)*factor
    return tangent.reshape((-1, h*w)).copy()

def tangentDistance(XTrain, XTest, Tangent):

    cross1 = XTest.dot(Tangent.T)
    cross2 = np.sum(XTrain*Tangent, axis=1)[np.newaxis, :]
    cross3 = np.sum(Tangent*Tangent, axis=1)[np.newaxis, :]
    Alpha = (cross1 - cross2)/cross3

    item = Euclidean(XTrain, XTest)
    item += -2*Alpha*(cross1-cross2)
    item += (Alpha**2)*cross3

    return item

def make_gaussian_window(n, sigma=1):

    nn = int((n-1)/2)
    a = np.asarray([[x**2 + y**2 for x in range(-nn,nn+1)] for y in range(-nn,nn+1)])
    return np.exp(-a/(2*sigma**2))

def conv(im, window):
    im_conv = convolve2d(im, window, mode='valid')
    return im_conv

if __name__ == "__main__":

    XTrain, YTrain = dataLoader('train')
    XTest, YTest = dataLoader('test')

    scale = 60000
    XTrain = XTrain[:scale, :]
    YTrain = YTrain[:scale]

    window = make_gaussian_window(win_size, 0.1)
    train, test = [], []
    for im in XTrain:
        train.append(conv(im.reshape(28, 28), window).reshape(-1).copy())
    XTrain = np.array(train)
    for im in XTest:
        test.append(conv(im.reshape(28, 28), window).reshape(-1).copy())
    XTest = np.array(test)

    print(XTrain.shape, XTest.shape)

    print(KNN(XTrain, YTrain, XTest, YTest, disFunc=Euclidean, k=3, tangent=False))
 
