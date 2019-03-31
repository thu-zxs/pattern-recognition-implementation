import numpy as np
import struct, sys, time
from sklearn.cluster import spectral_clustering

np.random.seed(2018)

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

def NMI(data_cls, data_gt):

    n = data_cls.shape[0]
    clses = list(set(data_cls.tolist()))
    num_clses = []
    for i in clses:
        num_clses.append(np.where(data_cls==i)[0])

    clses_gt = list(set(data_gt.tolist()))
    num_clses_gt = []
    for i in clses_gt:
        num_clses_gt.append(np.where(data_gt==i)[0])

    nmi = 0
    for c1 in num_clses:
        ns = c1.shape[0]
        for c2 in num_clses_gt:
            nt = c2.shape[0]
            n_st = len(set(c1.tolist()).intersection(set(c2.tolist())))
            if n_st == 0: continue
            nmi += n_st*np.log(float(n)*n_st/(ns*nt))

    sum_ns = 0
    for c1 in num_clses:
        ns = c1.shape[0]
        sum_ns += ns*np.log(float(ns)/n)

    sum_nt = 0
    for c2 in num_clses_gt:
        nt = c2.shape[0]
        sum_nt += nt*np.log(float(nt)/n)

    nmi /= np.sqrt(sum_ns*sum_nt)
    return nmi


        
        

def kmeans(data, label, cls_num=10):

    data_size = data.shape[0]
    data_cls = -np.ones((data_size, 1))

    # Random choose samples
    init_center_idx = np.random.choice(np.arange(data_size), size=cls_num)
    center = data[init_center_idx]

    data_cls[init_center_idx, :] = np.arange(cls_num)[:, np.newaxis]

    epsilon = 1e-3
    iter_cnt = 0
    XSquare = np.sum(data**2, axis=1)[:, np.newaxis]
    while True:
        # Compution of distance
        CSquare = np.sum(center**2, axis=1)[np.newaxis, :]
        XCCross = data.dot(center.T)
        Dist = np.sqrt(XSquare + CSquare - 2*XCCross)
        data_cls = np.argmin(Dist, axis=1)
        center_old = center.copy()
        J_e = 0
        for i in xrange(cls_num):
            clustering = data[np.where(data_cls==i)]
            if clustering.shape[0] == 0: continue
            center[i, :] = np.mean(clustering, axis=0)
            J_e += np.sum(np.sum(clustering**2, axis=1) + np.sum(center[i,:]**2) - 2*clustering.dot(center[i,:][:, np.newaxis]))

        iter_cnt += 1
        if np.sum(np.abs(center-center_old)) < 1e-3:
            break

        print("[{}] J_e = {}".format(iter_cnt, J_e))
        print("[{}] NMI = {}".format(iter_cnt, NMI(data_cls, label)))
    print(iter_cnt)
    return data_cls


def cluster_dist(c1, c2, kind="min"):
    
    if kind == "min" or kind == "max":
        c1sq = np.sum(c1**2, axis=1)[:, np.newaxis]
        c2sq = np.sum(c2**2, axis=1)[np.newaxis, :]
        cross = c1.dot(c2.T)
        Dist = np.sqrt(c1sq + c2sq - 2*cross)
        if kind == "min":
            return Dist.min()
        elif kind == "max":
            return Dist.max()
    elif kind == "mean":
        return np.sqrt(np.sum((c1.mean(axis=0)-c2.mean(axis=0))**2))



def hierarhical(data, label, kind="min", cls_num=10):

    data_size = data.shape[0]
    data_cls = np.arange(data_size)
    data_cls_set = list(set(data_cls.tolist()))
    min_dist_clses = None
    while len(data_cls_set) > cls_num:
        min_dist = np.inf
        for i, c1 in enumerate(data_cls_set):
            cluster1 = data[np.where(data_cls==c1)]
            for c2 in data_cls_set[i+1:]:
                cluster2 = data[np.where(data_cls==c2)]
                dist = cluster_dist(cluster1, cluster2, kind=kind)
                if dist < min_dist:
                    min_dist = dist
                    min_dist_clses = (c1, c2)
        data_cls[np.where(data_cls==min_dist_clses[1])] = min_dist_clses[0]
        data_cls_set = list(set(data_cls.tolist()))
    return data_cls


def spectral(data, cls_num=10, kind="cosine", n_components=784):

    Cosine = data.dot(data.T)
    Cosine /= np.linalg.norm(data, axis=1)[np.newaxis, :]
    Cosine /= np.linalg.norm(data, axis=1)[:, np.newaxis]

    labels = spectral_clustering(Cosine, n_clusters=cls_num, n_components=n_components)
    return labels

if __name__ == "__main__":

    XTrain, YTrain = dataLoader('train')
    XTest, YTest = dataLoader('test')
    
    scale = 60
    XTrain = XTrain[:scale, :]
    YTrain = YTrain[:scale]

    start = time.clock()
    kmeans(XTrain, YTrain)
    end = time.clock()
    print("kmeans: {}".format(end-start))

    start = time.clock()
    data_cls = hierarhical(XTrain, YTrain, kind="min")
    end = time.clock()
    print("hierarhical: {}".format(end-start))
    print("NMI-min: {}".format(NMI(data_cls, YTrain)))

    data_cls = hierarhical(XTrain, YTrain, kind="max")
    print("NMI-max: {}".format(NMI(data_cls, YTrain)))

    data_cls = hierarhical(XTrain, YTrain, kind="mean")
    print("NMI-mean: {}".format(NMI(data_cls, YTrain)))

    start = time.clock()
    data_cls = spectral(XTrain, n_components=10)
    end = time.clock()
    print("spectral: {}".format(end-start))
    print("comp-10: {}".format(NMI(data_cls, YTrain)))

    data_cls = spectral(XTrain, n_components=5)
    print("comp-5: {}".format(NMI(data_cls, YTrain)))

