import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def least_sq_L1(X, y, _lambda, w_0):

    X = X.astype(float)
    n, M = X.shape
    a = np.sum(X**2, axis=0)/n

    w = w_0
    err_tol = 1e-8;
    while True:
        max_err = 0
        w_old = w.copy()
        for k in xrange(M):
            w_minus_k = np.delete(w, k, axis=1)
            phi_minus_k = np.delete(X, k, axis=1)
            c_k = np.sum((y-phi_minus_k.dot(w_minus_k.T))*X[:, k][:, np.newaxis], axis=0)/float(n)
            if c_k < -_lambda:
                w[0][k] = (c_k + _lambda)/a[k]
            elif -_lambda <= c_k < _lambda:
                w[0][k] = 0
            elif c_k >= _lambda:
                w[0][k] = (c_k - _lambda)/a[k]
        max_err = np.max(np.abs(w - w_old))
        # print(max_err)
        if max_err < err_tol:
            break
    return w

def least_sq_multi(X, y, _lambda, w_0):

    _, M = X.shape
    L = _lambda.shape[0]
    W = np.zeros((L, M))
    w_l = w_0
    for l in xrange(L):
        w_l = least_sq_L1(X, y, _lambda[l], w_l)
        W[l, :] = w_l
    return W 

if __name__ == "__main__":

    mat = sio.loadmat("least_sq.mat")
    # _set = "train_small"
    # _set = "train_mid"
    _set = "train_large"

    X = mat[_set][0][0][0]
    y = mat[_set][0][0][1]
    n = X.shape[0]
    X_test = mat["test"][0][0][0]
    y_test = mat["test"][0][0][1]
    n_test = X_test.shape[0]
    _lambda = np.arange(0.01, 2.01, 0.01)
    w_0 = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y)).T

    W = least_sq_multi(X, y, _lambda, w_0)

    L = _lambda.shape[0]
    trainError = np.zeros(L)
    regPenalty = np.zeros(L)
    objective = np.zeros(L)
    num_non_zero = np.zeros(L)
    test_error = np.zeros(L)


    for l in xrange(_lambda.shape[0]):
        w = W[l, :][np.newaxis, :]
        trainError[l] = np.sum(0.5*(y-X.dot(w.T))**2, axis=0)/n
        regPenalty[l] = np.sum(np.abs(w)) 
        objective[l] = trainError[l]+_lambda[l]*regPenalty[l]
        num_non_zero[l] = np.sum(w!=0)
        test_error[l] = np.sum(0.5*(y_test-X_test.dot(w.T))**2, axis=0)/n_test

    plt.figure(figsize=(12, 8)) 

    ax = plt.subplot(231)
    ax.plot(_lambda, trainError)
    ax.set_title("training error")
    
    ax = plt.subplot(232)
    ax.plot(_lambda, regPenalty)
    ax.set_title("regularization penalty")
    
    ax = plt.subplot(233)
    ax.plot(_lambda, objective)
    ax.set_title("miminized objective")

    ax = plt.subplot(234)
    ax.plot(_lambda, num_non_zero)
    ax.set_title("number of non-zero")

    ax = plt.subplot(235)
    ax.plot(_lambda, test_error)
    ax.set_title("test error")

    plt.savefig("result_large.png")


