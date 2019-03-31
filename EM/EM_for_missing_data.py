import numpy as np
from numpy.linalg import inv

def _3_variate_gaussian():
    X = np.array([
        [ 0.42, -0.2, 1.3, 0.39, -1.6, -0.029, -0.23, 0.27, -1.9, 0.87 ], 
        [-0.087, -3.3, -0.32, 0.71, -5.3, 0.89, 1.9, -0.3, 0.076, -1.0 ],
        [0.58, -3.4, 1.7, 0.23, -0.15, -4.7, 2.2, -0.87, -2.1, -2.6 ] ])

    mu = np.zeros((3,1))
    Sigma = np.eye(3)
    epsilon = 1e-5
    mu_c = np.zeros((5, 1))
    sigma_c = np.zeros((5, 1))
    x_hat = np.zeros((3, 5))
    N_hat = np.zeros((5, 3, 3))
    N = np.zeros((5, 3, 3))

    error = 1
    while error > epsilon:
        for j in xrange(1, 11, 2):
            mu_c[j/2, 0] = mu[2,0] + np.squeeze(Sigma[2:3, 0:2].dot(inv(Sigma[0:2, 0:2])).dot(X[0:2, j:j+1]-mu[0:2, 0:1]))
            sigma_c[j/2, 0] = Sigma[2,2] - np.squeeze(Sigma[2:3, 0:2].dot(inv(Sigma[0:2,0:2])).dot(Sigma[0:2,2:3]))
            x_hat[:, j/2] = np.array([X[0, j],X[1, j], mu_c[j/2,0]])

        mu_new = 1.0/10 * (np.sum(x_hat, axis=1) + np.sum(X[:, ::2], axis=1))[:, np.newaxis]
        error = np.sum(np.abs(mu_new - mu))
        mu = mu_new

        for j in xrange(1, 11, 2):
            N_hat[j/2, :, :] = (x_hat[:, j/2]-mu).dot((x_hat[:, j/2]-mu).T)
            N[j/2, :, :] = (X[:, j-1]-mu).dot((X[:, j-1]-mu).T)

        Sigma_tmp = 1.0/10 * (np.sum(N_hat, axis=0) + np.sum(N, axis=0))
        Sigma_tmp[2,2] += 1.0/10*np.sum(sigma_c)
        error = np.sum(np.abs(Sigma_tmp-Sigma))
        Sigma = Sigma_tmp

    print("data missing (use EM Algorithm): \n -------------")
    print("mu:")
    print(mu)
    print("Sigma:")
    print(Sigma)
    print("\n")


    mu = np.mean(X, axis=1)[:, np.newaxis]
    Sigma = 1.0/10 * ((X-mu).dot((X-mu).T))
    print("No missing data (use max-likelihood method): \n --------------")
    print("mu:")
    print(mu)
    print("Sigma:")
    print(Sigma)


def _3_variate_uniform():

    X = np.array([
        [-0.4, -0.31, 0.38,-0.15, -0.35, 0.17, -0.011, -0.27, -0.065, -0.12],
        [0.58, 0.27, 0.055, 0.53, 0.47, 0.69, 0.55, 0.61, 0.49, 0.054],
        [0.089, -0.04, -0.035, 0.011, 0.034, 0.1, -0.18, 0.12, 0.0012, -0.063] ])

    xl = np.array([[-2], [-2], [-2]], dtype=np.float32)
    xr = np.array([[2], [2], [2]], dtype=np.float32)

    xl[0, 0] = np.min(X[0, :]); xr[0, 0] = np.max(X[0, :])
    xl[1, 0] = np.min(X[1, :]); xr[1, 0] = np.max(X[1, :])

    print("data missing \n --------")
    print("xl={}".format(xl))
    print("xr={}".format(xr))

    print("no missing data \n --------")
    print("xl={}".format(np.min(X, axis=1)[:, np.newaxis]))
    print("xr={}".format(np.max(X, axis=1)[:, np.newaxis]))

if __name__ == '__main__':
    _3_variate_gaussian()
    print('\n')
    print('===== Uniform Distribution =====' )
    _3_variate_uniform()
