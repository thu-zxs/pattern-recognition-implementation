# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

num = 1000
LR = -10.0
RR = 10.0
delta = (RR-LR)/(num-1)

###==== Basic Function ====

def gen2mixtures(mu1, sigma1, mu2, sigma2, alpha=0.2, n_samples=10000):

    norm1 = np.random.normal(mu1, sigma1, n_samples)
    norm2 = np.random.normal(mu2, sigma2, n_samples)
    norm = np.vstack((norm1, norm2))
    uniform = np.random.sample(n_samples)
    alphas = np.asarray([int(a>alpha) for a in uniform])
    
    return norm[alphas, np.arange(n_samples)]

def gaussian(x, mu, sigma):
    """ 高斯pdf
    """
    y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return y

def draw_pnx_pdf_with_parzen_window(samples, a):
    """ 根据a值绘出Parzen窗估计的概率密度函数p_n(x)
    """
    x = np.linspace(LR, RR, num)
    N = samples.shape[0]
    tmp = x[:, np.newaxis] - samples
    px = np.sum(1.0/a*((tmp<=0.5*a) & (tmp>=-0.5*a)).astype(np.int), axis=1)/N
    # plt.plot(x,px)
    # plt.show()
    return x, px 

def draw_pnx_pdf_with_gaussian_window(samples, a):
    """ 根据a值绘出Gaussian窗估计的概率密度函数p_n(x)
    """
    x = np.linspace(LR, RR, num)
    N = samples.shape[0]
    tmp = x[:, np.newaxis] - samples
    sigma = a**2
    px = np.sum(1.0/(2*np.pi*sigma)*np.exp(-(tmp)**2/(2*sigma)), axis=1)/N
    # plt.plot(x,px)
    # plt.show()
    return x, px

def draw_px_pdf(mu1, sigma1, mu2, sigma2, alpha=0.2):
    """ 高斯混合pdf, 即px的pdf
    """
    x = np.linspace(LR, RR, num)
    px = alpha*gaussian(x, mu1, sigma1) + (1-alpha)*gaussian(x, mu2, sigma2)
    return x, px 

##################

###==== Problems Solver ====

def draw_with_different_a(samples, window_type='Parzen'):
    """ a)问
    """
    if window_type == 'Parzen':
        x, pnx1 = draw_pnx_pdf_with_parzen_window(samples, a=0.1)
        _, pnx2 = draw_pnx_pdf_with_parzen_window(samples, a=1)
        _, pnx3 = draw_pnx_pdf_with_parzen_window(samples, a=10)

    elif window_type == 'Gaussian':
        x, pnx1 = draw_pnx_pdf_with_gaussian_window(samples, a=0.1)
        _, pnx2 = draw_pnx_pdf_with_gaussian_window(samples, a=1)
        _, pnx3 = draw_pnx_pdf_with_gaussian_window(samples, a=10)

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131)
    ax1.plot(x, pnx1); ax1.set_title("a=0.1")
    ax2 = fig.add_subplot(132)
    ax2.plot(x, pnx2); ax2.set_title("a=1")
    ax3 = fig.add_subplot(133)
    ax3.plot(x, pnx3); ax3.set_title("a=10")
    plt.suptitle("{} Window".format(window_type))
    # plt.show()
    plt.savefig("a_{}.jpg".format(window_type))

def compute_epsilon(samples, window_type='Parzen', a=1):
    """ b)问
    """
    if window_type == "Parzen":
        x, pnx = draw_pnx_pdf_with_parzen_window(samples, a=a)

    elif window_type == 'Gaussian':
        x, pnx = draw_pnx_pdf_with_gaussian_window(samples, a=a)

    _, px = draw_px_pdf(-1, 1, 1, 1)
    epsilon = np.sum((px[1:]-pnx[1:])**2*delta)

    return epsilon

def compute_exp_and_var_for_epsilon(window_type='Parzen'):
    """ c)问
    """

    # Fix n
    samples = gen2mixtures(-1, 1, 1, 1, n_samples=5, alpha=0.2)
    A = [0.02, 0.1, 1, 10, 50]
    res1 = []
    for a in A:
        res1.append(compute_epsilon(samples, window_type, a))
    print(res1)
    res1 = np.asarray(res1)
    mean1 = np.mean(res1)
    var1 = np.var(res1)

    # Fix a
    N = [100, 500, 1000, 5000, 10000]
    a = 1
    res2 = []
    for n in N:
        samples = gen2mixtures(-1, 1, 1, 1, n_samples=n, alpha=0.2)
        res2.append(compute_epsilon(samples, window_type, a))
    res2 = np.asarray(res2)
    mean2 = np.mean(res2)
    var2 = np.var(res2)

    return mean1, var1, mean2, var2




def main():

    # Draw n samples from p(x)
    samples = gen2mixtures(-1, 1, 1, 1, n_samples=10000, alpha=0.2)

    draw_with_different_a(samples, window_type='Gaussian')
    print(compute_epsilon(samples, window_type='Parzen'))
    print(compute_exp_and_var_for_epsilon(window_type='Gaussian'))
    

if __name__ == "__main__":
    main()
    

