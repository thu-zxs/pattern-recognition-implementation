import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def newline(p1, p2, color):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax], c=color)
    ax.add_line(l)
    return l

def generate_2cls_data(w, scale=5, n=100):
    """ w is the normal vector of the hyperplane
    """
    X1 = scale*(2*(np.random.randn(n,2)-0.5))
    margin_point_positive = np.argmax(X1.dot(w), axis=0)
    margin_distance_positive = np.max(X1.dot(w), axis=0)[0]

    X2 = scale*(2*(np.random.randn(n,2)-0.5))
    margin_point_negative = np.argmin(X2.dot(w), axis=0)
    margin_distance_negative = np.min(X2.dot(w), axis=0)[0]

    norm_w = np.linalg.norm(w)
    w_norm = w/norm_w
    dis_vec = (X1[margin_point_positive] - X2[margin_point_negative])
    mv_axis0 = dis_vec.dot(np.array([[1],[0]]))[0]
    mv_axis1 = dis_vec.dot(np.array([[0],[1]]))[0]
    X2[:,0] += mv_axis0*(1.2)
    X2[:,1] += mv_axis1*(1.2)

    return np.vstack((X1, X2)), np.hstack((-np.ones(n, dtype=int), np.ones(n, dtype=int)))

def perceptron(w0, X, Y):
    w = w0
    k = 0
    n = X.shape[0]
    X = np.hstack((X, np.ones((n,1))))
    all_correct = np.zeros(n, dtype=bool)
    while not np.all(all_correct):
        if not Y[k]*X[k,:].dot(w) > 0:
            w = w + Y[k]*X[k,:][:, np.newaxis]
        k = (k+1)%n
        all_correct = (Y*np.squeeze(X.dot(w)))>0
    return w

def perceptron_margin(w0, X, Y, margin):
    w = w0
    k = 0
    n = X.shape[0]
    X = np.hstack((X, np.ones((n,1))))
    all_correct_margin = np.zeros(n, dtype=bool)
    while not np.all(all_correct_margin):
        if not Y[k]*X[k,:].dot(w) > margin:
            w = w + Y[k]*X[k,:][:, np.newaxis]
        k = (k+1)%n
        all_correct_margin = (Y*np.squeeze(X.dot(w))) > margin
    return w



if __name__ == "__main__":

    X, Y = generate_2cls_data(w=np.array([[0.5],[-0.1]]), scale=5)
    rand_idx = np.random.choice(range(X.shape[0]), X.shape[0])
    X = X[rand_idx, :]
    Y = Y[rand_idx]
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
    plt.savefig('point.jpg')

    """ Problem1
    """
    w1 = perceptron(np.array([[-0.1], [-0.5], [3]]), X, Y)
    print(w1)
    w1 = np.squeeze(w1)
    p1 = [0, -w1[2]/w1[1]]
    p2 = [-w1[2]/w1[0], 0]
    l1 = newline(p1,p2,'r')
    plt.savefig('divider.jpg')

    """ Problem2
    """
    ax = plt.gca()
    ax.lines.remove(l1)
    colors = ['green', 'blue', 'navy', 'purple', 'brown']
    ls = []
    margins = [1, 10, 50, 100, 200]
    for i,k in enumerate(margins):
        w2 = perceptron_margin(np.array([[-0.1], [-0.5], [3]]), X, Y, margin=k)
        print("margin: {}\n{}".format(k, w2))
        w2 = np.squeeze(w2)
        p3 = [0, -w2[2]/w2[1]]
        p4 = [-w2[2]/w2[0], 0]
        l = newline(p3,p4,colors[i])
        ls.append(l)
    plt.legend(ls, np.array(margins, dtype=np.str))
    plt.savefig('divider_margin.jpg')


