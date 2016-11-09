#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from cluster import *
from sklearn import manifold


def plot_scatter_diagram(which_fig, x, y, x_label = 'x', y_label = 'y', title = 'title', cluster = None):
    '''
    Plot scatter diagram
    Args:
        which_fig  : which sub plot
        x          : x array
        y          : y array
        x_label    : label of x pixel
        y_label    : label of y pixel
        title      : title of the plot
    '''
    styles = ['k.', 'g.', 'r.', 'b.', 'y.', 'm.', 'c.']
    assert len(x) == len(y)
    if cluster != None:
        assert len(x) == len(cluster) and len(styles) >= len(set(cluster))
    plt.figure(which_fig)
    plt.clf()
    if cluster == None:
        plt.plot(x, y, styles[0])
    else:
        clses = set(cluster)
        xs, ys = {}, {}
        for i in range(len(x)):
            try:
                xs[cluster[i]].append(x[i])
                ys[cluster[i]].append(y[i])
            except KeyError:
                xs[cluster[i]] = [x[i]]
                ys[cluster[i]] = [y[i]]
        color = 1
        for idx, cls in enumerate(clses):
            if cls == -1:
                style = styles[0]
            else:
                style = styles[color]
                color += 1
            plt.plot(xs[cls], ys[cls], style)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(bottom = 0)
    plt.show()


def plot_rho_delta(rho, delta):
    '''
    Plot scatter diagram for rho-delta points
    Args:
        rho   : rho list
        delta : delta list
    '''
    plot_scatter_diagram(0, rho[1:], delta[1:], x_label='rho', y_label='delta', title='rho-delta')


def plot_cluster(dpcluster):
    '''
    Plot scatter diagram for final points that using multi-dimensional scaling(MDS) for data
    Args:
        dpcluster : DensityPeakCluster object
    '''
    dp = np.zeros((dpcluster.num, dpcluster.num), dtype = np.float32)
    cls = []
    for i in range(1, dpcluster.num):
        for j in range(i + 1, dpcluster.num + 1):
            dp[i - 1, j - 1] = dpcluster.distance[(i, j)]
            dp[j - 1, i - 1] = dpcluster.distance[(i, j)]
        cls.append(dpcluster.core[i])
    cls.append(dpcluster.core[dpcluster.num])
    cls = np.array(cls, dtype = np.float32)
    '''
    with open(r'./dpcluster.txt', 'w') as outfile:
        outfile.write('\n'.join(list(map(str, cls))))
        outfile.close()
    '''
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(max_iter=200, eps=1e-4, n_init=1, dissimilarity="precomputed")
    dp_mds = mds.fit_transform(dp)
    plot_scatter_diagram(1, dp_mds[:, 0], dp_mds[:, 1], title='cluster', cluster = cls)


if __name__ == '__main__':
    '''
    # test plot scatter diagram
    x =   np.array([1, 2, 3, 4, 5, 6, 7, 8, 7, 7])
    y =   np.array([2, 3, 4, 5, 6, 2, 4, 8, 5, 6])
    cls = np.array([1, 4, 2, 3, 5, -1, -1, 6, 6, 6])
    plot_scatter_diagram(0, x, y, cluster = cls)
    '''
    dpcluster = DensityPeakCluster()
    # distance, rho, delta, nearest_neighbor, num, dc = dpcluster.density_and_distance(r'./data/example_distances.data')
    # plot_rho_delta(rho, delta)
    dpcluster.cluster(r'./data/example_distances.data', 20, 0.15)
    plot_cluster(dpcluster)
    
