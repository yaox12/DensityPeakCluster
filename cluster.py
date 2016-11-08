#-*- coding: utf-8 -*-
import sys
import math
import numpy as np


def load_data(distance_file):
    '''
    Load distance from data file:
    column1: index1, column2: index2, column3: distance

    Return: distance dict, max distance, min distance, num of points
    '''
    distance = {}
    min_dis, max_dis = sys.float_info.max, 0.0
    num = 0
    with open(distance_file, 'r', encoding = 'utf-8') as infile:
        for line in infile:
            content = line.strip().split(' ')
            assert(len(content) == 3)
            idx1, idx2, dis = int(content[0]), int(content[1]), float(content[2])
            num = max(num, idx1, idx2)
            min_dis = min(min_dis, dis)
            max_dis = max(max_dis, dis)
            distance[(idx1, idx2)] = dis
            distance[(idx2, idx1)] = dis
        for i in range(1, num + 1):
            distance[(i, i)] = 0.0
        infile.close()
    
    return distance, num, max_dis, min_dis 


def auto_select_dc(distance, num, max_dis, min_dis):
    '''
    Auto select the dc so that the average number of neighbors is around 1 to 2 percent
    of the total number of points in the data set
    '''
    dc = (max_dis + min_dis) / 2
    
    while True:
        neighbor_percent = sum([1 for value in distance.values() if value < dc]) / num ** 2
        if neighbor_percent >= 0.01 and neighbor_percent <= 0.02:
            break
        if neighbor_percent < 0.01:
            min_dis = dc
        elif neighbor_percent > 0.02:
            max_dis = dc
        dc = (max_dis + min_dis) / 2
        if max_dis - min_dis < 0.0001:
            break

    return dc


def local_density(distance, num, dc, gauss = False, cutoff = True):
    '''
    Compute all points' local density
    Return: local density vector of points that index from 1
    '''
    assert gauss and cutoff == False and gauss or cutoff == True
    gauss_func = lambda dij, dc : math.exp(- (dij / dc) ** 2)
    cutoff_func = lambda dij, dc : 1 if dij < dc else 0
    func = gauss_func if gauss else cutoff_func
    rho = [-1] + [0] * num
    for i in range(1, num):
        for j in range(i + 1, num + 1):
            rho[i] += func(distance[(i, j)], dc)
            rho[j] += func(distance[(j, i)], dc)

    return np.array(rho, np.float32)


def min_distance(distance, num, max_dis, rho):
    '''
    Compute all points' min distance to a higher local density point
    Return: min distance vector, nearest neighbor vector
    '''
    sorted_rho_idx = np.argsort(-rho)
    delta = [0.0] + [max_dis] * num
    nearest_neighbor = [0] * (num + 1)
    delta[sorted_rho_idx[0]] = -1.0
    for i in range(1, num):
        idx_i = sorted_rho_idx[i]
        for j  in range(0, i):
            idx_j = sorted_rho_idx[j]
            if distance[(idx_i, idx_j)] < delta[idx_i]:
                delta[idx_i] = distance[(idx_i, idx_j)]
                nearest_neighbor[idx_i] = idx_j

    delta[sorted_rho_idx[0]] = max(delta)
    return np.array(delta, np.float32), np.array(nearest_neighbor, np.int)


class DensityPeakCluster(object):

    def local_density(self, distance_file, dc = None):
        distance, num, max_dis, min_dis = load_data(distance_file)
        if dc == None:
            dc = auto_select_dc(distance, num, max_dis, min_dis)
        rho = local_density(distance, num, dc)

        return distance, rho, num, max_dis, min_dis, dc

    def cluster(self, distance_file, density_threshold, distance_threshold, dc = None):
        distance, rho, num, max_dis, min_dis, dc = self.local_density(distance_file, dc = dc)
        delta, nearest_neighbor = min_distance(distance, num, max_dis, rho)
        cluster = [-1] * (num + 1)
        center = []

        for i in range(1, num + 1):
            if rho[i] >= density_threshold and delta[i] >= distance_threshold:
                center.append(i)
                cluster[i] = i
        
        #assignation
        sorted_rho_idx = np.argsort(-rho)
        for i in range(num):
            idx = sorted_rho_idx[i]
            if idx in center:
                continue
            cluster[idx] = cluster[nearest_neighbor[idx]]

        #halo
        halo = cluster[:]
        if len(center) > 1:
            rho_b = [0.0] * (num + 1)
            for i in range(1, num):
                for j in range(i + 1, num + 1):
                    if cluster[i] != cluster[j] and distance[(i, j)] < dc:
                        rho_avg = (rho[i] + rho[j]) / 2
                        rho_b[cluster[i]] = max(rho_b[cluster[i]], rho_avg)
                        rho_b[cluster[j]] = max(rho_b[cluster[j]], rho_avg)

            for i in range(1, num + 1):
                if rho[i] > rho_b[cluster[i]]:
                    halo[i] = -1

        for i in range(len(center)):
            n_ele, n_halo = 0, 0
            for j in range(1, num + 1):
                if cluster[j] == center[i]:
                    n_ele += 1
                if halo[j] == center[i]:
                    n_halo += 1
            n_core = n_ele - n_halo
            print("Cluster %d: Center: %d, Element: %d, Core: %d, Halo: %d\n" % (i + 1, center[i], n_ele, n_core, n_halo))

        self.cluster = cluster
        self.center = center
        self.distance = distance
        self.num = num

        return rho, delta, nearest_neighbor
        

if __name__ == '__main__':
    dpcluster = DensityPeakCluster()
    dpcluster.cluster(r'./data/example_distances.data', 20, 0.1)
