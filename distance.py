#-*- coding: utf-8 -*-
import numpy as np

class Distance(object):

    '''
    transfer points vector to distance vector
    '''

    def __init__(self):
        self.vectors = []


    def load_points(self, filename):
        '''
        Load points from input file.
        Format: n-dimension vectors in one line, each dimension split by a blank space
        '''
        with open(filename, 'r', encoding = 'utf-8') as infile:
            for line in infile:
                self.vectors.append(np.array(list(map(float, line.strip().split(' '))), dtype = np.float32))
            infile.close()
        self.vectors = np.array(self.vectors, dtype = np.float32)


    def euclidean_distance(self, vec1, vec2):
        '''
        Return the Euclidean Distance of two vectors
        '''
        return np.linalg.norm(vec1 - vec2)


    def build_distance_file(self, filename):
        '''
        Calculate distance, save the result for cluster
        '''
        with open(filename, 'w', encoding = 'utf-8') as outfile:
            for i in range(len(self.vectors) - 1):
                for j in range(i, len(self.vectors)):
                    distance = self.euclidean_distance(self.vectors[i], self.vectors[j])
                    outfile.write('%d\t%d\t%f\n' % (i + 1, j + 1, distance))
            outfile.close()


if __name__ == '__main__':
    builder = Distance()
    builder.load_points(r'./data/test.data')
    builder.build_distance_file(r'./data/test.forcluster')
