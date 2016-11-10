# DensityPeakCluster

A python implementation for 'Clustering by fast search and find of density peaks' in science 2014.  

Python version: 3.5

## Usage
0. The input file should be distance metrix beteen points. If you have data of points vector, `distance.py` may be helpful.
1. Run `python plot.py input_filename` and you will get the decision graph (of rho and delta).
2. Close the graph (save it if you want) and input the density_threshold(rho) and distance_threshold(delta) according to the decision graph, two float numbers separated by a blank space.
3. The cluster result will be write into file `dpcluster.txt` with two columns, points index in column 1 while
 cluster center in column 2 correspondingly (-1 for halo).
4. A graph of cluster result will be plotted meanwhile.

## Reference
- [Clustering by fast search and find of density peaks](http://www.sciencemag.org/content/344/6191/1492.full)
