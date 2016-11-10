# DensityPeakCluster

A implementation for 'Clustering by fast search and find of density peaks' in science 2014.  

## Usage
0. The input file should be distance metrix beteen points. If you have data of points vector, `distance.py` may be helpful.
1. run `python plot.py input_filename` and you will get the decison graph (of rho and delta).
2. Close (save it if you want) the graph and input the density_threshold(rho) and distance_threshold(delta) according to the decision graph, two float numbers separated by a blank space.
3. The cluster result will be write into file `dpcluster.txt` with two columns, points index in column 1 while
 cluster center in column 2 (-1 for halo).
4. A graph of cluster result will be plotted meanwhile.

## Reference
- [Clustering by fast search and find of density peaks](http://www.sciencemag.org/content/344/6191/1492.full)
