# Created by Tristan Bester.
import numpy as np



class KMeans(object):
    '''
    K-means is a clustering algorithm that functions to partition the given data
    into the specified number of clusters. Each instance is classified as belonging
    to the cluster with the nearest mean.
    
    Args:
        n_clusters (int): The number of clusters to partition the data into.
        n_iters (int): The number of times the K-means algorithm will be re-
                       initialized with random cluster centroids.
    
    Attributes:
        n_clusters (int): The number of clusters to partition the data into.
        n_iters (int): The number of times the K-means algorithm will be re-
                       initialized with random cluster centroids.
        centroids (numpy.ndarray): The optimal cluster centroids. The mean
                                   of each cluster. 
    '''
    def __init__(self, n_clusters=5, n_iters=10):
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        
        
    def __euclidean_distance(self, x1, x2):
        '''Calculate the Euclidean distance between the given vectors.'''
        diff = np.sum((x1-x2)**2)
        diff = np.sqrt(diff)
        return diff
        
    
    def fit(self, X):
        '''Calculate the mean of each of the clusters.'''
        best = np.inf
        self.centroids = None
        
        for itr in range(self.n_iters):
            centroid_idx = np.random.choice(range(X.shape[0]), size=self.n_clusters, replace=False)
            centroids = X[centroid_idx]
            c = []
            
            # Calculate index of closest cluster centroid for each instance.
            for x in X:
                idx = -1
                min_dist = np.inf
                for i,j in enumerate(centroids):
                    dist = self.__euclidean_distance(x, j)
                    if dist < min_dist:
                        idx = i
                        min_dist = dist
                c.append(idx)
            
            # Shift the cluster centroids.
            means = np.zeros((self.n_clusters, X.shape[1]))
            for x, clstr in zip(X, c):
                means[clstr] += x
    
            counts = np.bincount(c)
            for i,x in enumerate(counts):
                means[i] = means[i]/x
            centroids = means
            
            # Calculate the value of the cost function.
            cost = sum([self.__euclidean_distance(i, centroids[c[idx]]) for idx, i in enumerate(X)])
            cost /=  float(X.shape[0])
            
            # Store best clusters.
            if cost < best:
                best = cost
                self.centroids = centroids
        
        
    def predict(self, X):
        '''Calculate the cluster to which a given instance belongs.'''
        cluster = -1
        min_dist = np.inf
        for idx,i in enumerate(self.centroids):
            dist = self.__euclidean_distance(X, i)
            if dist < min_dist:
                min_dist = dist
                cluster = idx
        return cluster