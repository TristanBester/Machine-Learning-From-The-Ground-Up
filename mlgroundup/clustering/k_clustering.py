# Created by Tristan Bester.
import numpy as np



class KClustering(object):
    '''
    Super class for the KMeans and KMedians classes defined below.
    
    Args:
        algorithm (str): The clustering algorithm to be used.
        n_clusters (int): The number of clusters to partition the data into.
        n_iters (int): The number of times the clustering algorithm will be re-
                       initialized with random cluster centroids.
    
    Attributes:
        algorithm (str): The clustering algorithm used.
        n_clusters (int): The number of clusters to partition the data into.
        n_iters (int): The number of times the clustering algorithm will be re-
                       initialized with random cluster centroids.
        centroids (numpy.ndarray): The optimal cluster centroids.
    '''
    def __init__(self, algorithm='means',  n_clusters=5, n_iters=10):
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        
        
    def __euclidean_distance(self, x1, x2):
        '''Calculate the Euclidean distance between the given vectors.'''
        diff = np.sum((x1-x2)**2)
        diff = np.sqrt(diff)
        return diff
    
    
    def __manhattan_distance(self, x1, x2):
        '''Calculate the Manhattan distance between the given vectors.'''
        diff = abs(x1-x2)
        return sum(diff)
    
    
    def __distance(self, x1, x2):
        '''Ensure appropriate distance metric is used.'''
        if self.algorithm == 'means':
            return self.__euclidean_distance(x1,x2)
        else:
            return self.__manhattan_distance(x1,x2)
    
    
    def __assign_to_clusters(self, X, centroids):
        '''Assign each instance in training set to a cluster.'''
        c = []
        for x in X:
                idx = -1
                min_dist = np.inf
                for i,j in enumerate(centroids):
                    dist = self.__distance(x, j)
                    if dist < min_dist:
                        idx = i
                        min_dist = dist
                c.append(idx)
        return c
    
    
    def __shift_centroids(self, X, c):
        '''Shift the cluster centroids.'''
        clusters = {}
        for i in set(c):
            clusters[i] = []
           
        for x,clster in zip(X, c):
            clusters[clster].append(x)
        
        if self.algorithm == 'means':
            for i,x in clusters.items():
                clusters[i] = (sum(x))/len(x)
            return clusters
        else:
            medians = []
            for x in clusters.values():
                temp =  x[0]
                for i in x[1:]:
                    temp = np.c_[temp,i]
                if temp.ndim == 1:
                    temp = temp.reshape(-1,len(temp))
                temp = np.median(temp, axis=1)   
                medians.append(temp)
            return medians
    
    
    def __calculate_cost(self, X, c, centroids):
        '''Calculate cost function value.'''
        cost = sum([self.__distance(i, centroids[c[idx]]) for idx, i in enumerate(X)])
        cost /= float(X.shape[0])
        return cost
            
            
    def fit(self, X):
        '''Calculate optimal cluster centroids.'''
        best = np.inf
        self.centroids = None
        
        for itr in range(self.n_iters):
            centroid_idx = np.random.choice(range(X.shape[0]), size=self.n_clusters, replace=False)
            centroids = X[centroid_idx]
            
            # Calculate index of closest cluster centroid for each instance.
            c = self.__assign_to_clusters(X, centroids)
        
            # Shift the cluster centroids.
            centroids = self.__shift_centroids(X, c)
            
            # Calculate the value of the cost function.
            cost = self.__calculate_cost(X, c, centroids)
            
            # Store best cluster centroids.
            if cost < best:
                best = cost
                self.centroids = centroids
        
        if self.algorithm == 'means':
            self.centroids = list(self.centroids.values())
            

    def predict(self, X):
        '''Calculate the cluster to which a given instance belongs.'''
        cluster = -1
        min_dist = np.inf
        for idx,i in enumerate(self.centroids):
            
            dist = self.__distance(X, i)
            if dist < min_dist:
                min_dist = dist
                cluster = idx
        return cluster
    


class KMeans(KClustering):
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
        super().__init__(algorithm='means', n_clusters=n_clusters, n_iters=n_iters)
        
    

class KMedians(KClustering):
    '''
    K-medians is a clustering algorithm that functions to partition the given data
    into the specified number of clusters. Each instance is classified as belonging
    to the cluster with the nearest median.
    
    Args:
        n_clusters (int): The number of clusters to partition the data into.
        n_iters (int): The number of times the K-medians algorithm will be re-
                       initialized with random cluster centroids.
    
    Attributes:
        n_clusters (int): The number of clusters to partition the data into.
        n_iters (int): The number of times the K-medians algorithm will be re-
                       initialized with random cluster centroids.
        centroids (numpy.ndarray): The optimal cluster centroids. The median
                                   of each cluster.
    '''
    def __init__(self, n_clusters=3, n_iters=10):
        super().__init__(algorithm='median', n_clusters=n_clusters, n_iters=n_iters)