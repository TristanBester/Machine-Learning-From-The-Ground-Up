# Created by Tristan Bester.
import numpy as np



class KNN(object):
    ''' 
    Super class for the KNeighboursClassifier and KNeighboursRegressor classes
    defined below.
    
    Args:
        n_neighbours (int): The number of nearest neighbours to be used to 
                            calculate the target value of an instance.
        
    Attributes:
        n_neighbours (int): The number of nearest neighbours to be used to 
                            calculate the target value of an instance.
        train_set (numpy.ndarray): A dataset containing the feature values
                                   as well as the target values of all of the
                                   instances in the training set.
    '''  
    def __init__(self, n_neighbours=5):
        self.n_neighbours = n_neighbours
        
    
    def __euclidean_distance(self, x1, x2):
        '''Calculate the Euclidean distance between the given vectors.'''
        diff = np.sum((x1-x2)**2)
        diff = np.sqrt(diff)
        return diff
    
    
    def neighbours(self, sample):
        '''Find the k-nearest neighbours of the given instance.'''
        dists = [(idx, self.__euclidean_distance(x, sample)) for idx,x in enumerate(self.train_set[:, :-1])]
        sorted_dists = sorted(dists, key=lambda tup: tup[1])
        neighbour_idx = [sorted_dists[i][0] for i in range(self.n_neighbours)]
        return self.train_set[neighbour_idx]
    
    
    def fit(self, X, y):
        '''Store to training data to be used when finding nearest neighbours.'''
        self.train_set = np.c_[X,y]
    
    
     
class KNeighboursClassifier(KNN):
    '''
    A K-nearest neighbours classifier makes predictions based on the target
    values of the k-nearest instances in the feature space. The predicted 
    class of a given instance is the modal class of the k-nearest instances
    in feature space.
    
    Args:
        n_neighbours (int): The number of nearest neighbours to be used to 
                            calculate the target value of an instance.
        
    Attributes:
        n_neighbours (int): The number of nearest neighbours to be used to 
                            calculate the target value of an instance.
        train_set (numpy.ndarray): A dataset containing the feature values
                                   as well as the target values of all of the
                                   instances in the training set.
    '''
    def predict_proba(self, sample):
        '''Predict the class probabilities of the given instance.'''
        knn = self.neighbours(sample)
        y = knn[:, -1].astype(int)
        counts = np.bincount(y)
        counts = counts/float(sum(counts))
        return counts
    
    
    def predict(self, X):
        '''Predict the class of the given instance.'''
        return np.argmax(self.predict_proba(X))
    
    
    
class KNeighboursRegressor(KNN):
    '''
    A K-nearest neighbours regressor makes predictions based on the target
    values of the k-nearest instances in the feature space. The predicted 
    target value of a given instance is the average target value of the k-nearest 
    instances in the feature space.
    
    Args:
        n_neighbours (int): The number of nearest neighbours to be used to 
                            calculate the target value of an instance.
        
    Attributes:
        n_neighbours (int): The number of nearest neighbours to be used to 
                            calculate the target value of an instance.
        train_set (numpy.ndarray): A dataset containing the feature values
                                   as well as the target values of all of the
                                   instances in the training set.
    '''
    def predict(self, X):
        '''Predict the target value of the given instance.'''
        knn = self.neighbours(X)
        y = knn[:, -1]
        return y.mean()