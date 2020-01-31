# Created by Tristan Bester.
import numpy as np



class PCA(object):
    '''
    Principal Component Analysis is a dimensionality reduction technique. The axis
    (one-dimensional hyperplane) that accounts for the maximum variance of the 
    data is calculated then a second axis, orthogonal to the first, that accounts
    for the maximum remaining variance of the data is calculated. This process of
    repeatedly calculating the orthogonal axis that accounts for the largest
    amount of the remaining variance is repeated until an axis has been calculated
    for each dimension. These axes are the principal components of the data. The 
    data can then be projected onto the selected number of principal components, 
    with the data being projected onto the axes that account for more variance first.
    
    Args:
        n_components: The number of principal components the data will be projected
                      onto. This is the number of dimensions of the transformed dataset.
    
    Attributes:
        n_components: The number of principal components the data will be projected
                      onto. This is the number of dimensions of the transformed dataset.
        components: The principal components of the data.
    '''
    def __init__(self, n_components):
        self.n_componets = n_components
        
        
    def center(self, X):
        '''Center the data around the origin.'''
        mean  = X.mean(axis=0)
        return X - mean
        
    
    def fit_transform(self, X):
        '''Calculate the principal components of the data and transform the dataset.'''
        X_centered = self.center(X)
        
        # Use Singular Value Decomposition to calculate the principal components
        # of the data.
        U, sigma, V = np.linalg.svd(X_centered)
        self.components = V
        
        # Extract the selected number of principal components.
        W = V.T[:, :self.n_componets]
        
        # Project the data onto the selected number of principal components.
        return X_centered @ W