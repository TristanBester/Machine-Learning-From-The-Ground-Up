# Created by Tristan Bester
import numpy as np

class PolynomialFeaturesForDegree():
    '''
    Calculates the polynomial features of a dataset for a given degree.
    
    Args: 
        degree (int): The degree of the polynomial features to be calculated.
    
    Attributes:
        degree (int): The degree of the polynomial features to be calculated.
        options (list): A list containing n-tuples specifying the exponent of each
                        feature in the dataset for each term in the polynomial
                        expansion.
    '''
    
    def __init__(self, degree):
        self.degree = degree
        self.options = []
        
        
    def display_terms(self):
        '''
        If a training instance has three features, i.e. [a,b,c], and (0,0,2) is 
        displayed this indicates that a^0 * b^0 * c^2 is a term in the polynomial
        expansion of the dataset.
        '''
        
        print('Exponents of features at each index:')
        for i in sorted(self.options):
            print(i)
    
    
    def get_terms(self, features):
        m = features.shape[1]
        
        # Determines the number of possible combinations with a summation of m 
        # variables raised to power n.
        num = (self.degree+1)**m
        ls = [0] * m 
        
        # Calculate all possible combinations of the features in the dataset.
        for x in range(1,num):
            for idx in range(len(ls)-1, -1, -1):
                if ls[idx] == 0:
                    ls[idx] += 1
                    break
                elif ls[idx] % self.degree == 0:
                    ls[idx] = 0
                    if ls[idx - 1] != self.degree:
                        ls[idx-1] += 1
                        break
                    else:
                        deficit = -1
                        stop = False
                        while not stop and (idx + deficit) > -1:
                            if ls[idx + deficit] != self.degree:
                                ls[idx + deficit] += 1
                                stop = True
                            else:
                                ls[idx + deficit] = 0
                                deficit -= 1
                        break
                elif idx == (len(ls) - 1):
                    ls[idx] += 1
                    break
            
            # This is the Kronecker Delta Function used to eliminate terms that
            # do not form part of the polynomial expansion.
            if sum(ls) == self.degree:
                self.options.append(tuple(ls))
        
    
    def get_polynomial_features(self, features):
        # Calculate the terms in the polynomial expansion.
        self.get_terms(features)
        
        poly_features  = np.empty((len(features),0))
        feature_ls = []
        
        # Add vector of ones for degree zero.
        for i in range(features.shape[1]):
            feature_ls.append(features[:,i].reshape(-1,1))
        
        # Add the polynomial features to the dataset.
        for i,tup in enumerate(self.options):
            vector = np.full((len(features), 1), 1)
            
            # Calculate polynomial feature.
            for feature_idx, exp in enumerate(tup):
                vector *= feature_ls[feature_idx] ** exp 
            
            # Append polynomial feature to the dataset.
            poly_features = np.c_[poly_features,vector]
                
        return poly_features



class PolynomialFeatures():
    '''
    Calculates the polynomial features for all degrees up to and including the
    specified degree.
    
    Args:
        degree (int): The highest degree polynomial features to be calculated.
        
    Attributes:
        degree (int): The highest degree polynomial features to be calculated.
        transformers (list): A list containing one transformer for each degree
                            for which polynomial features are to be calculated.
    '''
    
    def __init__(self, degree):
        self.degree = degree
        
    def fit(self):
        self.transformers = []
        # Add one transformer for each degree.
        for i in range(1, self.degree + 1):
            self.transformers.append(PolynomialFeaturesForDegree(i))
        
    def transform(self,dataset):
        try:
            temp = np.ones((len(dataset),1))
            # Calculate and add polynomial features for each degree to the dataset.
            for t in self.transformers:
                features_for_deg = t.get_polynomial_features(dataset)
                temp = np.c_[temp, features_for_deg]
            return temp
        except AttributeError:
            print('Transformer not fitted, call \'fit\' with appropriate arguments before using transformer.')
        

                    
                    
                    
                    