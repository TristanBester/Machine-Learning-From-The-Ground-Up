# Created by Tristan Bester.
from mlgroundup.supervised import DecisionTreeClassifier
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


class AdaBoost(object):
    '''
    An AdaBoost classifier is a binary classifier that trains estimators sequentially.
    An initial estimator is fitted to the training set, the following estimator
    is then fitted to a training set that places more emphasis on the instances
    misclassified by the previous estimator. This process of fitting estimators
    to correct the misclassifications of the previous estimator is repeated until
    the ensemble contains the desired number of estimators.
    The method of altering the training set is defined here:
    https://mitpress.mit.edu/books/boosting
    
    
    Args:
        base_estimator (object): The base estimator used in the ensemble.
        n_estimators (int): The number of estimators to be used in the ensemble.
        learning_rate (float): The learning rate.
    
    Attributes:
        base_estimator (object): The base estimator used in the ensemble.
        n_estimators (int): The number of estimators to be used in the ensemble.
        learning_rate (float): The learning rate.
        estimators (list): A list storing all of the estimators in the ensemble.
        influence (list): A list storing the estimator weight for each estimator
                          in the ensemble.
    '''
    def __init__(self, 
                 base_estimator=DecisionTreeClassifier(max_depth=2),
                 n_estimators=10,
                 learning_rate=1):
        # Default base estimator is a Decision Stump.
        self.base_estimator = base_estimator
        self.n_estimators =  n_estimators
        self.learning_rate = learning_rate
    
    
    def __get_estimator_params(self):
        '''Extract the attributes of the base estimator to be used in the ensemble.'''
        dic = self.base_estimator.__dict__
        args = self.base_estimator.__init__.__code__.co_varnames
        params = {}
        
        for i in dic:
            if i in args:
                params[i] = dic[i]
        return params
    
    
    def __get_say(self, incorrect):
        '''Calculate the estimator weight.'''
        error = incorrect[:, -1].sum()
        
        # Prevent math error - undefined.
        if error == 1:
            error = 0.999
        elif error == 0:
            error = 0.001
            
        say =  self.learning_rate * (np.log((1-error)/float(error)))
        return say
    
    
    def __mod_weights(self, X, outcomes, say):
        '''Alter the weights of the samples in the training set.'''
        for i in range(X.shape[0]):
            if outcomes[i]:
                X[i, -1] = X[i, -1]
            else:
                X[i, -1] = (X[i, -1] * np.exp(say))
        
        X[:,-1] /= float(X[:,-1].sum())
        return X
    
    
    def __order_weights(self, X):
        '''Change weights to sum of all weights with smaller indices.'''
        summation = 0
        for i,x in enumerate(X[:,-1]):
            summation += x
            X[i, -1] = summation
        return X
    
    
    def __mod_datasets(self,X,y):
        '''Resample the dataset to place more emphasis on misclassified instances.'''
        temp_X = np.zeros(X.shape)
        temp_y = np.zeros(y.shape)
        
        X = self.__order_weights(X)
        for i in range(X.shape[0]):
            val = np.random.rand()
            idx = 0
            while val > X[idx, -1]:
                idx += 1
            temp_X[i] = X[idx]
            temp_y[i] = y[idx]
            
        # Reset weights to all be equal.
        temp_X[:,-1] = (1/float(temp_X.shape[0]))
        return temp_X, temp_y
        
            
    def fit(self, X, y):
        '''Fit model to the training set.'''
        # Append sample weights as last column.
        X = np.c_[X, np.full((X.shape[0], 1), (1/float(X.shape[0])))]
        
        self.estimators = []
        self.influence = []
        params = self.__get_estimator_params()
        
        for i in range(self.n_estimators):
            estimator = self.base_estimator.__class__(**params)
            estimator.fit(X,y)
            preds = [estimator.predict(x) for x in X]
            incorrect = X[preds != y]
            say = self.__get_say(incorrect)
            X = self.__mod_weights(X, preds == y, say)
            X,y = self.__mod_datasets(X,y)
            self.estimators.append(estimator)
            self.influence.append(say)
     
        
    def predict(self, X):
        '''Predict the class of an instance.'''
        try:
            # As AdaBoost is a binary classifier.
            preds = np.array([0,0])
            for estimator, say in zip(self.estimators, self.influence):
                pred = estimator.predict(X)
                if pred:
                    preds[0] += say
                else:
                    preds[1] += say
            return np.argmax(preds)
        except AttributeError:
            print('Model not fitted. Call \'fit\' with appropriate arguments to resolve.')
