# Created by Tristan Bester.
from mlgroundup.supervised import DecisionTreeRegressor
import numpy as np


    
class Node(object):
    '''
    Objects of this class serve as single nodes in the decision trees created by
    the classes defined below. The objects function to store the necessary
    information required by a node in a decision tree.

    Args:
        decision (tuple): Stores the feature index and threshold value used for
                          the decision at the node. Format: (threshold, feature index).
        prediction (float): The predicted log(odds) for the samples at this node.

    Attributes:
        decision (tuple): The information used to make the decision at the node.
        prediction (float): The predicted log(odds) value for samples at this node.
        left (Node): The left child node of the current node.
        right (Node): The right child node of the current node.
    '''
    def __init__(self, decision, prediction=None):
        self.decision = decision
        self.prediction = prediction
        self.left = None
        self.right = None
        


class RegressionTree(object):
    '''
    I have opted to recreate the decision tree class to suit the requirements of
    a gradient boosting classifier. I have done this as the method used to calculate
    the output value of the terminal regions of the tree differs greatly from that
    of a standard regression decision tree. Thus, in order to circumvent the confusion
    of users analysing the DecisionTreeRegressor class included in this repository
    I have implemented the required decision tree class below.
    
    Args:
        max_depth (int): The maximum depth of the decision tree that can be created.
        min_samples_split (int): The minimum number of samples required at a node
                                 for the samples to be split into smaller subsets.
        min_samples_leaf (int): The minimum number of samples required to be at
                                each leaf node.

    Attributes:
        root (Node): The root node of the decision tree.
        max_depth (int): The maximum depth of the decision tree that can be created.
        min_samples_split (int): The minimum number of samples required at a node
                                 for the samples to be split into smaller subsets.
        min_samples_leaf (int): The minimum number of samples required to be at
                                each leaf node.
    '''
    def __init__(self, max_depth=np.inf, min_samples_split=2, min_samples_leaf=1):
        self.root = None      
        
        if max_depth > 0:
            self.max_depth = max_depth
        else:
            raise AttributeError('Invalid max_depth value, max_depth must be greater than zero.')

        if min_samples_split > 1:
            self.min_samples_split = min_samples_split
        else:
            raise AttributeError('Invalid min_samples_split value, min_samples_split must be greater than one.')

        if min_samples_leaf > 0:
            self.min_samples_leaf = min_samples_leaf
        else:
            raise AttributeError('Invalid min_samples_leaf value, min_samples_leaf must be greater than zero.')
        
    
    def get_split_points(self, X):
        '''Calculate the splitting points in the data.'''
        cols = []
        split_pts = []
        
        for i,x in enumerate(np.sort(X.T)):
            x = np.unique(x)
            for j in range(x.shape[0]-1):
                split = (x[j] + x[j+1])/2.0
                split_pts.append(split)
                cols.append(i)
        return split_pts, cols
    
    
    def split(self, split_val, col, X,y):
        '''Split the given dataset based on the given feature index and threshold value.'''
        lower = []
        upper = []
        y_lower = []
        y_upper = []
        
        for i in range(X.shape[0]):
            if X[i, col] < split_val: 
                lower.append(X[i])
                y_lower.append(y[i])
            else:
                upper.append(X[i])
                y_upper.append(y[i])
                
        return np.array(lower), np.array(upper), np.array(y_lower), np.array(y_upper)
            
    
    def MSE(self, y):
        '''
        Calculate the mean squared error from predicting the mean target value
        of the instances at the node.
        '''
        y_hat = np.mean(y)
        mse = ((y_hat - y)**2).sum()
        return mse
    
    
    def CART(self, X, y):
        '''The CART algorithm for building decision trees.'''
        splits, cols = self.get_split_points(X)
        splits = np.c_[splits,cols]
        
        best = np.inf
        
        for split_pt in splits:
            lower, upper, y_lower, y_upper = self.split(split_pt[0], int(split_pt[1]), X, y)
            mse_lower = self.MSE(y_lower[:, 0])
            mse_upper = self.MSE(y_upper[:, 0])
            m = float(y_lower.shape[0] + y_upper.shape[0])
            cost = (len(y_lower)/m) * mse_lower + (len(y_upper)/m) * mse_upper
            
            if cost <= best:
                best = cost
                right  = lower
                y_right = y_lower
                left = upper
                y_left = y_upper
                best_split = split_pt
                        
        return left, y_left, right, y_right, best_split
                
    
    def __fit(self, subtree, X, y, curr_depth):
        '''
        If the dataset does not already have a mean squared error of zero and the
        regularization parameters have not been satisfied, create and add a node
        to the decision tree that predicts a more accurate target value for the
        instances in the given dataset. Then recursively call the __fit method to
        create the child nodes.
        '''
        if (
                curr_depth > self.max_depth or
                len(X) < self.min_samples_split or
                len(X) < self.min_samples_leaf or
                self.MSE(y[:, 0]) == 0
            ):
            residuals = y[:, 0]
            predictions = y[:, 1]
            numerator = residuals.sum()
            denominator = sum([x*(1-x) for x in predictions])
            
            # Prevent math error - Undefined. 
            if denominator == 0:
                denominator = 0.01
            output_value = numerator/denominator
            return Node(decision=None, prediction=output_value)
        else:
            left, y_left, right, y_right, best_split = self.CART(X, y)
            subtree = Node(decision=best_split)
            subtree.left = self.__fit(subtree.left, left, y_left, curr_depth+1)
            subtree.right = self.__fit(subtree.right, right, y_right, curr_depth+1)
            return subtree
            
        
    def fit(self, X, y):
        '''
        Build the decision tree by recursively calling the __fit method until the
        correct target value has been predicted for all of the training instances
        or the specified regularization criteria have been satisfied.
        '''
        self.root = self.__fit(self.root, X, y, 1)
        
        
    def __predict(self, subtree, val):
        '''Predict the class probabilities of an instance.'''
        if val.ndim == 0:
            val = np.array([val])
    
        if subtree.decision is None:
            return subtree.prediction
        elif val[int(subtree.decision[1])] > subtree.decision[0]:
            return self.__predict(subtree.left, val)
        else:
            return self.__predict(subtree.right, val)
    
    
    def predict(self, X):
        '''Predict the class of the given instance.'''
        return self.__predict(self.root, X)



class GradientBoostingClassifier(object):
    '''
    A GradientBoostingClassifier is an ensemble model in which estimators are
    trained sequentially with each successive estimator trained to predict the 
    pseudo-residuals of all of the estimators trained prior to it. Once the model
    has been trained the predictions of all of the estimators in the ensemble
    are aggregated to predict the class of a given instance.
    
    Args:
        n_estimators (int): The number of estimators to be used in the ensemble.
        learning_rate (float): The learning rate of the model.
        max_depth (int): The maximum depth of the decision trees in the ensemble.
        min_samples_split (int): The minimum number of samples required at a node
                                 for the samples to be split into smaller subsets.
        min_samples_leaf (int): The minimum number of samples required to be at
                                each leaf node in a decision tree.
                                
    Attributes:
        n_estimators (int): The number of estimators used in the ensemble.
        learning_rate (float): The learning rate of the model.
        max_depth (int): The maximum depth of the decision trees in the ensemble.
        min_samples_split (int): The minimum number of samples required at a node
                                 for the samples to be split into smaller subsets.
        min_samples_leaf (int): The minimum number of samples required to be at
                                each leaf node in a decision tree.    
        param_dict (dict): A dictionary storing the parameters required to create 
                           the decision trees to be used in the ensemble.
    '''
    def __init__(self, n_estimators=3,
                 learning_rate=0.1,
                 max_depth=2,
                 min_samples_split=2,
                 min_samples_leaf=1):
        self.max_depth = max_depth
        self.n_estimators=n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.param_dict = {'max_depth':self.max_depth, 'min_samples_split':
                           self.min_samples_split, 'min_samples_leaf':
                           self.min_samples_leaf}
        
           
    def __init_leaf(self, y):
        '''Return the value the model is initialized to predict.'''
        y = y.astype(int)
        class_one_count = np.bincount(y.flatten())[1]
        proba = float(class_one_count)/y.shape[0]
        log_odds = np.log((proba)/(1-proba))
        
        # Prevent math error - Undefined.
        if log_odds == 0:
            log_odds = 0.01
        return log_odds
    
    
    def predict_proba(self, X):
        '''Predict the class probabilities of the given instance.'''
        log_odds = self.estimators[0]
        for i in self.estimators[1:]:
            log_odds += self.learning_rate * i.predict(X)
        probability = (np.exp(log_odds))/(1 + np.exp(log_odds))
        return probability
    
    
    def predict(self, X):
        '''Predict the class of the given instance.'''
        return int(self.predict_proba(X) > 0.5)
    
        
    def fit(self, X, y):
        '''Fit the model to the given training set.'''
        leaf = self.__init_leaf(y)
        self.estimators = [leaf]
        residuals = y - leaf
        
        # Create a matrix containing the residuals and the associated probabilities
        # to be used to train a decision tree.
        y2 = np.c_[residuals, [leaf] * len(residuals)]
        
        for i in range(self.n_estimators - 1):
            tree = RegressionTree(**self.param_dict)
            tree.fit(X, y2)
            self.estimators.append(tree)
            
            # Update the predictions and residuals.
            proba = [self.predict_proba(x) for x in X]
            proba = (np.array(proba).reshape(-1,1))
            residuals = y - proba
            y2 = np.c_[residuals, proba]
            


class GradientBoostingRegressor(object):
    '''
    A GradientBoostingRegressor is an ensemble model in which estimators are
    trained sequentially with each successive estimator trained to predict the 
    pseudo-residuals of all of the estimators trained prior to it. Once the model
    has been trained the predictions of all of the estimators in the ensemble 
    are aggregated to predict the target value of a given instance.
    
    Args:
        n_estimators (int): The number of estimators to be used in the ensemble.
        learning_rate (float): The learning rate of the model.
        max_depth (int): The maximum depth of the decision trees in the ensemble.
        min_samples_split (int): The minimum number of samples required at a node
                                 for the samples to be split into smaller subsets.
        min_samples_leaf (int): The minimum number of samples required to be at
                                each leaf node in a decision tree.
   
    Attributes:
        n_estimators (int): The number of estimators used in the ensemble.
        learning_rate (float): The learning rate of the model.
        max_depth (int): The maximum depth of the decision trees in the ensemble.
        min_samples_split (int): The minimum number of samples required at a node
                                 for the samples to be split into smaller subsets.
        min_samples_leaf (int): The minimum number of samples required to be at
                                each leaf node in a decision tree.
        param_dict (dict): A dictionary storing the parameters required to create 
                           the decision trees to be used in the ensemble.
    '''                     
    def __init__(self,
                 n_estimators=2,
                 learning_rate=0.1,
                 max_depth=3,
                 min_samples_split=2,
                 min_samples_leaf=1
                 ):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.param_dict = {'max_depth':self.max_depth, 'min_samples_split':
                           self.min_samples_split, 'min_samples_leaf':
                           self.min_samples_leaf}
        
            
    def __calculate_residuals(self,X,y):
        '''Calculate the pseudo-residuals.'''
        residuals = np.zeros(y.shape)
        for i,x in enumerate(X):
            pred = self.predict(x)
            residuals[i] = y[i] -  pred
        return residuals
        
                   
    def fit(self, X, y):
        '''Fit the model to the given training set.'''
        # Calculate the value the model is initialized to predict.
        leaf = y.mean()
        self.estimators = [leaf]
        
        for i in range(self.n_estimators-1):
            residuals = self.__calculate_residuals(X,y)
            tree = DecisionTreeRegressor(**self.param_dict)
            tree.fit(X,residuals)
            self.estimators.append(tree)
        
        
    def predict(self, X):
        '''Predict the target value of the given instance.'''
        pred = self.estimators[0]
        pred += sum([self.learning_rate * estimator.predict(X) for estimator in self.estimators[1:]])
        return pred