from supervised import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
          
class VotingClassifier(object):
    '''
    A VotingClassifier is an ensemble model that aggregates the predictions of 
    multiple base models and makes a final prediction based on either hard or 
    soft voting.
    
    Args:
        estimators (list): A list storing the base estimators of the ensemble.
        voting (str): The voting method to be used.
    
    Attributes:
        estimators (list): A list storing the base estimators of the ensemble.
        voting (str): The voting method used.
    '''
    def __init__(self, estimators, voting='hard'):
        self.estimators = estimators
        
        if voting == 'hard' or voting == 'soft':
            self.voting = voting
        else:
            raise AttributeError('Invalid \'voting\' argument, \'hard\' and \'soft\' voting available.')
    
    
    def fit(self, X, y):
        # Fit each of the base estimators to the training set.
        for estimator in self.estimators:
            estimator.fit(X,y)
    
    
    def predict(self,X):
        if self.voting == 'soft':
            try:
                preds = [estimator.predict_proba(X) for estimator in self.estimators]
                preds = sum(preds)
                return np.argmax(preds)
            except AttributeError:
                # Throw an error if any of the estimators do not support class probability
                # prediction.
                print('The list of specified estimators does not support soft voting, set voting to \'hard\' to resolve.')
        else:
            preds = [estimator.predict(X) for estimator in self.estimators]
            preds = np.bincount(preds)
            return np.argmax(preds)


   
class VotingRegressor(object):
    '''
    A VotingRegressor is an ensemble model that aggregates the predictions of 
    multiple base models and makes a final prediction. The final prediction is 
    equal to the mean predicted target value of all of the base estimators in 
    the ensemble.
    
    Args:
        estimators (list): A list storing the base estimators of the ensemble.
    
    Attributes:
        estimators (list): A list storing the base estimators of the ensemble.
    '''
    def __init__(self, estimators):
        self.estimators = estimators
      
        
    def fit(self, X, y):
        # Fit each of the base estimators to the training set.
        for estimator in self.estimators:
            estimator.fit(X,y)
            
            
    def predict(self, X):
        print(X)
        pred = 0
        for estimator in self.estimators:
            print(estimator.predict(X))
            pred += estimator.predict(X)
        return pred / float(len(self.estimators))



class Bagging(object):
    '''
    Super class for  the BaggingClassifier, BaggingRegressor, RandomForestClassifier,
    RandomForestRegressor, ExtraTreesClassifier and ExtraTreesRegressor classes.
    
    Args:
        base_estimator (object): The base estimator to be used in the ensemble.
        n_estimators (int): The number of estimators to be used in the ensemble.
        max_samples (float): The fraction of the total number of training instances
                             to be used to train each estimator in the ensemble.
        max_features (float): The fraction of the total number of features to be 
                              used to train each estimator in the ensemble.
        bootstrap (bool): Set to True to make use of bagging, set to False to make
                          use of pasting.
        bootstrap_features (bool): Set to True to make use of feature bagging.
    
    Attributes:
        base_estimator (object): The base estimator used in the ensemble.
        n_estimators (int): The number of estimators used in the ensemble.
        max_samples (float): The fraction of the total number of training instances
                             used to train each estimator in the ensemble.
        max_features (float): The fraction of the total number of features  
                              used to train each estimator in the ensemble.
        bootstrap (bool): True if bagging is used, False if pasting is used.
        bootstrap_features (bool): True if feature bagging is used.
        estimators (list): A list storing all of the base estimators in the
                           ensemble.
        oob_score (float): The out-of-bag score of the ensemble.
        oob_valid (bool): True, if all of the datasets used to train the base
                          estimators had at least one associated out-of-bag sample.
                          False, if the out-of-bag score is not valid, as at least
                          one of the datasets used to train a base estimator
                          did not have any associated out-of-bag samples.
        features (list): A list storing a list of the features used to train each
                         of the base estimators in the ensemble.
    '''
    def __init__(self, 
                 base_estimator=None, 
                 n_estimators=10, 
                 max_samples=1.0, 
                 max_features=1.0,
                 bootstrap=True, 
                 bootstrap_features=False):
        self.base_estimator = base_estimator
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.estimators = []
        
        if n_estimators > 0:
            self.n_estimators = n_estimators
        else:
            raise AttributeError('Invalid number of estimators. \'n_estimators\' argument not valid.')
        
        if max_samples > 0 and max_samples <= 1:
            self.max_samples = max_samples
        else:
            raise AttributeError('Invalid maximum number of samples. \'max_samples\'',
                                  'argument not valid, set fraction of total number of',
                                  'samples within interval: [0,1].')
            
        if max_features > 0 and max_features <= 1:
            self.max_features = max_features
        else:
            raise AttributeError('Invalid maximum number of features. \'max_features\'',
                                  'argument not valid, set fraction of total number of',
                                  'features within interval: [0,1].')
    
    
    def __get_estimator_params(self):
        '''Extract the attributes of the base estimator to be used in the ensemble.'''
        dic = self.base_estimator.__dict__
        args = self.base_estimator.__init__.__code__.co_varnames
        params = {}
        
        for i in dic:
            if i in args:
                params[i] = dic[i]
        return params


    def __get_features(self, X):
        '''Randomly sample the preset number of feature indices.'''
        n_features = int(self.max_features * X.shape[1])
        mx = X.shape[1]
        sample_idx = np.random.choice(mx, n_features, replace=self.bootstrap_features)
       
        return sample_idx
      
    
    def __get_datasets(self,X, y):
        '''Randomly sample the preset number of training instances.'''
        n_samples = int(self.max_samples * X.shape[0])
        mx = X.shape[0]
        sample_idx = np.random.choice(mx, n_samples, replace=self.bootstrap)
        oob_idx = [i for i in range(mx) if i not in sample_idx]
        
        return X[sample_idx], X[oob_idx], y[sample_idx], y[oob_idx]
    
    
    def fit(self, X, y):
        '''Fit each of the base estimators in the ensemble to the training set.'''
        if X.ndim == 1:
            X = X.reshape(-1,1)
        
        self.oob_score = 0
        self.features = []
        self.oob_valid = True
        
        # Get the parameters to be passed to the estimator's constructor.
        param_dict = self.__get_estimator_params()
        
        for i in range(self.n_estimators):
            # Get subset of features to be used.
            features = self.__get_features(X)
            
            # Get the subset of training instances to be used, as well as the 
            # out-of-bag instances.
            X_data, X_oob, y_data, y_oob = self.__get_datasets(X[:, features], y)
            
            # Create an estimator object.
            estimator = self.base_estimator.__class__(**param_dict)
            estimator.fit(X_data,y_data)
            
            # If the dataset used to train the estimator has associated oob
            # instances, calculate the estimators oob score, else set the
            # oob score to be invalid.
            if len(y_oob) > 0 and self.oob_valid:
                score = np.array([estimator.predict(x) for x in X_oob])
        
                score = (score == y_oob).sum()
                
                self.oob_score += score / float(len(y_oob))
            else:
                self.oob_valid = False
            
            self.features.append(features)
            self.estimators.append(estimator)
        
        if self.oob_valid:
            self.oob_score /= float(self.n_estimators)
    


class BaggingClassifier(Bagging):
    '''
    A BaggingClassifier is an ensemble model that aggregates the predictions of
    multiple base estimators in order to make a final prediction. The base estimators
    are trained on random subsets of the training data. The final prediction is 
    calculated making use of soft voting if possible, else hard voting is employed.
    
    Args:
        base_estimator (object): The base estimator to be used in the ensemble.
        n_estimators (int): The number of estimators to be used in the ensemble.
        max_samples (float): The fraction of the total number of training instances
                             to be used to train each estimator in the ensemble.
        max_features (float): The fraction of the total number of features to be 
                              used to train each estimator in the ensemble.
        bootstrap (bool): Set to True to make use of bagging, set to False to make
                          use of pasting.
        bootstrap_features (bool): Set to True to make use of feature bagging.
    
    Attributes:
        base_estimator (object): The base estimator used in the ensemble.
        n_estimators (int): The number of estimators used in the ensemble.
        max_samples (float): The fraction of the total number of training instances
                             used to train each estimator in the ensemble.
        max_features (float): The fraction of the total number of features  
                              used to train each estimator in the ensemble.
        bootstrap (bool): True if bagging is used, False if pasting is used.
        bootstrap_features (bool): True if feature bagging is used.
        estimators (list): A list storing all of the base estimators in the
                           ensemble.
        oob_score (float): The out-of-bag score of the ensemble.
        oob_valid (bool): True, if all of the datasets used to train the base
                          estimators had at least one associated out-of-bag sample.
                          False, if the out-of-bag score is not valid, as at least
                          one of the datasets used to train a base estimator
                          did not have any associated out-of-bag samples.
        features (list): A list storing a list of the features used to train each
                         of the base estimators in the ensemble.
    '''
    def predict(self, X):
        try:
            # Perform soft voting if possible.
            preds = [estimator.predict_proba(X[features]) for estimator, 
                     features in zip(self.estimators, self.features)]
            mx = -1
            for i in preds:
                if i.shape[0] > mx:
                    mx = i.shape[0]
            preds = [np.pad(x, (0, mx - x.shape[0]), 'constant') for x in preds]
            preds = sum(preds)
            return np.argmax(preds)
        except AttributeError:
            preds = [estimator.predict(X[features]) for estimator, 
                     features in zip(self.estimators, self.features)]
            preds = np.bincount(preds)
            return np.argmax(preds)
 
    
    
class BaggingRegressor(Bagging):
    '''
    A BaggingRegressor is an ensemble model that aggregates the predictions of 
    multiple base models and makes a final prediction. The final prediction is 
    equal to the mean predicted target value of all of the base estimators in 
    the ensemble. The base estimators are trained on random subsets of the training
    data.
    
    Args:
        base_estimator (object): The base estimator to be used in the ensemble.
        n_estimators (int): The number of estimators to be used in the ensemble.
        max_samples (float): The fraction of the total number of training instances
                             to be used to train each estimator in the ensemble.
        max_features (float): The fraction of the total number of features to be 
                              used to train each estimator in the ensemble.
        bootstrap (bool): Set to True to make use of bagging, set to False to make
                          use of pasting.
        bootstrap_features (bool): Set to True to make use of feature bagging.
    
    Attributes:
        base_estimator (object): The base estimator used in the ensemble.
        n_estimators (int): The number of estimators used in the ensemble.
        max_samples (float): The fraction of the total number of training instances
                             used to train each estimator in the ensemble.
        max_features (float): The fraction of the total number of features  
                              used to train each estimator in the ensemble.
        bootstrap (bool): True if bagging is used, False if pasting is used.
        bootstrap_features (bool): True if feature bagging is used.
        estimators (list): A list storing all of the base estimators in the
                           ensemble.
        oob_score (float): The out-of-bag score of the ensemble.
        oob_valid (bool): True, if all of the datasets used to train the base
                          estimators had at least one associated out-of-bag sample.
                          False, if the out-of-bag score is not valid, as at least
                          one of the datasets used to train a base estimator
                          did not have any associated out-of-bag samples.
        features (list): A list storing a list of the features used to train each
                         of the base estimators in the ensemble.
    '''
    def predict(self, X):
        if X.ndim == 0:
            X = np.array([X])
        preds = np.array([estimator.predict(X[features]) for estimator,features in zip(self.estimators, self.features)])
        return preds.mean()
               

            
class RandomForestClassifier(Bagging):
    '''
    A random forest classifier is an ensemble model that makes use of decision 
    trees as the base estimator in the ensemble. The model trains each of the 
    base estimators on a random subset of the training data. Each subset contains 
    the same number of instances as the original training set, however instances 
    are sampled with replacement. Soft voting is used to make predictions.
    
    Args:
        n_estimators (int): The number of decision trees to be used in the ensemble.
        criterion (str): The criterion used by the decision trees in the ensemble.
        splitter (str): The splitter used by the decision trees in the ensemble.
        max_depth (int): The maximum depth of the decision trees in the ensemble.
        min_samples_split (int): The minimum number of samples required at a node
                                 in a decision tree for a split to be made.
        min_samples_leaf (int): The minimum number of samples required to be at
                                each leaf node in a decision tree.
    
    Attributes:
        n_estimators (int): The number of decision trees in the ensemble.
        criterion (str): The criterion used by the decision trees in the ensemble.
        splitter (str): The splitter used by the decision trees in the ensemble.
        max_depth (int): The maximum depth of the decision trees in the ensemble.
        min_samples_split (int): The minimum number of samples required at a node
                                 in a decision tree for a split to be made.
        min_samples_leaf (int): The minimum number of samples required to be at
                                each leaf node in a decision tree.
    '''
    def __init__(self,
                 n_estimators=10, 
                 criterion='gini',
                 splitter='best',
                 max_depth=2, 
                 min_samples_split=2,
                 min_samples_leaf=1):
       
        model = DecisionTreeClassifier(criterion=criterion, 
                                       splitter=splitter,
                                       max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_split)
        
        super().__init__(base_estimator=model, 
                       n_estimators=n_estimators)
    
    
    def predict_proba(self,X):
        '''Predict the class probabilities for a given instance.'''
        preds = [estimator.predict_proba(X) for estimator in self.estimators]
        mx = -1
        for i in preds:
            if i.shape[0] > mx:
                mx = i.shape[0]
        preds = [np.pad(x, (0, mx - x.shape[0]), 'constant') for x in preds]
        preds = sum(preds)/float(len(self.estimators))
        return preds


    def predict(self,X):
        '''Predict the class of a given instance.'''
        preds = self.predict_proba(X)
        return np.argmax(preds)
    
 
    
class RandomForestRegressor(Bagging):
    '''
    A random forest regressor is an ensemble model that makes use of decision 
    trees as the base estimator in the ensemble. The model trains each of the 
    base estimators on a random subset of the training data. Each subset contains 
    the same number of instances as the original training set, however instances 
    are sampled with replacement. The final prediction made by the ensemble is
    equal to the mean predicted target value of all of the decision trees in the 
    ensemble.
    
     Args:
        n_estimators (int): The number of decision trees to be used in the ensemble.
        splitter (str): The splitter used by the decision trees in the ensemble.
        max_depth (int): The maximum depth of the decision trees in the ensemble.
        min_samples_split (int): The minimum number of samples required at a node
                                 in a decision tree for a split to be made.
        min_samples_leaf (int): The minimum number of samples required to be at
                                each leaf node in a decision tree.
    
    Attributes:
        n_estimators (int): The number of decision trees in the ensemble.
        splitter (str): The splitter used by the decision trees in the ensemble.
        max_depth (int): The maximum depth of the decision trees in the ensemble.
        min_samples_split (int): The minimum number of samples required at a node
                                 in a decision tree for a split to be made.
        min_samples_leaf (int): The minimum number of samples required to be at
                                each leaf node in a decision tree.
    '''
    def __init__(self,
                  n_estimators=10,
                  splitter='best',
                  max_depth=2, 
                  min_samples_split=2,
                  min_samples_leaf=1):
       
        model = DecisionTreeRegressor(splitter=splitter,
                                      max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf)
        
        super().__init__(base_estimator=model, 
                       n_estimators=n_estimators)
    
    
    def predict(self, X):
        '''Predict the target value of the given instance.'''
        if X.ndim == 0:
            X = np.array([X])
        preds = np.array([estimator.predict(X) for estimator in self.estimators])
        return preds.mean()
    
    

class ExtraTreesClassifier(Bagging):
    '''
    An Extremely Randomized Trees Classifier is an ensemble model that aggregates 
    the predictions of multiple randomized decision trees (extra-trees). Each 
    of the decision trees are trained on a random subset of the training data.
    The decision trees used in the ensemble choose split points based on random
    threshold values. Soft voting is used to make predictions.
    
    Args:
        n_estimators (int): The number of decision trees to be used in the ensemble.
        criterion (str): The criterion used by the decision trees in the ensemble.
        max_depth (int): The maximum depth of the decision trees in the ensemble.
        min_samples_split (int): The minimum number of samples required at a node
                                 in a decision tree for a split to be made.
        min_samples_leaf (int): The minimum number of samples required to be at
                                each leaf node in a decision tree.
        max_samples (float): The fraction of the total number of training instances
                             used to train each decision tree in the ensemble.     
    
    Attributes:
        n_estimators (int): The number of decision trees used in the ensemble.
        criterion (str): The criterion used by the decision trees in the ensemble.
        max_depth (int): The maximum depth of the decision trees in the ensemble.
        min_samples_split (int): The minimum number of samples required at a node
                                 in a decision tree for a split to be made.
        min_samples_leaf (int): The minimum number of samples required to be at
                                each leaf node in a decision tree.
        max_samples (float): The fraction of the total number of training instances
                             used to train each decision tree in the ensemble.
    '''   
    def __init__(self,
                 n_estimators=10,
                 criterion='gini',
                 max_depth=2,
                 min_samples_split=2,
                 min_samples_leaf=2,
                 max_samples=1.0):
            
         model = DecisionTreeClassifier(splitter='random',
                                       criterion=criterion,
                                       max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf)
         super().__init__(base_estimator=model,
                          n_estimators=n_estimators,
                          max_samples=max_samples,
                          bootstrap=False)
    
    
    def predict_proba(self,X):
        '''Predict the class probabilities for a given instance.'''
        preds = [estimator.predict_proba(X) for estimator in self.estimators]
        mx = -1
        for i in preds:
            if i.shape[0] > mx:
                mx = i.shape[0]
        preds = [np.pad(x, (0, mx - x.shape[0]), 'constant') for x in preds]
        preds = sum(preds)/float(len(self.estimators))
        return preds


    def predict(self,X):
        '''Predict the class of a given instance.'''
        preds = self.predict_proba(X)
        return np.argmax(preds)

 
       
class ExtraTreesRegressor(Bagging):
    '''
    An Extremely Randomized Trees Regressor is an ensemble model that aggregates 
    the predictions of multiple randomized decision trees (extra-trees). Each 
    of the decision trees are trained on a random subset of the training data.
    The decision trees used in the ensemble choose split points based on random
    threshold values. The final prediction made by the ensemble is equal 
    to the mean predicted target value of all of the decision trees in the 
    ensemble.
    
    Args:
        n_estimators (int): The number of decision trees to be used in the ensemble.
        max_depth (int): The maximum depth of the decision trees in the ensemble.
        min_samples_split (int): The minimum number of samples required at a node
                                 in a decision tree for a split to be made.
        min_samples_leaf (int): The minimum number of samples required to be at
                                each leaf node in a decision tree.
        max_samples (float): The fraction of the total number of training instances
                             used to train each decision tree in the ensemble.     
    
    Attributes:
        n_estimators (int): The number of decision trees used in the ensemble.
        max_depth (int): The maximum depth of the decision trees in the ensemble.
        min_samples_split (int): The minimum number of samples required at a node
                                 in a decision tree for a split to be made.
        min_samples_leaf (int): The minimum number of samples required to be at
                                each leaf node in a decision tree.
        max_samples (float): The fraction of the total number of training instances
                             used to train each decision tree in the ensemble.
    ''' 
    def __init__(self,
                 n_estimators=10,
                 max_depth=2,
                 min_samples_split=2,
                 min_samples_leaf=2,
                 max_samples=1.0):
            
         model = DecisionTreeRegressor(splitter='random',
                                       max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf)
         super().__init__(base_estimator=model,
                          n_estimators=n_estimators,
                          max_samples=max_samples,
                          bootstrap=False)
      
        
    def predict(self, X):
        '''Predict the target value for a given instance.'''
        if X.ndim == 0:
            X = np.array([X])
        preds = np.array([estimator.predict(X) for estimator in self.estimators])
        return preds.mean()