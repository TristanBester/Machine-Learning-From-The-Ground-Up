# Created by Tristan Bester.
import numpy as np

class Node(object):
    '''
    Objects of this class serve as single nodes in the decision trees created by
    the classes defined below. The objects function to store the necessary
    information required by a node in a decision tree.

    Args:
        decision (tuple): Stores the feature index and threshold value used for
                          the decision at the node. Format: (threshold, feature index).
        classes (numpy.ndarray): An array containing the class frequencies of the
                                 instances associated wiht the node.
        prediction (float): The predicted class for the samples at this node
                            in the case of classification. The predicted target
                            value in the case of regression.

    Attributes:
        decision (tuple): The information used to make the decision at the node.
        classes (numpy.ndarray): An array containing the class frequencies of the
                                 instances associated wiht the node.
        prediction (float): The predicted class/target value for
                            samples at this node.
        left (Node): The left child node of the current node.
        right (Node): The right child node of the current node.
    '''
    def __init__(self, decision, classes=None, prediction=None):
        self.decision = decision
        if prediction is None:
            self.classes = classes
            self.prediction = np.argmax(classes)
        else:
            self.prediction = prediction
        self.left = None
        self.right = None



class Tree(object):
    '''
    This class is the super class for the DecisionTreeClassifier class and the
    DecisionTreeRegressor class.

    Args:
        splitter (str): The splitting method used by the decision tree.
        max_depth (int): The maximum depth of the decision tree that can be created.
        min_samples_split: (int): The minimum number of samples required at a node
                                  for the samples to be split into smaller subsets.
        min_samples_leaf: (int): The minimum number of samples required to be at
                                 each leaf node.

    Attributes:
        root (Node): The root node of the decision tree.
        splitter (str): The splitting method used by the decision tree.
        max_depth (int): The maximum depth of the decision tree that can be created.
        min_samples_split: (int): The minimum number of samples required at a node
                                  for the samples to be split into smaller subsets.
        min_samples_leaf: (int): The minimum number of samples required to be at
                                 each leaf node.
    '''
    def __init__(self, splitter='best', max_depth=np.inf, min_samples_split=2, min_samples_leaf=1):
        self.root = None
        
        if splitter == 'best' or splitter == 'random':
            self.splitter = splitter
        else:
            raise AttributeError('Invalid splitter value, \'best\' and \'random\' available.')
            
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


    def split(self, split_val, split_col, X, y):
        '''Split the given dataset based on the given feature index and threshold value.'''
        lower = []
        y_lower = []
        upper = []
        y_upper = []

        # Split the data at the based on the specified feature index and threshold value.
        for i,x in enumerate(X[:, split_col]):
            if x <= split_val:
                lower.append(X[i])
                y_lower.append(y[i])
            else:
                upper.append(X[i])
                y_upper.append(y[i])

        # The separated instances.
        lower = np.array(lower)
        upper = np.array(upper)
        # The labels corresponding to the separated instances.
        y_lower = np.array(y_lower).astype(int)
        y_upper = np.array(y_upper).astype(int)
        return lower, y_lower, upper, y_upper


    def __predict_proba(self, subtree, val):
        '''Return the class frequencies for instances associated with the node.'''
        if val.ndim == 0:
            val = np.array([val])

        if subtree.decision is None:
            return subtree.classes
        elif val[subtree.decision[1]] <= subtree.decision[0]:
            return self.__predict_proba(subtree.left, val)
        else:
            return self.__predict_proba(subtree.right, val)


    def predict_proba(self, val):
        '''Predict the class probabilities of an instance.'''
        if self.root is None:
            raise AttributeError('Model not fitted, call \'fit\' with appropriate arguments before using model.')
        else:
            classes = self.__predict_proba(self.root, val)
            return classes / np.sum(classes)


    def predict(self, val):
        '''Predict the class of an instance.'''
        if self.root is None:
            raise AttributeError('Model not fitted, call \'fit\' with appropriate arguments before using model.')
        else:
            classes = self.__predict_proba(self.root, val)
            return np.argmax(classes)

                 

class DecisionTreeClassifier(Tree):
    '''
    A decision tree classifier is a tree-like model in which instances are classified based
    on a series of attribute tests. Each instance moves from the root node, down
    the decision tree, until it reaches a leaf node at which point the instance
    is classified. The path the instance follows to reach a leaf node is determined
    based on the result of a set of predetermined attribute tests.

    Args:
        criterion (str): The metric used to determine the split points in the decision tree.
        splitter (str): The splitting method used by the decision tree.
        max_depth (int): The maximum depth of the decision tree that can be created.
        min_samples_split: (int): The minimum number of samples required at a node
                                  for the samples to be split into smaller subsets.
        min_samples_leaf: (int): The minimum number of samples required to be at
                                 each leaf node.

    Attributes:
        root (Node): The root node of the decision tree.
        criterion (str): The metric used to determine the split points in the decision tree.
        splitter (str): The splitting method used by the decision tree.
        max_depth (int): The maximum depth of the decision tree that can be created.
        min_samples_split: (int): The minimum number of samples required at a node
                                  for the samples to be split into smaller subsets.
        min_samples_leaf: (int): The minimum number of samples required to be at
                                 each leaf node.
    '''
    def __init__(self,  criterion='gini',  splitter='best', max_depth=np.inf, min_samples_split=2, min_samples_leaf=1):

        super().__init__(splitter, max_depth, min_samples_split, min_samples_leaf)

        if criterion == 'gini' or criterion == 'entropy':
            self.criterion = criterion
        else:
            raise AttributeError('Invalid criterion, Gini impurity and Entropy available.')


    def gini_impurity(self,y):
        '''Calculate the Gini impurity of the specified node.'''
        ratio = np.bincount(y)
        total = ratio.sum()
        gini = 1
        for i in ratio:
            gini -= (i/total)**2
        return gini


    def entropy(self, y):
        '''Calculate the entropy of the specified node.'''
        _, counts = np.unique(y, return_counts=True)

        n = float(sum(counts))
        summation = 0

        for i in counts:
            summation += -1 * (i/n) * np.log2(i/n)

        return summation


    def information_gain(self, y_left, y_right, y_parent):
        '''Calculate the information gain from the specified split.'''
        n = float(y_parent.shape[0])
        entropy_left = self.entropy(y_left)
        entropy_right = self.entropy(y_right)
        entropy_parent = self.entropy(y_parent)

        weighted_entropy = (((y_left.shape[0]/n) * entropy_left) +
                           ((y_right.shape[0]/n) * entropy_right))

        info_gain = entropy_parent - weighted_entropy
        return info_gain


    def calculate_improvement(self,y_lower, y_upper, y):
        '''
        Calculate the appropriate CART cost function value based on the criterion
        being used in the model.
        '''
        if self.criterion == 'gini':
            gini_left = self.gini_impurity(y_lower)
            gini_right = self.gini_impurity(y_upper)
            gini_left *= (len(y_lower) / len(y))
            gini_right *= (len(y_upper) / len(y))
            cost = gini_left + gini_right
            return cost
        else:
            gain = self.information_gain(y_lower, y_upper, y)
            return gain
     
        
    def get_split_points(self,X):
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


    def CART(self, X, y):
        '''The CART algorithm for building decision trees for classification.'''
        if self.splitter == 'best':
            splits, cols = self.get_split_points(X)
        else:
            pts = {}
            splits, cols = self.get_split_points(X)
            for i in np.unique(cols):
                pts[i] = []
            
            for i,x in zip(cols,splits):
                pts[i].append(x)
            
            cols = []
            splits = []
            
            for i,x in enumerate(pts.values()):
                cols.append(i)
                splits.append(np.random.choice(x))
    
        if self.criterion == 'gini':
            best = np.inf
        else:
            best  = -1
               
        # Test possible split points and store the optimal split point.
        for i,x in zip(cols,splits):
            lower, y_lower, upper, y_upper = self.split(x, i, X, y)
            
            val = self.calculate_improvement(y_lower, y_upper, y)

            if self.criterion == 'gini' and val < best:
                left = np.c_[lower,y_lower]
                right = np.c_[upper, y_upper]
                best = val
                # Split val and split col.
                best_split = (x, i)
            elif val > best:
                left = np.c_[lower,y_lower]
                right = np.c_[upper, y_upper]
                best = val
                # Split val and split col.
                best_split = (x, i)
        
        return left, right, best_split


    def __fit(self, subtree, X, y, curr_depth):
        '''
        If the dataset does not already have an impurity value of zero and the
        regularization parameters have not been satisfied, create and add a node
        to the decision tree that better classifies the instances in the given
        dataset. Then recursively call the __fit method to create the child nodes.
        '''
        classes = np.bincount(y)

        if self.criterion == 'gini':
            impurity = self.gini_impurity(y)
        else:
            impurity = self.entropy(y)

        if (
                curr_depth > self.max_depth or
                len(X) < self.min_samples_split or
                len(X) < self.min_samples_leaf or
                impurity == 0
            ):
            return Node(None, classes)
        else:
            (left, right, best_split) = self.CART(X,y)
            subtree = Node(best_split, classes)
            subtree.left = self.__fit(subtree.left, left[:,:-1], left[:, -1].astype(int), curr_depth + 1)
            subtree.right = self.__fit(subtree.right, right[:,:-1], right[:, -1].astype(int), curr_depth + 1)
            return subtree


    def fit(self,X,y):
        '''
        Build the decision tree by recursively calling the __fit method until
        all of the training instances have been correctly classified or the specified
        regularization criteria have been satisfied.
        '''
        self.root = self.__fit(self.root, X, y, 1)



class DecisionTreeRegressor(Tree):
    '''
    A decision tree regressor is a tree-like model in which the target value of
    instances is predicted based on a series of attribute tests. Each instance
    moves from the root node, down the decision tree, until it reaches a leaf
    node at which point the target value of the instance is predicted to be the
    average target value of all of the instances at that node. The path the
    instance follows to reach a leaf node is determined based on the result of
    a set of predetermined attribute tests.

    Args:
        max_depth (int): The maximum depth of the decision tree that can be created.
        min_samples_split: (int): The minimum number of samples required at a node
                                  for the samples to be split into smaller subsets.
        min_samples_leaf: (int): The minimum number of samples required to be at
                                 each leaf node.

    Attributes:
        root (Node): The root node of the decision tree.
        max_depth (int): The maximum depth of the decision tree that can be created.
        min_samples_split: (int): The minimum number of samples required at a node
                                  for the samples to be split into smaller subsets.
        min_samples_leaf: (int): The minimum number of samples required to be at
                                 each leaf node.
    '''
    def MSE(self, y):
        '''
        Calculate the mean squared error from predicting the mean target value
        of the instances at the node.
        '''
        y_hat = np.mean(y)
        mse = ((y_hat - y)**2).sum()
        return mse


    def CART(self, X, y):
        '''The CART algorithm for building decision trees for regression.'''
        if X.ndim == 1:
            X = X.reshape((-1,1))

        n = X.shape[1]
        splits = X.ravel(order='C')
        best = np.inf

        # Test all possible split points and store the optimal split point.
        for i,x in enumerate(splits):
            lower, y_lower, upper, y_upper = self.split(x, (i % n), X, y)

            # Skip futile splits.
            if len(lower) == 0 or len(upper) == 0:
                continue

            mse_lower = self.MSE(y_lower)
            mse_upper = self.MSE(y_upper)
            m = float(y_lower.shape[0] + y_upper.shape[0])
            cost = (len(y_lower)/m) * mse_lower + (len(y_upper)/m) * mse_upper

            if cost < best:
                left = np.c_[lower,y_lower]
                right = np.c_[upper, y_upper]
                best = cost
                # Split val and split col.
                best_split = (x, (i%n))

        return left, right, best_split


    def __fit(self, subtree, X, y, curr_depth):
        '''
        If the dataset does not already have a mean squared error of zero and the
        regularization parameters have not been satisfied, create and add a node
        to the decision tree that predicts a more accurate target value for the
        instances in the given dataset. Then recursively call the __fit method to
        create the child nodes.
        '''
        prediction = np.mean(y)

        if (
                curr_depth > self.max_depth or
                len(X) < self.min_samples_split or
                len(X) < self.min_samples_leaf or
                self.MSE(y) == 0
            ):
            return Node(decision=None, prediction=prediction)
        else:
            (left, right, best_split) = self.CART(X,y)
            subtree = Node(decision=best_split, prediction=prediction)
            subtree.left = self.__fit(subtree.left, left[:,:-1], left[:, -1].astype(int), curr_depth + 1)
            subtree.right = self.__fit(subtree.right, right[:,:-1], right[:, -1].astype(int), curr_depth + 1)
            return subtree


    def fit(self,X,y):
        '''
        Build the decision tree by recursively calling the __fit method until the
        correct target value has been predicted for all of the training instances
        or the specified regularization criteria have been satisfied.
        '''
        self.root = self.__fit(self.root, X, y, 1)


    def __predict(self, subtree, val):
        '''Predict the target value of an instance.'''
        if val.ndim == 0:
            val = np.array([val])

        if subtree.decision is None:
            return subtree.prediction
        elif val[subtree.decision[1]] <= subtree.decision[0]:
            return self.__predict(subtree.left, val)
        else:
            return self.__predict(subtree.right, val)


    def predict(self, val):
        '''Predict the target value of an instance.'''
        if self.root is None:
            raise AttributeError('Model not fitted, call \'fit\' with appropriate arguments before using model.')
        else:
            return self.__predict(self.root, val)