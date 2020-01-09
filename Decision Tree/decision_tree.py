# Created by Tristan Bester
import numpy as np

class Node(object):
    '''
    Objects of this class serve as single nodes in the decision trees created by
    the Tree class defined below. The objects function to store the necessary 
    information required by a node in a decision tree.
    
    Args:
        gini (float): The Gini impurity of the node.
        decision (tuple): Stores the feature and threshold used for the decision
                          at the node. Format: (threshold, feature index).
        classes (numpy.ndarray): The number of samples in each class for the instances
                                 to which the node applies.
    
    Attributes:
        gini (float): The Gini impurity of the node.
        decision (tuple): The information used to make the decision at the node.
        classes (numpy.ndarray): The number of samples in each class for the instances
                                 to which the node applies.
        prediction (int): The predicted class for all of the instances to which 
                          the node applies.
        left (Node): The left child node of the current node.
        right (Node): The right child node of the current node.
    '''
    def __init__(self, gini, decision, classes):
        self.gini = gini
        self.decision = decision
        self.classes = classes
        self.prediction = np.argmax(classes)
        self.left = None
        self.right = None
        


class Tree(object):
    '''
    A decision tree is a tree-like model in which instances are classified based
    on a series of attribute tests. Each instance moves from the root node, down 
    the decision tree, until it reaches a leaf node at which point the instance
    is classified. The path the instance follows to reach a leaf node is determined
    based on the reesult of a set a predetermined attribute tests.
    
    Args:
        None.
    
    Attributes:
        root (Node): The root node of the decision tree.
        decisions (list): A list storing all of the decision used
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
        
        
    def split(self, split_val, split_col, X, y):
        '''Split the given dataset based on the given feature and threshold value.'''
        lower = []
        y_lower = []
        upper = []
        y_upper = []
        
        # Split the data at the based on the specified feature and threshold value.
        for i,x in enumerate(X[:,split_col]):
            if x < split_val:
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
    
    
    def gini_impurity(self,y):
        '''Calculate the Gini impurity of the specified node.'''
        ratio = np.bincount(y)
        total = ratio.sum()
        gini = 1
        for i in ratio:
            gini -= (i/total)**2
        return gini
    
    
    def CART(self, X,y):
        '''CART algorithm for classification.'''
        m,n = X.shape
        splits = X.ravel(order='C')
        min_cost = np.inf
        ginis = []
        
        # Testing all possible split points.
        for i,x in enumerate(splits):
            lower, y_lower, upper, y_upper = self.split(x, (i % 2), X, y)
            
            gini_left = self.gini_impurity(y_lower) 
            gini_right = self.gini_impurity(y_upper)
            
            
            gini_left *= (len(lower) / m)
            gini_right *= (len(upper) / m)
            cost = gini_left + gini_right
            
            ginis.append((gini_left + gini_right, (x, (i % 2))))
            # Store split that results in lowest value for CART cost function
            # for classification.
            if cost < min_cost:
                min_cost = cost
                best_split = (x, (i % 2))
        
        # Return the node that best classifies the given data set.
        return Node(min_cost, best_split, np.bincount(y))
        

    def __fit(self, subtree, X, y, classes, curr_depth):
        '''
        If the dataset does not already have a Gini impurity of zero, create and
        add a node to the decision tree that better classfies the instances in
        the given dataset. The recursively call the _fit method to create the 
        child nodes.
        '''
        if curr_depth > self.max_depth or len(X) < self.min_samples_split or len(X) < self.min_samples_leaf or self.gini_impurity(y) == 0: 
            return Node(0, None, np.bincount(y))
        else:
            subtree = self.CART(X,y)
            lower, y_lower, upper, y_upper = self.split(subtree.decision[0], 
                                                        subtree.decision[1], X, y)
            
            subtree.left = self.__fit(subtree.left, lower, y_lower, subtree.classes, curr_depth + 1)
            subtree.right = self.__fit(subtree.right, upper, y_upper, subtree.classes, curr_depth + 1)
            return subtree
        
        
        
    def fit(self,X,y):
        '''
        Build the decision tree by recursively calling the __fit method until
        all of the training instances have been correctly classified or the specified
        regularization criteria have been satisified.
        '''
        self.root = self.CART(X,y)
    
        lower, y_lower, upper, y_upper = self.split(self.root.decision[0], 
                                                    self.root.decision[1], X, y)
        self.root.left = self.__fit(self.root.left, lower, y_lower, np.bincount(y_lower), 1)
        self.root.right = self.__fit(self.root.right, upper, y_upper, np.bincount(y_upper), 1)
        
        
    def __predict(self, subtree, val):
        '''Predict the class of an instance.'''
        if subtree.decision is None:
            return subtree.prediction
        elif val[subtree.decision[1]] < subtree.decision[0]:
            return self.__predict(subtree.left, val)
        else:
            return self.__predict(subtree.right, val)
        
    def predict(self, val):
        '''Predict the class of an instnace.'''
        #print(self.root.decision)
        return self.__predict(self.root, val)
        







   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # def __traverse(self, subtree):
    #     if subtree.left is not None:
    #         self.__traverse(subtree.left)
    #     if subtree.decision is not None:
    #         self.decisions.append(subtree.decision)
    #     if subtree.right is not None:
    #         self.__traverse(subtree.right)
           
        
    # def traverse(self):
    #     self.decisions = []
    #     self.__traverse(self.root)



















# fig,ax = plt.subplots()

# ax.set_xlim(left=X_one_min, right=X_one_max)
# ax.set_ylim(bottom=X_two_min, top=X_two_max)

# tree = Tree()
# tree.fit(X,y)
# tree.traverse()

# ax.scatter(X[:,0], X[:, 1], c=y, cmap='winter')

# patches = []

# for x in tree.decisions:
#     if x[1] == 1: #hor line
#         print('ji')
#         x1 = X_one_min
#         x2 = X_one_max
#         if abs(x[0] - X_two_min) > abs(x[0]- X_two_max):
#             y1 = X_two_max
#         else:
#             y1 = X_two_min

#         y2 = x[0]
        
#         polygon = Polygon([(x1,y1), (x1,y2), (x2,y2), (x2,y1)])
#         patches.append(polygon)
#     else: #vert line
#         print('ji')
#         y1 = X_two_min
#         y1 = X_two_max
#         if abs(x[0] - X_one_min) > abs(x[0]- X_one_max):
#             x1 = X_one_max
#         else:
#             x1 = X_one_min

#         x2 = x[0]
        
#         polygon = Polygon([(x1,y1), (x1,y2), (x2,y2), (x2,y1)])
#         patches.append(polygon)
  
# p = PatchCollection(patches, cmap=plt.cm.jet, alpha=0.4)
# colors = 100*np.random.rand(len(patches))
# p.set_array(np.array(colors))

# ax.add_collection(p)

# plt.show()




# itemindex = np.where(X[:,1] == tree.decisions[0][0])
# print('Index: ', itemindex)
# point  = X[itemindex]
# print(point)

    
    



# fig,ax = plt.subplots()

# ax.scatter(X[:,0], X[:, 1], c=y, cmap='winter')

# pos = X[y==1]
# neg = X[y==0]

# patches = []

# polygon = Polygon(pos, True)
# patches.append(polygon)
# polygon = Polygon(neg, True)
# patches.append(polygon)

# p = PatchCollection(patches, cmap=plt.cm.jet, alpha=0.4)
# colors = 100*np.random.rand(len(patches))
# p.set_array(np.array(colors))

# ax.add_collection(p)

# plt.show()









# tree = Tree()
# tree.fit(X,y)
# tree.traverse()

# print(tree.decisions)
# # print(tree.root.classes)
# # print(tree.root.left.classes)
# # print(tree.root.right.classes)
# # print(tree.root.left.left.classes)
# # print(tree.root.left.right.classes)
# # print(tree.root.left.right.left.classes)
# # print(tree.root.left.right.right.classes)


# #tree.traverse()

# #print(tree.decisions)

# preds = []

# for i,x in zip(X,y):
#     preds.append(tree.predict(i))
# print(np.sum(preds - y))

# plt.scatter(X[:,0], X[:, 1], c=y, cmap='winter')

# my_map = plt.cm.cool
# # sm = plt.cm.ScalarMappable(cmap=my_map,norm=plt.Normalize(0,len(tree.decisions)))


# colours = [my_map(i) for i in np.linspace(0,1,len(tree.decisions))]

# for i,x in enumerate(tree.decisions):
#     if x[1] == 1:
#         plt.axhline(y=x[0], color=colours[i], label=i)
#     else:
#         plt.axvline(x=x[0], color=colours[i], label=i)

# # fig = plt.gcf()
# # fig.colorbar(sm)
# plt.legend()
















































# def split(split_val, split_col):
#     lower = []
#     y_lower = []
#     upper = []
#     y_upper = []
    
#     #split
#     for i,x in enumerate(X[:,split_col]):
#         if x < split_val:
#             lower.append(X[i])
#             y_lower.append(y[i])
#         else:
#             upper.append(X[i])
#             y_upper.append(y[i])
#     return lower, y_lower, upper, y_upper
        
# def gini_impurity(y):
#     ratio = np.bincount(y)
#     total = ratio.sum()
#     gini = 1
#     for i in ratio:
#         gini -= (i/total)**2
#     return gini



# # lower, y_lower, upper, y_upper = split(3, 1)
# # print(gini_impurity(y))
# # print(gini_impurity(y_lower))
# # print(gini_impurity(y_upper))



# m,n = X.shape
# count = 0
# splits = X.ravel(order='C')
# test_val = 1

# min_cost = 99

# ginis = []

# for i,x in enumerate(splits):
#     lower, y_lower, upper, y_upper = split(x, (i % 2))
    
#     gini_left = gini_impurity(y_lower) 
#     gini_right = gini_impurity(y_upper)
    
    
#     gini_left *= (len(lower) / m)
#     gini_right *= (len(upper) / m)
#     cost = gini_left + gini_right
#     ginis.append((gini_left + gini_right, (x, (i % 2))))
#     if cost < min_cost:
#         min_cost = cost
#         best_split = (x, (i % 2))
    
# for i in ginis:
#     print(i)
# print(best_split)

# # print(count)
# # print(count % n)
# # print(f'split val: {test_val}, col: {count % n}')
# #plt.scatter(X[:,0], X[:,1], c=y, cmap='winter') 
        
        
        
        
      


























  
        