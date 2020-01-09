from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection



class Node(object):
    def __init__(self, val, decision, classes):
        self.gini = val
        self.decision = decision
        self.classes = classes
        self.prediction = np.argmax(classes)
        self.left = None
        self.right = None
        
        
class Tree(object):
    def __init__(self):
        self.root = None
        
    def split(self, split_val, split_col, X, y):
        lower = []
        y_lower = []
        upper = []
        y_upper = []
        
        #split
        for i,x in enumerate(X[:,split_col]):
            if x < split_val:
                lower.append(X[i])
                y_lower.append(y[i])
            else:
                upper.append(X[i])
                y_upper.append(y[i])
                
        lower = np.array(lower)
        upper = np.array(upper)
        y_lower = np.array(y_lower).astype(int)
        y_upper = np.array(y_upper).astype(int)
        return lower, y_lower, upper, y_upper   
    
    def gini_impurity(self,y):
        ratio = np.bincount(y)
        total = ratio.sum()
        gini = 1
        for i in ratio:
            gini -= (i/total)**2
        return gini
    
    def CART(self, X,y):
        m,n = X.shape
        splits = X.ravel(order='C')
        min_cost = np.inf
        ginis = []
        
        for i,x in enumerate(splits):
            lower, y_lower, upper, y_upper = self.split(x, (i % 2), X, y)
            
            gini_left = self.gini_impurity(y_lower) 
            gini_right = self.gini_impurity(y_upper)
            
            
            gini_left *= (len(lower) / m)
            gini_right *= (len(upper) / m)
            cost = gini_left + gini_right
            
            ginis.append((gini_left + gini_right, (x, (i % 2))))
            if cost < min_cost:
                min_cost = cost
                best_split = (x, (i % 2))
        #print(best_split)        
        return Node(min_cost, best_split, np.bincount(y))
        


    def __fit2(self, subtree, X, y, classes):
        if self.gini_impurity(y) == 0: 
            return Node(0, None, np.bincount(y))
        else:
            subtree = self.CART(X,y)
            lower, y_lower, upper, y_upper = self.split(subtree.decision[0], 
                                                        subtree.decision[1], X, y)
            
            subtree.left = self.__fit2(subtree.left, lower, y_lower, subtree.classes)
            subtree.right = self.__fit2(subtree.right, upper, y_upper, subtree.classes)
            return subtree
       
    def fit(self,X,y):
        self.root = self.CART(X,y)
    
        lower, y_lower, upper, y_upper = self.split(self.root.decision[0], 
                                                    self.root.decision[1], X, y)
        
        self.root.left = self.__fit2(self.root.left, lower, y_lower, np.bincount(y_lower))
        self.root.right = self.__fit2(self.root.right, upper, y_upper, np.bincount(y_upper))
        # print(self.root.left.classes)
        # print(self.root.right.classes)
        
            
            
            
    def __traverse(self, subtree):
        if subtree.left is not None:
            self.__traverse(subtree.left)
        #print(subtree.gini, end = ' ')
        #print(subtree.decision, end = ' ')
        print(subtree.decision, end = ' ')
        if subtree.decision is not None:
            self.decisions.append(subtree.decision)
        if subtree.right is not None:
            self.__traverse(subtree.right)
           
        
    def traverse(self):
        self.decisions = []
        self.__traverse(self.root)
        print()
        
    def __predict(self, subtree, val):
        if subtree.decision is None:
            return subtree.prediction
        elif val[subtree.decision[1]] < subtree.decision[0]:
            return self.__predict(subtree.left, val)
        else:
            return self.__predict(subtree.right, val)
        
    def predict(self, val):
        return self.__predict(self.root,val)
        

data = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=5, 
                  random_state=0)
X = data[0]
y = data[1]   

train_set = X.copy() 
labels = y.copy()

X_one_min = X[:,0].min() * 1.1
X_one_max = X[:,0].max() * 1.1
X_two_min = X[:,1].min() * 1.1
X_two_max = X[:,1].max() * 1.1

tree = Tree()
tree.fit(X,y)


X1, X2 = np.meshgrid(np.linspace(X_one_min, X_one_max, 500), np.linspace(X_two_min, X_two_max,500))
X = [[x1,x2] for x1,x2 in zip(X1.flatten(),X2.flatten())]
y = lambda x: tree.predict(x)
Y = []
for x in X:
    Y.append(y(x))

Y = np.array(Y)
Y = Y.reshape(X1.shape)

fig = plt.gcf()
fig.set_size_inches((10,7))


cp1 = plt.contourf(X1, X2, Y, [-1,0, 1], cmap='winter')
plt.scatter(train_set[:,0], train_set[:,1], c=labels, cmap='winter', edgecolors='black')
#plt.scatter(X1, X2, c=Y, s=0.1)

    
plt.xlabel('Feature one (Dimension one)')
plt.ylabel('Feature two (Dimension two)')
plt.title(f'Decision boundary of a decision tree with no regularization:')
plt.show()

plt.show()



























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
        
        
        
        
      


























  
        