import decision_tree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier


data = make_blobs(n_samples=50, n_features=2, centers=3, cluster_std=4, 
                  random_state=0)
X = data[0]
y = data[1]   

tree = decision_tree.Tree(max_depth=1)
tree2 = DecisionTreeClassifier()
tree.fit(X,y)
tree2.fit(X,y)

ls =[]
ls2 = []

for i in X:
    ls.append(tree.predict(i))
    ls2.append(tree2.predict(i.reshape(1,-1)))
ls = np.array(ls)
ls2 = np.array(ls2)
print((ls-y).sum())
print((ls2-y).sum())


train_set = X.copy() 
labels = y.copy()

X_one_min = X[:,0].min() * 1.1
X_one_max = X[:,0].max() * 1.1
X_two_min = X[:,1].min() * 1.1
X_two_max = X[:,1].max() * 1.1



X1, X2 = np.meshgrid(np.linspace(X_one_min, X_one_max, 500), np.linspace(X_two_min, X_two_max,500))
X = [[x1,x2] for x1,x2 in zip(X1.flatten(),X2.flatten())]
y = lambda x: tree.predict(x)
y2 = lambda x: tree2.predict(np.array(x).reshape(1,-1))
Y = []
Y2 = []
for x in X:
    Y.append(y(x))
    Y2.append(y2(x))

Y = np.array(Y)
Y = Y.reshape(X1.shape)
Y2 = np.array(Y2)
Y2 = Y2.reshape(X1.shape)

fig = plt.gcf()
fig.set_size_inches((10,7))

cp1 = plt.contourf(X1, X2, Y, [-1,0, 1, 2], cmap='winter')
plt.scatter(train_set[:,0], train_set[:,1], c=labels, cmap='winter', edgecolors='black')

    
plt.xlabel('Feature one (Dimension one)')
plt.ylabel('Feature two (Dimension two)')
plt.title(f'Decision boundary of a decision tree with no regularization:')
plt.show()

plt.show()



fig = plt.gcf()
fig.set_size_inches((10,7))

cp1 = plt.contourf(X1, X2, Y2, [-1,0, 1, 2], cmap='winter')
plt.scatter(train_set[:,0], train_set[:,1], c=labels, cmap='winter', edgecolors='black')

    
plt.xlabel('Feature one (Dimension one)')
plt.ylabel('Feature two (Dimension two)')
plt.title(f'Decision boundary of a decision tree with no regularization:')
plt.show()

plt.show()

