# Created by Tristan Bester.
from mlgroundup.ensemble import VotingClassifier
from mlgroundup.supervised import DecisionTreeClassifier, LogisticRegression
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Create training set.
X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=2,
                  random_state=0)

# Create base estimators.
decision_tree = DecisionTreeClassifier()
log_reg = LogisticRegression(eta=0.5, n_iters=100, decision_boundary=0.5)

# Create ensemble model.
clf = VotingClassifier(estimators=[decision_tree, log_reg], voting='hard')
clf.fit(X,y)

train_set = X.copy() 
labels = y.copy()

X_one_min = X[:,0].min() 
X_one_max = X[:,0].max()
X_two_min = X[:,1].min()
X_two_max = X[:,1].max()

# Calculate the decision boundary of the model.
X1, X2 = np.meshgrid(np.linspace(X_one_min, X_one_max, 500), np.linspace(X_two_min, X_two_max,500))
X = [[x1,x2] for x1,x2 in zip(X1.flatten(),X2.flatten())]
y = lambda x: clf.predict(np.array(x))
Y = [y(x) for x in X]
Y = np.array(Y)
Y = Y.reshape(X1.shape)

fig = plt.gcf()
fig.set_size_inches((10,7))

# Plot the decision boundary of the model.
cp1 = plt.contourf(X1, X2, Y, [-1,0, 1, 2], cmap='winter')
plt.scatter(train_set[:,0], train_set[:,1], c=labels, cmap='winter', edgecolors='black')

plt.xlim(X_one_min, X_one_max) 
plt.ylim(X_two_min, X_two_max)  
plt.xlabel('Feature one (Dimension one)')
plt.ylabel('Feature two (Dimension two)')
plt.title('Voting Classifier:')
plt.show()