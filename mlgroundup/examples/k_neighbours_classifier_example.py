# Created by Tristan Bester.
from mlgroundup.supervised import KNeighboursClassifier
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Create training set.
X, y = make_blobs(n_samples=100, n_features=2, centers=5, cluster_std=5,
                  random_state=42)

knn = KNeighboursClassifier(5)
knn.fit(X,y)

# Get feature limits of dataset.
x_one_min = X[:, 0].min()
x_one_max = X[:, 0].max()
x_two_min = X[:, 1].min()
x_two_max = X[:, 1].max()

# Create filled contour plot.
x1, x2 = np.meshgrid(np.linspace(x_one_min, x_one_max, 100), np.linspace(x_two_min, x_two_max, 100))
X2 = [[x1, x2] for x1, x2 in zip(x1.flatten(),x2.flatten())]
y_2 = lambda x: knn.predict(x)
y2 = np.array([y_2(x) for x in X2])
y2 = y2.reshape(x1.shape)

fig = plt.gcf()
fig.set_size_inches((10, 7))
plt.xlim([x_one_min, x_one_max])
plt.ylim([x_two_min, x_two_max])
plt.contourf(x1, x2, y2, [-0.5,0.5,1.5,2.5,3.5,4.5], cmap='spring')
cbar = plt.colorbar()
cbar.set_ticks([0,1,2,3,4])
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer', edgecolor='black')
plt.xlabel('Feature one (Dimension one):')
plt.ylabel('Feature two (Dimension two):')
plt.title('K-nearest neighbours classifier:')