# Created by Tristan Bester.
from mlgroundup.unsupervised import PCA
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Create dataset.
X, y = make_blobs(n_samples=100, n_features=3, centers=2, cluster_std=4, random_state=42)

# Apply PCA to the dataset.
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

fig = plt.gcf()
ax = Axes3D(fig)
ax.view_init(azim=160, elev=20)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='cool')

# Plot the principal components.
for x,i in enumerate(pca.components):
    comp = np.array([i * 15])
    zeros = np.zeros((1,len(i)))
    comp = np.concatenate((comp,zeros), axis=0)
    ax.plot(comp[:, 0], comp[:, 1], comp[:, 2], label=f'PC: {x + 1}')
ax.legend()