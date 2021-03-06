# Created by Tristan Bester.
from mlgroundup.clustering import KMedians
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# Create data set.
X,_ = make_blobs(500, 2, 2, 5, random_state=42)

# Calculate clusters.
km = KMedians(n_clusters=5, n_iters=10)
km.fit(X) 

# Calculate the cluster to which each instance belongs. 
cluster = [km.predict(x) for x in X]

# Plot the clusters.
plt.scatter(X[:, 0], X[:, 1], c=cluster, cmap='jet')
plt.xlabel('Feature one (Dimension one):')
plt.ylabel('Feature two (Dimension two):')
plt.title('K-medians clustering:')