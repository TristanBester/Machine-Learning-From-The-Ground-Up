# Created by Tristan Bester.
from mlgroundup.supervised import KNeighboursRegressor
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

# Create training set.
X = np.linspace(-2, 2, 50)
y =np.array([x**4  + np.random.rand()*3 for x in X])

knn = KNeighboursRegressor(n_neighbours=5)
knn.fit(X, y)

# Make predictions.
preds = [knn.predict(x) for x in X]

fig = plt.gcf()
fig.set_size_inches((10,7))

# Plot the model's predictions.
plt.scatter(X,y, c='darkviolet')
plt.plot(X, preds, c='black')
plt.xlabel('Feature one (Dimension one):')
plt.ylabel('Feature two (Dimension two):')
plt.title('K-nearest neighbours regressor:')