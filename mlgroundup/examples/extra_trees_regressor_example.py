# Created by Tristan Bester.
from mlgroundup.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
import numpy as np

# Create the training set.
X = np.linspace(-10,10, 100).astype(float)
y = [x**2 + np.random.rand() * 25 for x in X]
X = np.array(X)
y = np.array(y)

# Create ensemble model.
forest = ExtraTreesRegressor(n_estimators=100)
forest.fit(X,y)

preds = [forest.predict(i) for i in X]

# Plot the predictions.
fig = plt.gcf()
fig.set_size_inches((10,7))

plt.scatter(X, y, s=20, c='black', marker='x', label='Target')
plt.plot(X,preds,linewidth=0.5, c='r', label='Extra-Trees regressor')

plt.title('Extra-Trees regressor:')
plt.xlabel('Feature one (Dimension one)')
plt.ylabel('Feature two (Dimension two)')
plt.legend(loc='upper center')