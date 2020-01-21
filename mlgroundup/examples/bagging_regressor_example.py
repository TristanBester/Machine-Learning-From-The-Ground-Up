# Created by Tristan Bester.
from mlgroundup.supervised import DecisionTreeRegressor
from mlgroundup.ensemble import BaggingRegressor
import matplotlib.pyplot as plt
import numpy as np

# Create the training set.
X = np.linspace(-10,10, 100).astype(float)
y = [x**2 + np.random.rand() * 25 for x in X]
X = np.array(X)
y = np.array(y)

# Create base estimator.
tree = DecisionTreeRegressor(max_depth=4)

# Create ensemble model.
bag = BaggingRegressor(base_estimator=tree, n_estimators=20)
bag.fit(X,y)

preds = [bag.predict(i) for i in X]

# Plot the predictions.
fig = plt.gcf()
fig.set_size_inches((10,7))

plt.scatter(X, y, s=20, c='black', marker='x', label='Target')
plt.plot(X,preds,linewidth=0.5, c='r', label='Bagging regressor')

plt.title('Bagging regressor:')
plt.xlabel('Feature one (Dimension one)')
plt.ylabel('Feature two (Dimension two)')
plt.legend(loc='upper center')