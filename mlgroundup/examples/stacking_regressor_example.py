# Created by Tristan Bester.
from mlgroundup.ensemble import StackingRegressor
from mlgroundup.supervised import DecisionTreeRegressor, LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Create training set.
X = np.linspace(-10, 10, 30)
y = np.array([x**3 for x in X])

# First layer estimators to be used in model. 
base = [DecisionTreeRegressor(max_depth=1),
        DecisionTreeRegressor(max_depth=2),
        DecisionTreeRegressor(max_depth=3),
        DecisionTreeRegressor(max_depth=4),
        DecisionTreeRegressor(max_depth=5),
        DecisionTreeRegressor(max_depth=6),
        DecisionTreeRegressor(max_depth=7)]

# Meta learner to be used in model.
final = LinearRegression(eta=0.00000001, n_iters=10000)

stack = StackingRegressor(base, final)
stack.fit(X, y)
preds = [stack.predict(x) for x in X]

fig = plt.gcf()
fig.set_size_inches((5,5))
plt.scatter(X, y, c='black', marker='v', label='Target')
plt.plot(X, preds, c='mediumspringgreen', label='Prediction')
plt.xlabel('Feature one (Dimension one):')
plt.ylabel('Feature two (Dimension two):')
plt.title('Predictions of StackingRegressor:')
plt.legend()