# Created by Tristan Bester.
from mlgroundup.supervised import DecisionTreeRegressor
from mlgroundup.ensemble import VotingRegressor
import matplotlib.pyplot as plt
import numpy as np

# Create the training set.
X = np.linspace(-10,10, 100).astype(float)
y = [x**2 + np.random.rand() * 25 for x in X]
X = np.array(X)
y = np.array(y)

# Create base estimators.
tree_one = DecisionTreeRegressor(max_depth=4)
tree_two = DecisionTreeRegressor(max_depth=6)
tree_three = DecisionTreeRegressor(max_depth=8)

# Create ensemble model.
vote = VotingRegressor(estimators=[tree_one, tree_two, tree_three])
vote.fit(X,y)

preds = [vote.predict(i) for i in X]

# Plot the predictions.
fig = plt.gcf()
fig.set_size_inches((10,7))

plt.scatter(X, y, s=20, c='black', marker='x', label='Target')
plt.plot(X,preds,linewidth=0.5, c='r', label='Voting regressor')

plt.title('Voting regressor:')
plt.xlabel('Feature one (Dimension one)')
plt.ylabel('Feature two (Dimension two)')
plt.legend(loc='upper center')