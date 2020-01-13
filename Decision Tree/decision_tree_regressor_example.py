# Created by Tristan Bester.
import decision_tree
import numpy as np
import matplotlib.pyplot as plt

# Create the training set.
X = np.linspace(-10,10, 100)
y = [x**2 + np.random.rand() * 25 for x in X]
X = np.array(X)
y = np.array(y)

# Instantiate objects of the DecisionTreeRegressor class.
tree_one = decision_tree.DecisionTreeRegressor()
tree_two = decision_tree.DecisionTreeRegressor(max_depth=3)
tree_three = decision_tree.DecisionTreeRegressor(min_samples_split=40)

# Fit each model to the training set.
tree_one.fit(X,y)
tree_two.fit(X,y)
tree_three.fit(X,y)

# Get the predictions of each model.
preds_one = []
preds_two = []
preds_three = []

for i in X:
    preds_one.append(tree_one.predict(i))
    preds_two.append(tree_two.predict(i))
    preds_three.append(tree_three.predict(i))


# Plot the predictions of each model on the same set of axes.
fig = plt.gcf()
fig.set_size_inches((10,7))

plt.scatter(X, y, s=10, c='black', marker='x', label='Target')
plt.plot(X,preds_one,linewidth=0.5, c='r', label='No regularization')
plt.plot(X,preds_two, c='lawngreen', label='max_depth = 3')
plt.plot(X,preds_three, c='b', label='min_samples_split = 40')


plt.title('Decision tree regressors with various regularization parameters:')
plt.xlabel('Feature one (Dimension one)')
plt.ylabel('Feature two (Dimension two)')
plt.legend(loc='upper center')