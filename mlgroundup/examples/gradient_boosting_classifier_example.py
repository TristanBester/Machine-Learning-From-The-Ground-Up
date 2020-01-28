# Created by Tristan Bester.
from mlgroundup.boosting import GradientBoostingClassifier
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Create training set.
X, y = make_blobs(n_samples=100, n_features=10, centers=2, cluster_std=5,
                  random_state=42)
diffs = []

# Train models and store the deviations of the model's predictions from
# the target values.
for x,i in enumerate(range(3, 50, 5)):
    grad = GradientBoostingClassifier(max_depth=2, n_estimators=i, learning_rate=0.1)
    grad.fit(X, y)
    y2 = [grad.predict_proba(x) for x in X]
    
    # Calculate and store the deviations of the model's predictions from the 
    # target values.
    diffs.append(abs((y - y2).sum()))

fig = plt.gcf()
fig.set_size_inches((10,7))
plt.xlabel('Number of estimators in the ensemble:')
plt.ylabel('Deviations of the model\'s predicted\nprobabilities from the target values:')
plt.title('Gradient Boosting Classifier:')
plt.plot(range(3, 50, 5), diffs, c='fuchsia',
         marker='X', markeredgecolor='black', markersize=10, markeredgewidth=1.5)