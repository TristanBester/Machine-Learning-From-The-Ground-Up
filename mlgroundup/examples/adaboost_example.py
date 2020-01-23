# Created by Tristan Bester.
from mlgroundup.boosting import AdaBoost
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Create training set.
X,y = make_blobs(n_samples=100, n_features=3, centers=2, cluster_std=4, random_state=42)
accuracy = []

for i in range(0, 1000, 20):
    # Create classifier with n_estimators equal to the loop variable.
    ada = AdaBoost(n_estimators=i)
    ada.fit(X,y)
    y2 = [ada.predict(x) for x in X]
    acc = (y2 == y).sum()
    # Store the accuracy of the classifier.
    accuracy.append(acc)

# Plot the accuracies of the classifiers with varying number of estimators.
fig = plt.gcf()
fig.set_size_inches((10,7))
plt.plot(range(0,1000,20), accuracy, c='fuchsia')
plt.xlabel('Number of decision stumps:')
plt.ylabel('Accuracy score (percentage):')
plt.title('Accuracy of AdaBoost classifiers with varying number of estimators:')