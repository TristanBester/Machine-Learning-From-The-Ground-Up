# Created by Tristan Bester.
from mlgroundup.supervised import ElasticNet
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt 

# Create training set.
X,y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Create model.
net = ElasticNet(eta=0.01, n_iters=10, alpha=0.5, l1_ratio=0.5)

# Train model.
net.fit(X,y)

# Calculate model predictions.
preds = [net.predict(x) for x in X]

# Plot model predictions.
plt.scatter(X,y, c='black', label='Target')
plt.plot(X, preds, c='fuchsia', label='Prediction')
plt.title('Elastic Net:')
plt.xlabel('Feature one (Dimension one):')
plt.ylabel('Feature two (Dimension two):')
plt.legend()