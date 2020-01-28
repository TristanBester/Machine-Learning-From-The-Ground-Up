# Created by Tristan Bester.
from mlgroundup.boosting import GradientBoostingRegressor
import matplotlib.pyplot as plt
import numpy as np

# Create training set.
X = np.array([i for i in np.linspace(-10,10,25, endpoint=True)])
y = np.array([x**2 + np.random.rand() * 20 for x in X])

# Create colourmap.
col = plt.cm.jet(np.linspace(0,1,50, endpoint=True))   

# Train models and plot predictions.
for i in range(1, 50, 1):
    grad = GradientBoostingRegressor(n_estimators=i)
    grad.fit(X,y)
    preds = np.array([grad.predict(x) for x in X])
    plt.plot(X, preds, c=col[i])

fig = plt.gcf()
fig.set_size_inches((10,7))
sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=1, vmax=50))
cbar = fig.colorbar(sm)
cbar.set_label('Number of estimators:')
plt.scatter(X, y, c='black', marker='o')
plt.xlabel('Feature one (Dimension one):')
plt.ylabel('Feature two (Dimension two):')
plt.title('Gradient Boosting Regressors with varying number of estimators:')


