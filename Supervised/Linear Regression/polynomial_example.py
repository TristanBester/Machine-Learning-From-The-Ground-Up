import linear_model
import polynomial_features
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
np.random.seed(42)

'''
--- 1 ---
Create a plot to visualize the training process.
'''

def create_training_plot(y, training_data, x_label=None, y_label=None, heading=None, output_file=None):
    '''Creates a plot that can be used to visualize the training process'''
    
    fig = plt.gcf()
    fig.set_size_inches(8,5)
    
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if heading:
        plt.title(heading)
        plt.rc('figure', titlesize=20)
        
    plt.scatter(range(len(y)), y, c='black', label='Target')
    plt.plot(range(len(y)), training_data[-1], c='r', label='Current model')
    plt.plot(range(len(y)), training_data[0], linestyle='--', c='b', label='Start')
    
    for i in training_data[:-2]:
        plt.plot(range(len(y)), i, linestyle=':', c='g', label='Training')
        
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    if output_file:
        plt.savefig(output_file)
    
    plt.show()


# Generate training data.
X_multi = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]])  
y_multi = np.array([1 + 1*i + 4*(i**2) + 4*j + 5 * (j**2) + k**2 + np.random.sample()*1 for i,j,k in X_multi])

# Preprocess the training data to include polynomial features.
poly = polynomial_features.PolynomialFeatures(degree=3)
poly.fit()
X_poly_multi = poly.transform(X_multi)

# Create and train model.
lasso_model = linear_model.LassoRegression(0.0000001, 50, 5)
train = lasso_model.fit(X_poly_multi, y_multi, True)

create_training_plot(y_multi, train, x_label='X', y_label='y', heading='Polynomial Regression')



'''
--- 2 ---
Create an animation that demonstrates how the predictions of the model change as
it is trained.
'''
# Generate training data.
X = np.arange(-12,12)
X = X.reshape(-1, 1)
y = np.array([0.01*(i**3) + 3 + np.random.sample() * 3 for i in X])  

# Preprocess the training data to include polynomial features.
poly2 = polynomial_features.PolynomialFeatures(degree=3)
poly2.fit()
X_poly = poly2.transform(X)

# Create and train the model.
lin_model = linear_model.LinearRegression(0.000001, 10000)
train_linear_all = lin_model.fit(X_poly, y, True)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data(X, train_linear_all[i])
    line.set_label('Prediction')
    plt.legend()
    return line,

fig = plt.figure()
ax = plt.axes(xlim=[X.min(),X.max()], ylim=[y.min(),y.max()])
ax.scatter(X, y, c='black', label='Target')
line, = ax.plot([], [], lw=3)

plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.rc('figure', titlesize=20)
plt.legend()

anim = FuncAnimation(fig, animate, init_func=init,
                           frames=np.arange(0, len(train_linear_all)), interval=500, blit=True)

anim.save('Polynomial_Regression.gif', writer='imagemagick')
