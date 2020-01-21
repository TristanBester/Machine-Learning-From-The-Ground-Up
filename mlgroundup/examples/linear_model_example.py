# Created by Tristan Bester
from mlgroundup.supervised import LinearRegression, RidgeRegression, LassoRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
np.random.seed(42)

    
'''
--- 1 ---
Visualize the effect of the various forms of regularization on the coefficients
of the model
'''

# Generate training data   
X = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5]])  
y = np.array([1 + i + 4*j + k + x + np.random.sample() for i,j,k,x in X])

lin_coef = []
ridge_coef = []
lasso_coef =[]


# Get model coefficients at 10 stages during training (each 10% of total epochs).
for i in range(0,1000,100):
    lin_model = LinearRegression(0.0001, i)
    ridge_model = RidgeRegression(0.01, i, 10)
    lasso_model = LassoRegression(0.01, i, 1)
    
    lin_model.fit(X,y)
    ridge_model.fit(X,y)
    lasso_model.fit(X,y)
    
    lin_coef.append(lin_model.weights)
    ridge_coef.append(ridge_model.weights)
    lasso_coef.append(lasso_model.weights)
 
    
# Plot the coefficients, grouping all coefficients of one model by colour.
plt.xlabel('Iterations (Hundreds)')
plt.ylabel('Coefficient value')
plt.title('Model coefficient comparison')

plt.plot(range(len(lin_coef)), lin_coef, c='g', label='Linear Regression')
plt.plot(range(len(ridge_coef)), ridge_coef, c='r', label='Ridge Regression')
plt.plot(range(len(lasso_coef)), lasso_coef, c='b', label='Lasso Regression')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.gcf().set_size_inches(8,5)


  
'''
--- 2 ---
Create and save an animation that illustrates the training process
of the model
'''

# Generate training data
X_anim = np.arange(30)
X_anim  = X_anim .reshape(-1, 1)
y_anim = np.array([2*i + 10 + np.random.sample() * 10 for i in X_anim])  

lin_model = LinearRegression(0.0001, 1000)

# Fit model for training plot
linear_train_partial = lin_model.fit(X_anim, y_anim, True) 
# Fit model for animation
linear_train_all = lin_model.fit(X_anim, y_anim, True, True)


def init():
    line.set_data([], [])
    return line,

def animate(i):
    X = range(len(linear_train_all[i]))
    y = linear_train_all[i]
    line.set_data(X, y)
    line.set_label('Prediction')
    plt.legend()
    return line,
 
    
x_lim = X_anim.max()
y_lim = y_anim.max()

fig = plt.figure()
ax = plt.axes(xlim=[0,x_lim],ylim=[0,y_lim])
ax.scatter(X_anim, y_anim, c='black', label='Target')
line, = ax.plot([], [], lw=3)

plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.rc('figure', titlesize=20)
plt.legend()


# Run animation
anim = FuncAnimation(fig, animate, init_func=init,
                           frames=np.arange(0,len(linear_train_all), 10), interval=5, blit=True)

anim.save('Linear_Regression.gif', writer='imagemagick')
plt.clf()



'''
--- 3 ---
Display a plot that illustrates the training process
'''

def create_training_plot(X, y, training_data, x_label=None, y_label=None, heading=None, output_file=None):
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
 

# Display training plot
create_training_plot(X_anim, y_anim, linear_train_partial, 'X','y','Linear Regression', 'Linear_Regression.png')
