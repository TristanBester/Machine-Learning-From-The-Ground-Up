# Created by Tristan Bester
from mlgroundup.supervised import SupportVectorMachine
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from sklearn.datasets import make_blobs
import seaborn as sns
sns.set()

def generate_test_data(n_samples=100, std=0.5):
    '''Generate a test dataset with two dimensional instances.'''
    X, y = make_blobs(n_samples=n_samples, n_features=2, centers=2,
                  random_state=0, cluster_std=std)
    
    y[y == 0] = -1
    tmp = np.ones(len(X))
    y = tmp * y
    return X,y

def generate_test_data_3D(n_samples=100, std=0.5):
    '''Generate a test dataset with three dimensional instances.'''
    X, y = make_blobs(n_samples=n_samples, n_features=3, centers=2,
                  random_state=0, cluster_std=std)
    
    y[y == 0] = -1
    tmp = np.ones(len(X))
    y = tmp * y
    return X,y

def create_contour_plot(svm,X,y, kernel='Linear'):
    '''Generate a contour plot for the given model using the dataset X with class labels y.'''
    train_set = X.copy()
    
    x_min = X[:,0].min()
    x_max = X[:,0].max()
    y_min = X[:,1].min()
    y_max = X[:,1].max()

    
    X1, X2 = np.meshgrid(np.linspace(x_min,x_max,50), np.linspace(y_min,y_max,50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = svm.decision_function(X).reshape(X1.shape)
    
    fig = plt.gcf()
    fig.set_size_inches((10,7))
   
    
    plt.scatter(train_set[:,0], train_set[:,1], c=y, cmap='winter')
    cp1 = plt.contour(X1, X2, Z, [-1,0.0,1], colors=['black','fuchsia','black'], linewidths=1)

    
    plt.clabel(cp1, inline=True, fontsize=10)
    
    plt.xlabel('Feature one (Dimension one)')
    plt.ylabel('Feature two (Dimension two)')
    plt.title(f'Support Vector Machine\nKernel: {kernel}')
    plt.show()
    
    
def create_filled_contour_plot(svm, X, y, kernel='linear'):
    '''Generate a  filled contour plot for the given model using the dataset X with class labels y.'''
    train_set = X.copy()
    
    x_min = X[:,0].min()
    x_max = X[:,0].max()
    y_min = X[:,1].min()
    y_max = X[:,1].max()

    
    X1, X2 = np.meshgrid(np.linspace(x_min,x_max,50), np.linspace(y_min,y_max,50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = svm.decision_function(X).reshape(X1.shape)
    
    
    fig = plt.gcf()
    fig.set_size_inches((10,7))
    axes = plt.gca()
    axes.set_xlim([x_min,x_max])
    axes.set_ylim([y_min,y_max])
    
    cp = plt.contourf(X1,X2,Z, cmap='cool')
    plt.scatter(train_set[:,0], train_set[:,1], c=y, cmap='binary')
    
    plt.xlabel('Feature one (Dimension one)')
    plt.ylabel('Feature two (Dimension two)')
    plt.title(f'Support Vector Machine\nKernel: {kernel}')
    plt.colorbar(cp)
    plt.show()
    
    

'''
--- 1 ---
Display a plot that illustrates the margin and decision boundary of a SVM
with a linear kernel on linearly separable data.
'''

X,y = generate_test_data(std=0.3)
svm = SupportVectorMachine(C=0)
svm.fit(X,y)
create_contour_plot(svm,X,y)   
 

'''
--- 2 ---
Display a plot that illustrates the margin and decision boundary of a SVM
with a linear kernel on non-linearly separable data.
'''

X,y = generate_test_data(std=0.98)
svm = SupportVectorMachine(C=1.5)
svm.fit(X,y)
create_contour_plot(svm,X,y)


'''
--- 3 ---
Display a plot that illustrates the margin and decision boundary of a SVM
with a polynomial kernel on non-linearly separable data.
'''

X,y = generate_test_data(std=0.98)
svm = SupportVectorMachine(kernel='polynomial', degree=3, C=1)
svm.fit(X,y)
create_contour_plot(svm,X,y, 'Polynomial')


'''
--- 4 ---
Display plots that illustrates the margin and decision boundary of a SVM
with a Radial Basis Function kernel on non-linearly separable data.
'''

X,y = generate_test_data(std=0.9)
svm = SupportVectorMachine(kernel='rbf', gamma=1, C=4)
svm.fit(X,y)
create_contour_plot(svm,X,y, 'Gaussian Radial Basis Function')
create_filled_contour_plot(svm,X,y, 'Gaussian Radial Basis Function')


'''
--- 5 ---
Create an animation to illustrate the decision boundary hyperplane of a linear
kernel SVM in three dimensions.
'''

X,y = generate_test_data_3D(std=0.7)

train_set = X.copy()

svm = SupportVectorMachine()
svm.fit(X,y)

fig = plt.figure()
ax = Axes3D(fig)
ax.view_init(azim=340, elev=30)

x_min = X[:,0].min()
x_max = X[:,0].max()
y_min = X[:,1].min()
y_max = X[:,1].max()

z = lambda x,y: (-svm.b - x * svm.w[0] - y * svm.w[1])/svm.w[2]
X1, X2 = np.meshgrid(np.linspace(-1,3,15), np.linspace(-2,4,15))

def update(degree):
    ax.view_init(azim=degree)

points = ax.scatter(train_set[:,0], train_set[:,1],train_set[:,2], c=y, cmap='winter')
plane = ax.plot_surface(X1,X2,z(X1,X2), cmap='winter')

ani = animation.FuncAnimation(fig, update, range(185,340), interval=50, blit=False)
ani.save('Linear3D.gif', writer='imagemagick')