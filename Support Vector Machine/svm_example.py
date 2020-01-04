import support_vector_machine 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from sklearn.datasets import make_blobs
import seaborn as sns
sns.set()

def generate_test_data(n_samples=100, std=0.5):
    X, y = make_blobs(n_samples=n_samples, n_features=2, centers=2,
                  random_state=0, cluster_std=std)
    
    y[y == 0] = -1
    tmp = np.ones(len(X))
    y = tmp * y
    return X,y

def generate_test_data_3D(n_samples=100, std=0.5):
    X, y = make_blobs(n_samples=n_samples, n_features=3, centers=2,
                  random_state=0, cluster_std=std)
    
    y[y == 0] = -1
    tmp = np.ones(len(X))
    y = tmp * y
    return X,y

def create_contour_plot(svm,X,y, kernel='Linear'):
    train_set = X.copy()
    
    x_min = X[:,0].min()
    x_max = X[:,0].max()
    y_min = X[:,1].min()
    y_max = X[:,1].max()

    
    X1, X2 = np.meshgrid(np.linspace(x_min,x_max,50), np.linspace(y_min,y_max,50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = svm.decision_function(X).reshape(X1.shape)
   
    
    plt.scatter(train_set[:,0], train_set[:,1], c=y, cmap='winter')
    cp1 = plt.contour(X1, X2, Z, [-1,0.0,1], colors=['black','fuchsia','black'], linewidths=1)

    
    plt.clabel(cp1, inline=True, fontsize=10)
    
    plt.xlabel('Feature one (Dimension one)')
    plt.ylabel('Feature two (Dimension two)')
    plt.title(f'Support Vector Machine\nKernel:{kernel}')
    plt.show()
    
    
def create_filled_contour_plot(svm, X, y, kernel='linear'):
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
    plt.title(f'Support Vector Machine\nKernel:{kernel}')
    plt.colorbar(cp)
    plt.show()
    
    

#Linearly separable demonstration
X,y = generate_test_data(std=0.3)
svm = support_vector_machine.SupportVectorMachine()
svm.fit(X,y)
create_contour_plot(svm,X,y)
       
#Non-linearly separable demonstration
X,y = generate_test_data(std=0.98)
svm = support_vector_machine.SupportVectorMachine(C=1.5)
svm.fit(X,y)
create_contour_plot(svm,X,y)

# Polynomial svm demonstration
X,y = generate_test_data(std=0.98)
svm = support_vector_machine.SupportVectorMachine(kernel='polynomial', degree=3, C=1)
svm.fit(X,y)
create_contour_plot(svm,X,y, 'Polynomial')

# Radial Basis Function kernel svm demonstration
X,y = generate_test_data(std=0.9)
svm = support_vector_machine.SupportVectorMachine(kernel='rbf', gamma=1, C=4)
svm.fit(X,y)
create_contour_plot(svm,X,y, 'Gaussian Radial Basis Function')


X,y = generate_test_data(std=0.9)
svm = support_vector_machine.SupportVectorMachine(kernel='rbf', gamma=1, C=4)
svm.fit(X,y)
#create_contour_plot(svm,X,y, 'Gaussian Radial Basis Function')
create_filled_contour_plot(svm,X,y, 'Gaussian Radial Basis Function')


X,y = generate_test_data_3D(std=0.7)

train_set = X.copy()


svm = support_vector_machine.SupportVectorMachine()
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

#ani = animation.FuncAnimation(fig, update, range(185,340), interval=50, blit=False)
#ani.save('matplot003.gif', writer='imagemagick')




























'''

plt.scatter(X[:,0], X[:,1], c=y, cmap='winter')





plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

plt.axis("tight")
plt.show()
'''





'''
preds = svm.predict(X,X,y)
preds2 = svm.decision_function(X,X,y)
print((preds-y).sum())



idx = y != preds

for i,x in enumerate(idx):
    if x:
        plt.scatter(X[i,0], X[i,1], c='r')


for i,j in enumerate(svm.alphas):
    if j > 1e-4:
        plt.scatter(X[i,0], X[i,1], c='fuchsia', label='Support vector')



#print(svm.alphas)


preds = svm.predict(X,X,y)
pred2 = svm.predict_proba(X,X,y)
#print(pred2)
#print(preds)
#print(y)
print((preds-y).sum())

idx = y != preds

for i,x in enumerate(idx):
    if x:
        plt.scatter(X[i,0], X[i,1], c='r')
        

for i,j in enumerate(svm.alphas):
    if j > 1e-4:
        plt.scatter(X[i,0], X[i,1], c='fuchsia', label='Support vector')

'''




















     
'''    
X, y = make_blobs(n_samples=10, n_features=2, centers=2,
                  random_state=0, cluster_std=0.95)
y[y == 0] = -1
tmp = np.ones(len(X))
y = tmp * y

X = X * 1.0
y = y *1.0 

    
    
svm = svm_cvx.SupportVectorMachine('polynomial', C=0, degree=1, gamma=0, r =10)
svm.fit(X,y)
print(svm.alphas)


z = lambda x,y: (-svm.b -svm.w[0]*x - svm.w[1]*y) / svm.w[2]

tmp = np.linspace(-1,2.5,10)
x,y = np.meshgrid(tmp,tmp)  
fig = plt.figure()


ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y[:], c=y, cmap='winter')

# load some test data for demonstration and plot a wireframe
#z = lambda xi,x,y: svm.predict(xi, x, y)



tmp = np.linspace(-1,2.5,10)
tmp2 = np.linspace(-0.5, 3, 10)
xx,yy = np.meshgrid(tmp,tmp2)

#print(xx)

print(X[0])

for i in xx.flatten():
    for j in yy.flatten():
        print(np.c_[i,j][0])
        print(svm.predict(np.c_[i,j][0], X, y))
        break
    break
        
    
    
    

'''

'''
z = []

for x in xx:
    for y in yy:
        sample = np.c_[x,y]
        
        print(sample)
        z.append(svm.predict(np.c_[x,y], X, y))
        
        
        
        
z = np.array(z)
z = z.reshape(xx.shape)    
print(z)


#ax.plot_surface(xx, yy, z , cmap='jet', alpha=1)
 '''   
    

    
'''
def update_plot(degree):
    ax.clear()
    ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='winter', alpha=0.6)
    
    ax.view_init(azim=degree * 0.6)
'''


    
    

#ani = animation.FuncAnimation(fig, update_plot, frames=np.arange(1,50,1), interval=100)
#ani.save('testing.gif', writer='imagemagick')    
    
    
    
    
    

'''

X_min_one = X[:,0].min()
X_max_one = X[:,0].max()
X_min_two = X[:,1].min()
X_max_two = X[:,1].max()
'''
    
    
    
    
    
    
    
    
    
    