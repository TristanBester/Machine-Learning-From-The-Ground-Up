# Created by Tristan Bester
import numpy as np
import cvxopt


class SupportVectorMachine(object):
    '''
    A support vector machine is a classifier that finds the hyperplane that best
    separates the classes in the training data that the model is fitted to. The model
    predicts into which class an instance falls based on it's position relative
    to the separating hyperplane (the decision boundary).
    
    Args:
        kernel (string): The kernel function to be used by the SVM.
                         Kernels:
                        'linear' - Linear kernel
                        'polynomial' - Polynomial kernel
                        'rbf' - Gaussian Radial Basis Function kernel
        C (float): The regularization parameter that controls the trade-off between
                    the model's bias and variance. Large C value, high variance.
        gamma (float): A model hyperparameter that is used with the RBF kernel.
                       The magnitude of the value of gamma determines the bias
                       and variance of the model. Large gamma value, high bias.
                       Ignored by kernels other than the RBF kernel.
        degree (int): A model hyperparameter that is used with the polynomial
                      kernel. The value of the degree parameter is the degree of
                      the polynomial features that will be used for the model.
                      Ignored by kernels other than the Polynomial kernel.
    
    Attributes:
        kernel (string): The kernel function of the SVM.
        C (float): The value of the C hyperparameter of the SVM.
        gamma (float): The value of the gamma hyperparameter of the SVM.
        degree (int): The value of the degree hyperparameter of the SVM.
        alphas (numpy.ndarray): An array storing the Lagrangian multipliers of 
                              the model.
        support_alpha_indices (numpy.ndarray): An array storing the indices of the 
                                             Lagrangian multipliers for each of
                                             the support vectors in the alphas
                                             array.
        support_alphas (numpy.ndarray): An array storing the Lagrangian multipliers
                                      for each of the support vectors.
        support_vectors (numpy.ndarray): An array storing all of the support vectors.
        support_ys (numpy.ndarray): An array storing the labels of all of the support 
                                 vectors.
        b (float): The intercept of the model.
        w (numpy.ndarray): An array storing the weights of the model if the model
                           is making use of a linear kernel.
    '''
    def __init__(self, kernel='linear', C=None, gamma=1, degree=3):
        self.gamma = gamma
        self.degree = degree
        self.C = C
        
        # Only allow valid kernel functions.
        if kernel != 'linear' and kernel != 'polynomial' and kernel != 'rbf':
             raise Exception(f'{kernel}, is not a valid kernel function. Available kernels: linear,polynomial and rbf.')
        
        # Polynomial kernel SVM with degree 1 is equivalent to a linear kernel SVM.
        if degree == 1:
            self.kernel = 'linear'
        else:
            self.kernel = kernel
            
        if not C is None:
            if C != 0:
                self.C = float(C)
            else:
                self.C = None
                
            
    def linear_kernel(self, x1, x2):
        '''Linear kernel function.'''
        return np.dot(x1, x2)


    def polynomial_kernel(self, x, y):
        '''Polynomial kernel function.'''
        return (1 + np.dot(x, y)) ** self.degree
    
    
    def radial_basis_function_kernel(self, x, y):
        '''Gaussian Radial Basis Function.'''
        return np.exp(-self.gamma * (np.linalg.norm(x-y) ** 2))
     
        
    def apply_kernel_function(self,x,y):
        '''Apply the correct kernel function.'''
        if self.kernel == 'linear':
            return self.linear_kernel(x,y)
        elif self.kernel == 'polynomial':
            return self.polynomial_kernel(x,y)
        else:
            return self.radial_basis_function_kernel(x, y)
     
        
    def fit(self, X, y):
        n = X.shape[0]
          
        # Calculate the Gram matrix.
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i,j] = self.apply_kernel_function(X[i], X[j])
        
        # Setup necessary matrices to be used in the calculation of the Lagrangian
        # multipliers.
        P = cvxopt.matrix(np.outer(y,y) * K)        
        q = cvxopt.matrix(-np.ones((n,1)))
        A = cvxopt.matrix(y.reshape(1,-1))
        b = cvxopt.matrix(np.zeros((1,1)))
        
        if self.C is None:
            h = cvxopt.matrix(np.zeros((n, 1)))
            G = cvxopt.matrix(-np.eye(n))
        else:
            G_lower = -np.eye(n)
            G_upper = np.eye(n)
            h_lower = np.zeros((n,1))
            h_upper = np.full((n,1),self.C)
            G = cvxopt.matrix(np.concatenate((G_lower, G_upper), axis=0))
            h = cvxopt.matrix(np.concatenate((h_lower,h_upper), axis=0))
        
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)  
        
        # Extract the Lagrangian multipliers.
        self.alphas = (np.array(solution['x'])).flatten()
        
        self.support_alphas_indices = []
        self.support_alphas = []
        self.support_vectors = []
        self.support_ys = []
        
        # Determine the support vectors, their associated Lagrangian multipliers and labels.
        for i, x in enumerate(self.alphas):
            if x > 1e-4:
                self.support_alphas_indices.append(i)
                self.support_alphas.append(self.alphas[i])
                self.support_vectors.append(X[i])
                self.support_ys.append(y[i])
          
        self.support_alphas_indices = np.array(self.support_alphas_indices)
        self.support_alphas = np.array(self.support_alphas)
        self.support_vectors = np.array(self.support_vectors)
        self.support_ys = np.array(self.support_ys)
        
        isSV = self.alphas > 1e-5
        
        # Calculate the model intercept.
        self.b = 0
        for idx, i in enumerate(self.support_alphas_indices):
            self.b += self.support_ys[idx]
            kernel_vec = K[i]
            for j,k,l in zip(self.support_alphas, self.support_ys, kernel_vec[isSV]):
                self.b -= (j * k * l)
        self.b /= len(self.support_alphas)
        
        # Calculate the model weights if applicable.
        if self.kernel == 'linear':
            self.w = np.zeros(X.shape[1])
            for i in range(len(self.support_alphas)):
                multiplier = self.support_alphas[i] * self.support_ys[i]
                for x in range(X.shape[1]):
                    self.w[x] +=  multiplier * self.support_vectors[i][x]
        else:
            self.w = None
            
            
    def decision_function(self,X):
        '''Calculate and return the real valued prediction of the model.'''
        try:
            if self.w is not None:
                return X @ self.w + self.b
            else:
                preds = []
                for x in X:
                    summation = 0
                    for alpha,y,sv in zip(self.support_alphas, self.support_ys, self.support_vectors):
                        summation += alpha * y * self.apply_kernel_function(x,sv)
                    preds.append(summation)
                return preds + self.b   
        except AttributeError:
            print('Model not fitted, call \'fit\' with appropriate arguments before using model.')
            return -1
            
            
    def predict(self,X):
        '''Return the models predicted class for each of the given instances.'''
        try:
            return np.sign(self.decision_function(X))
        except AttributeError:
            print('Model not fitted, call \'fit\' with appropriate arguments before using model.')
            