import numpy as np
import cvxopt
from sklearn.datasets import make_blobs


#split up into nice little methods

class SupportVectorMachine(object):
    def __init__(self, kernel='linear', C=None, gamma=1, degree=3):
        self.gamma = gamma
        self.degree = degree
        self.C = C
        if degree == 1:
            self.kernel = 'linear'
        else:
            self.kernel = kernel
        
        if not C is None:
            print('in')
            self.C = float(C)
            
    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def polynomial_kernel(self, x, y):
        return (1 + np.dot(x, y)) ** self.degree
    
    def radial_basis_function_kernel(self, x, y):
        return np.exp(-self.gamma * (np.linalg.norm(x-y) ** 2))
            
    def apply_kernel_function(self,x,y):
        if self.kernel == 'linear':
            return self.linear_kernel(x,y)
        elif self.kernel == 'polynomial':
            return self.polynomial_kernel(x,y)
        else:
            return self.radial_basis_function_kernel(x, y)
        
    def fit(self, X, y):
        n = X.shape[0]
           
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i,j] = self.apply_kernel_function(X[i], X[j])
                 
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
        
        self.alphas = (np.array(solution['x'])).flatten()
        self.support_alphas_indices = []
        self.support_alphas = []
        self.support_vectors = []
        self.support_ys = []
         
        for i, x in enumerate(self.alphas):
            if x > 1e-4:
                self.support_alphas_indices.append(i)
                self.support_alphas.append(self.alphas[i])
                self.support_vectors.append(X[i])
                self.support_ys.append(y[i])
          
        self.support_alphas_indices = np.array(self.support_alphas_indices) #inidices in self.alphas
        self.support_alphas = np.array(self.support_alphas)
        self.support_vectors = np.array(self.support_vectors)
        self.support_ys = np.array(self.support_ys)
        
        isSV = self.alphas > 1e-5
        
        
        self.b = 0
        for idx, i in enumerate(self.support_alphas_indices):
            self.b += self.support_ys[idx]
            kernel_vec = K[i]
            for j,k,l in zip(self.support_alphas, self.support_ys, kernel_vec[isSV]):
                self.b -= (j * k * l)

        self.b /= len(self.support_alphas)
        
        if self.kernel == 'linear':
            self.w = np.zeros(X.shape[1])
            
            for i in range(len(self.support_alphas)):
                multiplier = self.support_alphas[i] * self.support_ys[i]
                for x in range(X.shape[1]): # loop through each feature adding the value to the appropriate weight
                    self.w[x] +=  multiplier * self.support_vectors[i][x]
        else:
            self.w = None
            
            
            
            
    def decision_function(self,X):
        
        #test to see if fitted model
        
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
            
            
    def predict(self,X):
        return np.sign(self.decision_function(X))
            
            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            