# Created by Tristan Bester
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)


#Superclass for  models.
class LinearRegression(object):
    '''
    Fits model with coefficients w = (w0,w1,...,wn) to minimise the MSE cost 
    function between the observed targets in the dataset, and the targets
    predicted by the linear approximation.
    
    Args:
        eta (float): The learing rate of the model.
        n_iters (int): Number of training epochs.
    
    Attributes:
        eta (float): The learning rate of the model
        n_iters (int): Number of epochs performed when training model.
        weights (np.ndarray): Model coefficients.
    '''
    
    def __init__(self, eta, n_iters):
        self.eta = eta 
        self.n_iters = n_iters
    
    def init_weights(self, X):
        '''Initialise model weights with random within interval [0,1)'''
        X = X.reshape(len(X), -1)
        self.weights = np.array([np.random.sample() for i in range(X.shape[1] + 1)])
        
    def prepare_dataset(self, X):
        '''Prepend one to each instance for bias weight'''
        if X.ndim < 2:
            if not isinstance(X, np.ndarray):
                X = np.array([X])
            X = X.reshape(len(X), -1)
            return np.insert(X, 0, 1)
        else:
            ones = np.full((X.shape[0], 1), 1)
            return np.c_[ones,X]
            
        
    def fit(self, X, y, return_training=False, return_all=False):
        self.init_weights(X)
        X = self.prepare_dataset(X)
        
        if return_training:
            training_preds = []
    
        # stochastic gradient descent
        for i in range(self.n_iters):
            if return_training:
                # ensure only 10 epochs are recorded
                if return_all:
                    record_epoch = True
                else:
                    record_epoch =  i % (0.1 * self.n_iters) == 0
            
            for instance,target in zip(X,y):
                # calculate model prediction
                y_hat = np.dot(self.weights, instance)
                # gradient of cost function w.r.t. each weight
                grads = np.multiply(instance, (2*(y_hat - target)))
                
                # adjust weights to decrease cost
                self.weights -= np.multiply(self.eta, grads)
                
            if return_training and record_epoch:
                training_preds.append(X @ self.weights.reshape(len(self.weights),-1))

        if return_training:
            return training_preds
        

    def predict(self, X):
        try:
            X = self.prepare_dataset(X)
            return np.dot(self.weights,X)
        except AttributeError:
            print('Model not fitted, call \'fit\' with appropriate arguments before using model.')
                    
        
class RidgeRegression(LinearRegression):
    '''
    A regularized version of Linear Regression model. Model coefficients are 
    regularized through Tikhonov regularization in which the regularization 
    term, equal to the hyperparameter alpha multiplied by half the square of 
    the l2 norm of the weight vector, is added to the cost function (MSE) while
    training the model.
    
    Args:
        eta (float): The learning rate of the model.
        n_iters (int): Number of training epochs.
        alpha (float): Hyperparameter that controls the amount of regularization
                       used.
        
    Attributes:
        eta (float): The learning rate of the model
        n_iters (int): Number of epochs performed when training model.
        weights (np.ndarray): Model coefficient
        alpha (float): The amount of regularization used in the model.
    '''
           
    def __init__(self, eta, n_iters, alpha):
        super().__init__(eta, n_iters)
        self.alpha = alpha
    
    def fit(self, X, y, return_training=False):
        self.init_weights(X)
        X = self.prepare_dataset(X)
        
        if return_training:
            training_preds = []
    
        # stochastic gradient descent
        for i in range(self.n_iters):
            if return_training:
                # ensure only 10 epochs are recorded
                record_epoch = i % (0.1 * self.n_iters) == 0
            
            for instance,target in zip(X,y):
                # calculate model prediction
                y_hat = np.dot(self.weights, instance)
                # gradient of cost function w.r.t. each weight
                grads = np.multiply(instance, (2*(y_hat - target)))
                
                # calculate regularization term
                term = self.alpha * self.weights[1:]
                # prevent bias(intercept) from being regularized
                term = np.insert(term,0,0)
                # add regularization term
                grads += term
               
                # adjust weights to decrease cost
                self.weights -= np.multiply(self.eta, grads)

            if return_training and record_epoch:
                training_preds.append(X @ self.weights.reshape(len(self.weights),-1))
        
        if return_training:
            return training_preds
 
        
    
class LassoRegression(LinearRegression):
    '''
    Least Absolute Shrinkage and Selection Operator Regression. A regularized 
    version of Linear Regression. Model coefficients are regularized through 
    the addition of the regularization term, equal to the hyperparameter alpha
    multiplied by the l1 norm of the weight vector, to the cost function
    (MSE) while training the model.
    
    Args:
        eta (float): The learning rate of the model.
        n_iters (int): Number of training epochs.
        alpha (float): Hyperparameter that controls the amount of regularization
                       used.
        
    Attributes:
        eta (float): The learning rate of the model
        n_iters (int): Number of epochs performed when training model.
        weights (np.ndarray): Model coefficient
        alpha (float): The amount of regularization used in the model.  
    '''
    
    def __init__(self, eta, n_iters, alpha):
        super().__init__(eta, n_iters)
        self.alpha = alpha

    def fit(self, X, y, return_training=False):
        self.init_weights(X)
        X = self.prepare_dataset(X)
        
        if return_training:
            training_preds = []
    
        # stochastic gradient descent
        for i in range(self.n_iters):
            if return_training:
                # ensure only 10 epochs are recorded
                record_epoch = i % (0.1 * self.n_iters) == 0
            
            for instance,target in zip(X,y):
                # calculate model prediction
                y_hat = np.dot(self.weights, instance)
                # gradient of cost function w.r.t. each weight
                grads = np.multiply(instance, (2*(y_hat - target)))
                
                # calculate regularization term
                signs = np.sign(self.weights)
                # prevent bias(intercept) from being regularized
                signs[0] = 1 
                # add regularization term
                grads += self.alpha * signs
                
                # adjust weights to decrease cost
                self.weights -= np.multiply(self.eta, grads)
                
            
            if return_training and record_epoch:
                training_preds.append(X @ self.weights.reshape(len(self.weights),-1))
        
        if return_training:
            return training_preds

    
class ElasticNet(LinearRegression):
    '''
    A regularized version of the Linear Regression model. Elastic Net is a combination
    of Ridge Regression and Lasso Regression. The coefficients of the model are
    regularized through the addition of a regularization term to the cost
    function (MSE) while training the model. The regularization term of Elastic
    Net is a combination of the Ridge Regression regularization term and the 
    Lasso Regression regularization term.
    
    Args:
        eta (float): The learning rate of the model.
        n_iters (int): Number of training epochs.
        alpha (float): Hyperparameter that controls the amount of regularization
                       used.
        l1_ratio (float): Hyperparameter that controls the influence of the Ridge
                         and Lasso regularization terms on the model. l1_ratio = 1 
                         is equivalent to Lasso Regression and l1_ratio = 0 is equivalent
                         to Ridge Regression.
        
    Attributes:
        eta (float): The learning rate of the model
        n_iters (int): Number of epochs performed when training model.
        weights (np.ndarray): Model coefficient
        alpha (float): The amount of regularization used in the model.
        l1_ratio (float): Hyperparameter that controls the influence of the Ridge
                         and Lasso regularization terms on the model. l1_ratio = 1 
                         is equivalent to Lasso Regression and l1_ratio = 0 is equivalent
                         to Ridge Regression.
    '''
    
    def __init__(self, eta, n_iters, alpha, l1_ratio):
        super().__init__(eta, n_iters)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
    
    
    def fit(self, X, y, return_training=False):
        self.init_weights(X)
        X = self.prepare_dataset(X)
        
        if return_training:
            training_preds = []
    
        # stochastic gradient descent
        for i in range(self.n_iters):
            if return_training:
                # ensure only 10 epochs are recorded
                record_epoch = i % (0.1 * self.n_iters) == 0
            
            for instance,target in zip(X,y):
                # calculate model prediction
                y_hat = np.dot(self.weights, instance)
                # gradient of cost function w.r.t. each weight
                grads = np.multiply(instance, (2*(y_hat - target)))
                
                # calculate Ridge regularization term
                ridge = self.alpha * self.weights[1:]
                # prevent bias(intercept) from being regularized
                ridge = np.insert(ridge,0,0)
                # add regularization term

                # calculate Lasso regularization term
                lasso = np.sign(self.weights)
                # prevent bias(intercept) from being regularized
                lasso[0] = 1 
                
                #add regularization terms
                grads += self.alpha * ((self.l1_ratio * lasso) + ((1 - self.l1_ratio) * ridge))
                
                # adjust weights to decrease cost
                self.weights -= np.multiply(self.eta, grads)
                
            
            if return_training and record_epoch:
                training_preds.append(X @ self.weights.reshape(len(self.weights),-1))
        
        if return_training:
            return training_preds
        
      
   
    
    
class LogisticRegression(LinearRegression):
    '''
    Fits model with coefficients w = (w0,w1,...,wn) to predict a binary dependent
    variable. This model is a binary classifier that predicts the probability
    that an instance belongs to a specific class and classifies the instance
    as belonging to the class of highest associated probability. Logarithmic
    loss is the cost function used by model.
    
    Args:
        eta (float): The learning rate of the model.
        n_iters (int): Number of training epochs.
        decision_boundary (float): Hyperparameter that controls the threshold 
                                   probability that must be passed for an instance
                                   to be classified as belonging to the positive
                                   class.
        
    Attributes:
        eta (float): The learning rate of the model
        n_iters (int): Number of epochs performed when training model.
        weights (np.ndarray): Model coefficient
        decision_boundary (float): Hyperparameter that controls the threshold 
                                   probability that must be passed for an instance
                                   to be classified as belonging to the positive
                                   class.
    '''
    
    def __init__(self, eta, n_iters, decision_boundary):
        super().__init__(eta, n_iters)
        self.decision_boundary = decision_boundary
    
    def __sigmoid_function(self, x):
        return 1/(1 + np.exp(-x))
    
    
    def fit(self, X, y):
        self.init_weights(X)
        X = self.prepare_dataset(X)
        
        # stochastic gradient descent
        for i in range(self.n_iters):
            for instance,target in zip(X,y):
                # calculate model prediction
                y_hat = np.dot(self.weights, instance)
                y_hat = self.__sigmoid_function(y_hat)
                # gradient of cost function w.r.t. each weight
                grads = np.multiply(instance, (2*(y_hat - target)))
                
                # adjust weights to decrease cost
                self.weights -= np.multiply(self.eta, grads)
    
    
    def predict(self, X):
        try:
            X = self.prepare_dataset(X)
            y_hat = np.dot(self.weights, X)
            y_hat = self.__sigmoid_function(y_hat)
            if y_hat < self.decision_boundary:
                return 0
            else:
                return 1
        except AttributeError:
            print('Model not fitted, call \'fit\' with appropriate arguments before using model.')
             
            
        