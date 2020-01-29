# Created by Tristan Bester.
import numpy as np
import random

def train_test_split(X, y, test_size=0.4, random_state=None):
    if random_state is not None:
        random.seed(random_state)
        
    test_idx = random.sample(population=range(len(X)), k=int(len(X)*test_size))
    train_idx= [x for x in range(len(X)) if x not in test_idx]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]