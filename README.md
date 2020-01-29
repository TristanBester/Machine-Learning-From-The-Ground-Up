# Machine-Learning-From-The-Ground-Up
## Purpose
The purpose of this project is to illustrate how some of the models used in machine learning work. The implementations have been designed so as to facilitate understanding of the model and not in manner that prioritizes computational efficiency.

## Table of contents
- [Machine-Learning-From-The-Ground-Up]
  + [Linear Regression and Regularization Techniques](#linear-regression-and-regularization-techniques)
    * [Linear Regression Examples](#linear-regression-examples)
  + [Polynomial Regression](#polynomial-regression)
    * [Polynomial Regression Examples](#polynomial-regression-examples)
  + [Support Vector Machine](#support-vector-machine)
    * [Support Vector Machine Examples](#support-vector-machine-examples)
  + [Decision Trees (Classification and Regression)](#decision-trees)
    * [Classification Decision Tree Example](#classification-decision-tree-example)
    * [Regression Decision Tree Example](#regression-decision-tree-example)
  + [Ensemble Learning](#ensemble-learning)
    * [Classification with Ensemble Models](#classification-with-ensemble-models)
    * [Classification with Ensemble Models Example](#classification-with-ensemble-models-example)
    * [Regression with Ensemble Models](#regression-with-ensemble-models)
    * [Regression with Ensemble Models Example](#regression-with-ensemble-models-example)
  + [Stacking](#stacking)
    * [Stacking Examples](#stacking-examples)
  + [Boosting](#boosting)
    * [AdaBoost](#adaboost)
    * [AdaBoost Example](#adaboost-example)
    * [Gradient Boosting](#gradient-boosting)
    * [Gradient Boosting Classification Example](#gradient-boosting-classification-example)
    * [Gradient Boosting Regression Example](#gradient-boosting-regression-example) 

## Linear Regression and Regularization Techniques
Linear regression fits a model with coefficients w = (w0,w1,...,wn) to minimise the mean squared error cost function between the observed targets in the dataset, and the targets predicted by the linear approximation. The linear model can be regularized with Ridge Regression, Lasso Regression and Elastic Net to decrease variance.

### Linear Regression Examples
<p align="center">
    <img src="https://machinelearningjourney.com/wp-content/uploads/2019/12/Linear_Regression.gif"\>
</p>
<p align="center">
    Figure 1: Training process of the Linear Regression model.
</p>

<p align="center">
    <img src="https://machinelearningjourney.com/wp-content/uploads/2019/12/Linear_Regression.png"\>
</p>
<p align="center">
    Figure 2: A visualization of how the accuracy of the model increases over training epochs.
</p>

<p align="center">
    <img src="https://machinelearningjourney.com/wp-content/uploads/2019/12/coef.png"\>
</p>
<p align="center">
    Figure 3: A simple visulization of how regularization techniques affect the coefficients of the model.
</p>

## Polynomial Regression
Polynomial regression fits a model with coefficients w = (w0,w1,...,wn) to minimise the mean squared error cost function between the observed targets in the dataset, and the targets predicted by the linear approximation. Polynomial regression is a powerful model able
to detect complex non-linear relationships between features in the dataset. 

### Polynomial Regression Examples
<p align="center">
    <img src="https://machinelearningjourney.com/wp-content/uploads/2019/12/Polynomial_Regression.gif"\>
</p>
<p align="center">
    Figure 1: Training process of the Polynomial Regression model.
</p>

<p align="center">
    <img src="https://machinelearningjourney.com/wp-content/uploads/2019/12/polynomial.png"\>
</p>
<p align="center">
    Figure 2: A visualization of how the accuracy of the model increases over training epochs.
</p>

## Support Vector Machine
A support vector machine is a classifier that finds the hyperplane that best separates the classes in the training data that the model is fitted to. The model predicts into which class an instance falls based on it's position relative to the separating hyperplane (the decision boundary).

### Support Vector Machine Examples

<p align="center">
    <img src="https://machinelearningjourney.com/wp-content/uploads/2020/01/Linear3D-1.gif"\>
</p>

<p align="center">
    Figure 1: A visualization of the separating hyperplane of a support vector machine in three dimensions.
</p>

<p align="center">
    <img src="https://machinelearningjourney.com/wp-content/uploads/2020/01/collage.png"\>
</p>
<p align="center">
    Figure 2: A visualizaton of the decision boundaries of support vector machines with various kernels.
</p>

<p align="center">
    <img src="https://machinelearningjourney.com/wp-content/uploads/2020/01/filled.png"\>
</p>
<p align="center">
    Figure 3: An illustration of the prediction confidence of a support vector machine with a Radial Basis Function kernel on 
  non-linearly separable data.
</p>

## Decision Trees
A decision tree is a tree-like model in which the target value of instances is predicted based on a series of attribute tests. Each instance moves from the root node, down the decision tree, until it reaches a leaf node at which point the target value of the instance is predicted. The path the instance follows to reach a leaf node is determined based on the result of a set of predetermined attribute tests.

### Classification Decision Tree Example
<p align="center">
    <img src="https://machinelearningjourney.com/wp-content/uploads/2020/01/Figure_1.png"\>
</p>

<p align="center">
    Figure 1: A visualization of the decision boundary of a decision tree.
</p>

### Regression Decision Tree Example
<p align="center">
    <img src="https://machinelearningjourney.com/wp-content/uploads/2020/01/Figure.png"\>
</p>

<p align="center">
    Figure 1: A visualization of the predictions of decision trees with various regularization parameters.
</p>

## Ensemble Learning
Ensemble models aggregate the predictions of multiple base estimators in order to make a final prediction. 
This repository includes implementations of various ensemble models. Implemetations are listed below.

### Classification with Ensemble Models
Classification ensemble models make use of various strategies in order to effectively aggregate the predictions
of multiple base estimators in order to make a final prediction. Classification ensemble models included:
VotingClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier.

### Classification with Ensemble Models Example
<p align="center">
    <img src="https://machinelearningjourney.com/wp-content/uploads/2020/01/collage-1.png"\>
</p>

<p align="center">
    Figure 1: A visualization of the decision boundaries of various classification ensemble models
    (VotingClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier).
</p>

### Regression with Ensemble Models
Regression ensemble models make use of various strategies in order to effectively aggregate the predictions
of multiple base estimators in order to make a final prediction. Regression ensemble models included:
VotingRegressor, BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor.

### Regression with Ensemble Models Example
<p align="center">
    <img src="https://machinelearningjourney.com/wp-content/uploads/2020/01/collage2.png"\>
</p>

<p align="center">
    Figure 1: A visualization of the predictions of various regression ensemble models
    (VotingRegressor, BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor).
</p>

## Stacking
A stacking model is an ensemble model in which the predictions of the
estimators in the ensemble are aggregated through the use of a final estimator,
the blender/meta learner.

### Stacking Examples
<p align="center">
    <img src="https://machinelearningjourney.com/wp-content/uploads/2020/01/coll.png" width="700" height="250"\>
</p>

<p align="center">
    Figure 1: A visualization of the predictions of a StackingClassifier and StackingRegressor.
</p>


## AdaBoost
An AdaBoost classifier is a classifier that trains estimators sequentially.
An initial estimator is fitted to the training set, the following estimator
is then fitted to a training set that places more emphasis on the instances
misclassified by the previous estimator. This process of fitting estimators
to correct the misclassifications of the previous estimator is repeated until
the ensemble contains the desired number of estimators.

### AdaBoost Example
<p align="center">
    <img src="https://machinelearningjourney.com/wp-content/uploads/2020/01/Figure_1-1.png"\>
</p>

<p align="center">
    Figure 1: A visualization of the accuracies of AdaBoost classifiers with varying number of estimators.
</p>


## Gradient Boosting
A gradient boosting model is an ensemble model in which estimators are
trained sequentially with each successive estimator trained to predict the 
pseudo-residuals of all of the estimators trained prior to it. Once the model
has been trained the predictions of all of the estimators in the ensemble
are aggregated to predict the target value of an instance.

### Gradient Boosting Classification Example
<p align="center">
    <img src="https://machinelearningjourney.com/wp-content/uploads/2020/01/Figure_2.png"\>
</p>

<p align="center">
    Figure 1: The deviations of the predictions of gradient boosting classifiers with varying number 
    of estimators from the target values.
</p>

### Gradient Boosting Regression Example
<p align="center">
    <img src="https://machinelearningjourney.com/wp-content/uploads/2020/01/Figure_3.png"\>
</p>

<p align="center">
    Figure 1: A visualization of the predictions of gradient boosting regressors with varying number of estimators.
</p>








  
  
  
