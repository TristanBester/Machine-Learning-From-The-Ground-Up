# Supervised-Learning-Models-From-Scratch
## Purpose
The purpose of this project is to illustrate how some of the models used in supervised machine learning work. The implementations have been designed so as to facilitate understanding of the model and not in manner that prioritizes computational efficiency.

## Table of contents
- [Supervised-Learning-Models-From-Scratch]
  + [Linear Regression and Regularization Techniques](#linear-regression-and-regularization-techniques)
    * [Linear Regression Examples](#linear-regression-examples)
  + [Polynomial Regression](#polynomial-regression)
    * [Polynomial Regression Examples](#polynomial-regression-examples)
  + [Support Vector Machine](#support-vector-machine)
    * [Support Vector Machine Examples](#support-vector-machine-examples)
  + [Decision Trees (Classification and Regression)](#decision-trees)
    * [Classification Decision Tree Example](#classification-decision-tree-example)
    * [Regression Decision Tree Example](#regression-decision-tree-example)

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




  
  
  
