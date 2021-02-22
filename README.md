# Machine Learning from Scratch

This project is built to implement the machine learning algorithms from scratch to deepen my understand of them.

# Decision Tree

The entropy-based decision tree is implemented in models/DecisionTreeClassifier.py. 
MNIST is used to testify the performance of the implementation.
the accuracy of this implementation on is 87.25%
while the accuracy of sklearn is 87.37%

The CSV files can be downloaded from 

https://www.kaggle.com/oddrationale/mnist-in-csv

```python
python runDecisionTreeOnMNIST.py
```

# Random Forest

Based on the DecisionTreeClassifier implemented in this project,
Random forest is built.
Same as decision tree, MNIST is used to evaluate the performance.
the accuracy of this implementation is 90.8%,
while the accuracy of sklearn is 90.71%

```python
python runRandomForestOnMNIST.py
```

# Logistic Regression

Logistic regression is implemented with cross entropy loss and gradient descent using tensorflow.
the accuracy of this implementation is 91.98%,
the accuracy of sklearn is 91.75%

```python
python run runLogisticRegressionOnMNIST.py
```

#AdaBoost

Implemented AdaBoost binary classifier based on DecisionTree of Sklearn (since the decision tree of this project doesn't support sample weight).
Samples of two digits (1 and 2) are extracted from MNIST and trained the classifier. The classifier consists of 5 decision trees with maximum height of 3 to 
suppress the performance of the decision tree. Experiment shows the accuracy of the five decision trees are [94.9%, 92.4%, 85.3%, 68.3%, 89.5%] and the 
overall accuracy of the AdaBoost classifier is 97.97%.

```python
python runAdaBoostOnMnist.py
```