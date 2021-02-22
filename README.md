# Machine Learning from Scratch

This project is built to implement the machine learning algorithms from scratch to deepen my understand of them.

# Decision Tree

The entropy-based decision tree is implemented in models/DecisionTreeClassifier.py. 
MNIST is used to testify the performance of the implementation.
the accuracy of this implementation on is 87.3%
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