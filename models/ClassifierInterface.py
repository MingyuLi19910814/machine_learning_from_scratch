from abc import ABC, abstractmethod


'''
define the interface that all classifiers in this project should follow
'''
class Classifier(ABC):
    '''
    the input should be numpy array with shape (n, m),
    n is the number of sample and m is the feature dimension.
    '''
    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass


class BinaryClassifier(Classifier):
    '''
        the input should be numpy array with shape (n, m)
        the output should be numpy array with shape (n, 1) indicating whether the data is the target
    '''
    @abstractmethod
    def predict(self, x):
        pass

class MultiClassClassifier(Classifier):
    '''
            the input should be numpy array with shape (n, m)
            the output should be numpy array with shape (n, 1) indicating the classification of the data
    '''
    @abstractmethod
    def predict(self, x):
        pass

'''
this interface is for only for the binary classifiers that can output the certainty(possibility) of the classification 
result, which is necessary for the OneVsAllMultiClassClassifier.
'''
class ClassifierOutputPossibility(BinaryClassifier):
    '''
    the input should be numpy array with shape (n, m)
    the output should be numpy array with shape (n, 1) indicating the possibility that the data is the target
    '''
    @abstractmethod
    def predictPossibility(self, x):
        pass