from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def evaluate_accuracy(classifier, train_x, test_x, train_y, test_y):
    classifier.fit(train_x, train_y)
    pred_y = classifier.predict(test_x)
    return accuracy_score(test_y, pred_y)

def evaluate_classifier(cls1, cls2, test_size=0.2, n_samples=1000, n_classes=2, n_features=10, n_informative=6):
    X, Y = make_classification(n_samples=n_samples,
                               n_classes=n_classes,
                               n_features=n_features,
                               n_informative=n_informative,
                               random_state=0)
    #X = StandardScaler().fit_transform(X)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=test_size, random_state=0)
    acc1 = evaluate_accuracy(cls1, train_x, test_x, train_y, test_y)
    acc2 = evaluate_accuracy(cls2, train_x, test_x, train_y, test_y)
    print('acc1 = {}, acc2 = {}'.format(acc1, acc2))