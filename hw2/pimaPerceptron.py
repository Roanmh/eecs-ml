import csv
import numpy as np
import pytest


from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.metrics import accuracy_score


def parse_pima():
    """"""
    input_ = []
    targets = []
    with open('pima-indians-diabetes.csv') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            input_.append([float(field) for field in row[:-1]])
            targets.append([int(field) for field in row[-1:]])
    return (input_, targets)


class SLPerceptron:
    """Single-Layer Perceptron Network"""
    def __init__(self, train_data, targets, max_iter=100, eta=0.1,
                 concat_bias=True):
        """Ititialize and train the network."""
        self.concat_bias = concat_bias
        self.train_data = np.array(train_data)
        self.targets = np.array(targets)
        self.max_iter = max_iter
        self.eta = eta

        # Initial Weights
        self.weights_0 = np.random.rand(
            self.train_data.shape[1] + 1,
            self.targets.shape[1]) * 0.1 - 0.05  # XXX: Validate expression
        self.weights = self.weights_0

        # Training
        for n in range(self.max_iter):
            y = self.classify(self.train_data)
            if np.array_equal(y, self.targets):
                self.batch_count = n
                return
            else:
                self.weights = self._batch_update(y)

        if not np.array_equal(y, self.targets):
            self.batch_count = n
            raise Exception("Training exceeded allowed iterations.")

    def classify(self, input_):
        """"""
        cls_input_ = np.array(input_)
        if self.concat_bias:
            cls_input_ = np.concatenate(
                (-np.ones((cls_input_.shape[0], 1)), cls_input_),
                axis=1
            )
        return np.where(np.dot(cls_input_, self.weights) > 0, 1, 0)

    def _batch_update(self, output):
        """"""
        return self.weights - self.eta * np.dot(np.transpose(self.train_data),
                                                output - self.targets)


class LRThreshold:
    """"""
    def __init__(self, train_data, targets, threshold=0.5, concat_bias=True):
        """"""
        self.concat_bias = concat_bias
        self.train_data = np.array(train_data)
        if self.concat_bias:
            self.train_data = np.concatenate(
                (-np.ones((self.train_data.shape[0], 1)), self.train_data),
                axis=1
            )
        self.targets = np.array(targets)
        self.threshold = threshold

        # Linear Regression
        self.parameters = np.dot(np.linalg.pinv(self.train_data), self.targets)

    def classify(self, input_):
        """"""
        cls_input_ = np.array(input_)
        if self.concat_bias:
            cls_input_ = np.concatenate(
                (-np.ones((cls_input_.shape[0], 1)), cls_input_),
                axis=1
            )
        return np.dot(cls_input_, self.parameters) > self.threshold


def fold2_cross_valid(train_data, targets, classifier):
    train_data = np.array(train_data)
    targets = np.array(targets)

    # Sort Data
    train_data0 = train_data[0::2]
    train_data1 = train_data[1::2]
    targets0 = targets[0::2]
    targets1 = targets[1::2]

    # Classifier 0 result
    classifier.fit(train_data0, targets0)
    cls0_acc = accuracy_score(np.where(classifier.predict(train_data1) > 0.5, 1, 0), targets1)

    # Classifier 1 result
    classifier.fit(train_data1, targets1)
    cls1_acc = accuracy_score(np.where(classifier.predict(train_data0) > 0.5, 1, 0), targets0)

    return (cls0_acc, cls1_acc)


def standardize_columns(input_):
    input_ = np.array(input_)
    std_input = np.empty(input_.shape)
    for i in range(input_.shape[1]):
        std_input[:, i] = ((input_[:, i] - np.mean(input_[:, i]))
                           / np.std(input_[:, i]))
    return std_input

train_data, targets = parse_pima()

# Un-Processed
print("Unproccessed: ")
print("Perceptron: {}".format(fold2_cross_valid(train_data,
                                                targets,
                                                Perceptron(max_iter=1000, tol=1e-3))))
print("LinearRegression: {}".format(fold2_cross_valid(train_data,
                                                      targets,
                                                      LinearRegression())))

# Standardized
print("Standardized:")
print("Perceptron: {}".format(fold2_cross_valid(standardize_columns(train_data),
                                                targets,
                                                Perceptron(max_iter=1000, tol=1e-3))))
print("LinearRegression: {}".format(fold2_cross_valid(standardize_columns(train_data),
                                                      targets,
                                                      LinearRegression())))

print()




train_param = [
    ([[1], [2], [3], [4]], [[0], [0], [1], [1]]),
    ([[1], [2], [3], [4]], [[1], [1], [1], [1]]),
    ([[1], [2], [3], [4]], [[0], [1], [1], [1]]),
]


@pytest.mark.parametrize("train_data, targets", train_param)
def test_slperceptron_train(train_data, targets):
    trained_model = SLPerceptron(train_data, targets)
    assert np.array_equal(trained_model.classify(train_data),
                          trained_model.targets) is True


@pytest.mark.parametrize("train_data, targets", train_param)
def test_lrthreshold_train(train_data, targets):
    trained_model = LRThreshold(train_data, targets)
    assert np.array_equal(trained_model.classify(train_data),
                          trained_model.targets) is True
