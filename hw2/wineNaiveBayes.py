import csv
import matplotlib.pyplot as pl
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB


SHOW_BERNOULLINB_ACC_PLOT = False


def wine_parse():
    """"""
    input_ = []
    targets = []
    with open("wine.data.csv") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            input_.append([float(field) for field in row[1:]])
            targets.append(int(row[0]))
    return (input_, targets)


def standardize_columns(input_):
    input_ = np.array(input_)
    std_input = np.empty(input_.shape)
    for i in range(input_.shape[1]):
        std_input[:, i] = ((input_[:, i] - np.mean(input_[:, i]))
                           / np.std(input_[:, i]))
    return std_input


def fold2_cross_validate(input_data, targets, classifier):
    input_data = np.array(input_data)
    targets = np.array(targets)
    classifier = classifier

    # Sort Data
    train_data = input_data[0::2]
    test_data = input_data[1::2]
    train_targets = targets[0::2]
    test_targets = targets[1::2]

    # Classifier Accuracy
    classifier.fit(train_data, train_targets)
    acc = accuracy_score(classifier.predict(test_data), test_targets)

    return acc


def max_bernnb_acc(std_examples, raw_targets):
    acc_pts = np.array([[x * 0.1,
                         fold2_cross_validate(
                             std_examples,
                             raw_targets,
                             BernoulliNB(binarize=(x * 0.1)))]
                        for x in range(-15, 16)])
    pl.figure()
    pl.scatter(acc_pts[:, 0], acc_pts[:, 1])
    pl.xlabel('Binarize Threshold')
    pl.ylabel('Accuracy')
    if SHOW_BERNOULLINB_ACC_PLOT:
        pl.show()

    max_index = np.argmax(acc_pts[:, 1])
    return tuple(acc_pts[max_index])


def main():
    raw_examples, raw_targets = wine_parse()
    std_examples = standardize_columns(raw_examples)

    # GaussianNB Test
    gauss_acc = fold2_cross_validate(std_examples, raw_targets, GaussianNB())
    print("Gaussian Accuracy (2-fold cross): {}".format(gauss_acc))

    # BernoulliNB Test
    bern_acc = fold2_cross_validate(std_examples,
                                    raw_targets,
                                    BernoulliNB(binarize=0.0))
    print("Bernoulli Accuracy (2-fold cross, binarize={}): {}".
          format(0.0, bern_acc))

    max_bernnb_acc(std_examples, raw_targets)
    print(
        "Max Bernoulli Accuracy (2-fold cross): {1} at binarize threshold {0}".
        format(*max_bernnb_acc(std_examples, raw_targets)))


if __name__ == "__main__":
    main()
