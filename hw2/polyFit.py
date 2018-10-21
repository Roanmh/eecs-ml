import csv
import numpy as np
import matplotlib.pyplot as pl


from sklearn.linear_model import LinearRegression, Ridge


def parse_train_poly():
    """"""
    input_ = []
    targets = []
    with open('trainPoly.csv') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            input_.append([float(field) for field in row[0:1]])
            targets.append([float(field) for field in row[-1:]])
    return (input_, targets)


def parse_test_poly():
    """"""
    input_ = []
    targets = []
    with open('testPoly.csv') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            input_.append([float(field) for field in row[0:1]])
            targets.append([float(field) for field in row[-1:]])
    return (input_, targets)


def polyize_input(input_, deg):
    polyized_input = np.repeat(np.empty(np.array(input_).shape), deg, axis=1)

    for i in range(polyized_input.shape[1]):
        polyized_input[:, i] = polyized_input[:, i] ** (i + 1)

    return polyized_input


def sum_squared_err(predictions, targets):
    sum = 0
    for prediction, target in zip(predictions, targets):
        sum += (target - prediction) ** 2
    return sum


def ridge_err(predictions, targets, param):
    return sum_squared_err(predictions, targets) + (10**-6 * (np.linalg.norm(param) ** 2))


train_data, train_targets = parse_train_poly()
test_data, test_targets = parse_test_poly()

deg_axis = range(1, 10)
# A: Sum of Squared Errors
train_sse = []
test_sse = []
mean_squared_norm = []
train_ridge_err = []
test_ridge_err = []
for deg in deg_axis:
    model = LinearRegression().fit(polyize_input(train_data, deg), train_targets)
    train_sse.append(sum_squared_err(model.predict(polyize_input(train_data, deg)), train_targets))
    test_sse.append(sum_squared_err(model.predict(polyize_input(test_data, deg)), test_targets))

    mean_squared_norm.append(np.linalg.norm(model.coef_) ** 2 / deg)

    model = Ridge(alpha=10**-6).fit(polyize_input(train_data, deg), train_targets)
    train_ridge_err.append(ridge_err(model.predict(polyize_input(train_data, deg)), train_targets, model.coef_))
    test_ridge_err.append(ridge_err(model.predict(polyize_input(test_data, deg)), test_targets, model.coef_))

train_sse = np.ravel(train_sse)
test_sse = np.ravel(test_sse)
train_ridge_err = np.ravel(train_ridge_err)
test_ridge_err = np.ravel(test_ridge_err)

pl.scatter(deg_axis, train_sse, label="Train Data")
pl.scatter(deg_axis, test_sse, label="Test Data")
pl.xlabel("Max Degree")
pl.ylabel("Sum of Squared Errors")
pl.legend(loc="best")
pl.show()

pl.figure()
pl.scatter(deg_axis, mean_squared_norm, label="Mean Squared Norm")
pl.yscale("log")
pl.xlabel("Max Degree")
pl.ylabel("Mean Squared Norm")
pl.legend(loc="best")
pl.show()

pl.figure()
pl.scatter(deg_axis, train_ridge_err, label="Train Data")
pl.scatter(deg_axis, test_ridge_err, label="Test Data")
pl.xlabel("Max Degree")
pl.ylabel("Ridge Error")
pl.legend(loc="best")
pl.show()
