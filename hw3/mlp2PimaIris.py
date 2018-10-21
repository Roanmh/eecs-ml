import csv
import random

import matplotlib.pyplot as pl
import numpy as np


class mlp:
    """ A Multi-Layer Perceptron"""

    def __init__(
            self,
            inputs,
            targets,
            nhidden,
            beta=1,
            momentum=0.9,
            outtype='logistic'):
        """ Constructor """
        # Set up network size
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype

        # Initialise network
        self.weights1 = ((np.random.rand(self.nin + 1,
                                         self.nhidden) - 0.5) * 2
                         / np.sqrt(self.nin))
        self.weights2 = ((np.random.rand(self.nhidden + 1,
                                         self.nout) - 0.5) * 2
                         / np.sqrt(self.nhidden))

    def fwd_accuracy(self, input_, targets):
        cm = self.confmatr(input_, targets)
        return np.trace(cm) / np.sum(cm) * 100

    def error_fxn(self, output, targets):
        return 0.5 * np.sum((targets - output)**2)

    def fwd_error(self, input_, targets):
        input_ = np.concatenate(
            (input_, -np.ones((np.shape(input_)[0], 1))), axis=1)
        output = self.mlpfwd(input_)
        return self.error_fxn(output, targets)

    def earlystopping(
            self,
            inputs,
            targets,
            valid,
            validtargets,
            eta,
            niterations=100):

        valid = np.concatenate(
            (valid, -np.ones((np.shape(valid)[0], 1))), axis=1)

        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000

        count = 0
        while (((old_val_error1 - new_val_error) > 0.001)
               or ((old_val_error2 - old_val_error1) > 0.001)):
            count += 1
            print(count)
            self.mlptrain(inputs, targets, eta, niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = self.error_fxn(validtargets, validout)

        print("Stopped", new_val_error, old_val_error1, old_val_error2)
        return new_val_error

    def mlptrain(self, inputs, targets, eta, niterations):
        """ Train the thing """
        inputsNoBias = np.copy(inputs)
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs, -np.ones((self.ndata, 1))), axis=1)
        change = list(range(self.ndata))

        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))

        for n in range(niterations):

            self.outputs = self.mlpfwd(inputs)

            error = self.error_fxn(self.outputs, targets)
            # if (np.mod(n, 10) == 0):
                # print("Iteration: ", n, " Error: ", error)
                # Print confusion matrix if doing classification
                # if self.outtype != 'linear':
                    # self.confmat(inputsNoBias, targets)

            # Different types of output neurons
            if self.outtype == 'linear':
                deltao = (self.outputs - targets) / self.ndata
            elif self.outtype == 'logistic':
                deltao = self.beta * (self.outputs - targets) * \
                    self.outputs * (1.0 - self.outputs) / self.ndata
            elif self.outtype == 'softmax':
                deltao = ((self.outputs - targets)
                          * (self.outputs *
                             (-self.outputs) + self.outputs) / self.ndata)
            else:
                print("error")

            deltah = (self.hidden * self.beta *
                      (1.0 - self.hidden)
                      * (np.dot(deltao, np.transpose(self.weights2))))

            updatew1 = (eta * (np.dot(np.transpose(inputs),
                                      deltah[:, :-1]))
                        + self.momentum * updatew1)
            updatew2 = eta * (np.dot(np.transpose(self.hidden),
                                     deltao)) + self.momentum * updatew2
            self.weights1 -= updatew1
            self.weights2 -= updatew2

            # Randomise order of inputs (not necessary for matrix-based
            # calculation)
            # np.random.shuffle(change)
            # inputs = inputs[change,:]
            # targets = targets[change,:]

    def mlpfwd(self, inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs, self.weights1)
        self.hidden = 1.0 / (1.0 + np.exp(-self.beta * self.hidden))
        self.hidden = np.concatenate(
            (self.hidden, -np.ones((np.shape(inputs)[0], 1))), axis=1)

        outputs = np.dot(self.hidden, self.weights2)

        # Different types of output neurons
        if self.outtype == 'linear':
            return outputs
        elif self.outtype == 'logistic':
            return 1.0 / (1.0 + np.exp(-self.beta * outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs), axis=1) * \
                np.ones((1, np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs)) / normalisers)
        else:
            print("error")

    def confmat(self, inputs, targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate(
            (inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)
        outputs = self.mlpfwd(inputs)

        nclasses = np.shape(targets)[1]

        if nclasses == 1:
            nclasses = 2
            outputs = np.where(outputs > 0.5, 1, 0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0)
                                  * np.where(targets == j, 1, 0))

        print("Confusion matrix is:")
        print(cm)
        print("Percentage Correct: ", np.trace(cm) / np.sum(cm) * 100)

        return cm


class mlp2:
    """ A Multi-Layer Perceptron"""

    def __init__(
            self,
            inputs,
            targets,
            nhidden,
            beta=1,
            momentum=0.9,
            outtype='logistic'):
        """ Constructor """
        # Set up network size
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype

        # Initialise network
        self.weights1 = ((np.random.rand(self.nin + 1,
                                         self.nhidden) - 0.5) * 2
                         / np.sqrt(self.nin))
        self.weights2 = ((np.random.rand(self.nhidden + 1,
                                         self.nout) - 0.5) * 2
                         / np.sqrt(self.nhidden))

    def fwd_accuracy(self, input_, targets):
        cm = self.confmatr(input_, targets)
        return np.trace(cm) / np.sum(cm) * 100

    def error_fxn(self, output, targets):
        return -np.sum(np.dot(targets,
                              np.transpose(np.log(output)))
                       + np.dot(np.ones_like(targets) - targets,
                                np.transpose(np.log(np.ones_like(output) - output))))

    def fwd_error(self, input_, targets):
        input_ = np.concatenate(
            (input_, -np.ones((np.shape(input_)[0], 1))), axis=1)
        output = self.mlpfwd(input_)
        return self.error_fxn(output, targets)

    def earlystopping(
            self,
            inputs,
            targets,
            valid,
            validtargets,
            eta,
            niterations=100):

        valid = np.concatenate(
            (valid, -np.ones((np.shape(valid)[0], 1))), axis=1)

        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000

        count = 0
        while (((old_val_error1 - new_val_error) > 0.001)
               or ((old_val_error2 - old_val_error1) > 0.001)):
            count += 1
            print(count)
            self.mlptrain(inputs, targets, eta, niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = self.error_fxn(validtargets, validout)

        print("Stopped", new_val_error, old_val_error1, old_val_error2)
        return new_val_error

    def mlptrain(self, inputs, targets, eta, niterations):
        """ Train the thing """
        inputsNoBias = np.copy(inputs)
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs, -np.ones((self.ndata, 1))), axis=1)
        change = list(range(self.ndata))

        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))

        for n in range(niterations):

            self.outputs = self.mlpfwd(inputs)

            error = self.error_fxn(self.outputs, targets)
            # if (np.mod(n, 10) == 0):
                # print("Iteration: ", n, " Error: ", error)
                # Print confusion matrix if doing classification
                # if self.outtype != 'linear':
                    # self.confmat(inputsNoBias, targets)

            # Different types of output neurons
            if self.outtype == 'linear':
                deltao = (self.outputs - targets) / self.ndata
            elif self.outtype == 'logistic':
                deltao = self.beta * (self.outputs - targets) * \
                    self.outputs * (1.0 - self.outputs) / self.ndata
            elif self.outtype == 'softmax':
                deltao = ((self.outputs - targets)
                          * (self.outputs *
                             (-self.outputs) + self.outputs) / self.ndata)
            else:
                print("error")

            deltah = (self.hidden * self.beta *
                      (1.0 - self.hidden)
                      * (np.dot(deltao, np.transpose(self.weights2))))

            updatew1 = (eta * (np.dot(np.transpose(inputs),
                                      deltah[:, :-1]))
                        + self.momentum * updatew1)
            updatew2 = eta * (np.dot(np.transpose(self.hidden),
                                     deltao)) + self.momentum * updatew2
            self.weights1 -= updatew1
            self.weights2 -= updatew2

            # Randomise order of inputs (not necessary for matrix-based
            # calculation)
            # np.random.shuffle(change)
            # inputs = inputs[change,:]
            # targets = targets[change,:]

    def mlpfwd(self, inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs, self.weights1)
        self.hidden = 1.0 / (1.0 + np.exp(-self.beta * self.hidden))
        self.hidden = np.concatenate(
            (self.hidden, -np.ones((np.shape(inputs)[0], 1))), axis=1)

        outputs = np.dot(self.hidden, self.weights2)

        # Different types of output neurons
        if self.outtype == 'linear':
            return outputs
        elif self.outtype == 'logistic':
            return 1.0 / (1.0 + np.exp(-self.beta * outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs), axis=1) * \
                np.ones((1, np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs)) / normalisers)
        else:
            print("error")

    def confmat(self, inputs, targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate(
            (inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)
        outputs = self.mlpfwd(inputs)

        nclasses = np.shape(targets)[1]

        if nclasses == 1:
            nclasses = 2
            outputs = np.where(outputs > 0.5, 1, 0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0)
                                  * np.where(targets == j, 1, 0))

        print("Confusion matrix is:")
        print(cm)
        print("Percentage Correct: ", np.trace(cm) / np.sum(cm) * 100)

        return cm


def parse_pima():
    """Parse Pima csv and provide an input and target array of results"""
    input_ = []
    targets = []
    with open('pima-indians-diabetes.csv') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            input_.append([float(field) for field in row[:-1]])
            targets.append([int(field) for field in row[-1:]])
    return (input_, targets)


def shuffle_data(input_, targets):
    """Shuffle and split given data into 50:25:25 training, test, and valid."""

    indicies = list(range(len(input_)))
    random.shuffle(indicies)

    shf_input = []
    shf_targs = []

    for index in indicies:
        shf_input.append(input_[index])
        shf_targs.append(targets[index])

    return (shf_input, shf_targs)


def split_data(input_, targets):
    training_input = [x for i, x in enumerate(input_) if i % 2]
    training_targs = [x for i, x in enumerate(targets) if i % 2]
    test_input = [x for i, x in enumerate(input_) if (not i % 2) and (i % 4)]
    test_targs = [x for i, x in enumerate(targets) if (not i % 2) and (i % 4)]
    validation_input = [x for i, x in enumerate(input_) if not (i % 2) and not (i % 4)]
    validation_targs = [x for i, x in enumerate(targets) if not (i % 2) and not (i % 4)]

    return (training_input, training_targs,
            test_input, test_targs,
            validation_input, validation_targs)


def error_over_iterations(training_input,
                          training_targs,
                          test_input,
                          test_targs,
                          validation_input,
                          validation_targs,
                          max_iter=50):
    iterations = list(range(max_iter))
    mlp_train_err = []
    mlp_valid_err = []
    mlp2_train_err = []
    mlp2_valid_err = []
    for i in range(max_iter):
        mlp_test = mlp(training_input, training_targs, 2)
        mlp2_test = mlp2(training_input, training_targs, 2)

        mlp_test.mlptrain(training_input, training_targs, 0.25, i)
        mlp2_test.mlptrain(training_input, training_targs, 0.25, i)

        mlp_train_err.append(mlp_test.fwd_error(training_input, training_targs))
        mlp_valid_err.append(mlp_test.fwd_error(validation_input, validation_targs))
        mlp2_train_err.append(mlp2_test.fwd_error(training_input, training_targs))
        mlp2_valid_err.append(mlp2_test.fwd_error(validation_input, validation_targs))

    pl.figure()
    pl.plot(iterations, mlp_train_err, 'r.')
    pl.plot(iterations, mlp_valid_err, 'r-')
    pl.figure()
    pl.plot(iterations, mlp2_train_err, 'b.')
    pl.plot(iterations, mlp2_valid_err, 'b-')
    pl.show()

    mlp_test = mlp(training_input, training_targs, 2)
    mlp2_test = mlp2(training_input, training_targs, 2)

    mlp_test.earlystopping(training_input, training_targs, validation_input, validation_targs, 0.25, max_iter)
    mlp2_test.earlystopping(training_input, training_targs, validation_input, validation_targs, 0.25, max_iter)

    print("SSE-trained Accuracy: {}".format(mlp.fwd_accuracy(validation_input, validation_targs)))
    print("Cross-Entropy-trained Accuracy: {}".format(mlp2.fwd_accuracy(validation_input, validation_targs)))


# Ad-hoc Tests
def dbg():
    all_data = parse_pima()
    training, test, validation = split_data(all_data[0], all_data[1])
    m = mlp(training[0], training[1], 2)
    # m.mlptrain(training[0], training[1], 0.25, 5)
    print(m.fwd_error(validation[0], validation[1]))
    m.mlptrain(training[0], training[1], 0.25, 1)


# Unit Tests
def test_shuffle_data():
    input_ = range(10)
    targets = [x % 2 for x in range(10)]

    shf_input, shf_targs = shuffle_data(input_, targets)

    assert set(input_) == set(shf_input)
    assert set(targets) == set(shf_targs)
