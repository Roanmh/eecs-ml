import numpy as np

OUTPUT_NEURON_COUNT = 1
ETA = 0.1

train_data = np.array([(-1, i) for i in range(1, 5)])
"""Matrix of training exaples, per row.

The number of rows (`shape[0]`) indicates number of training examples. The
number of columens (`shape[1]`) insidcates the dimensionality of the input.

"""

weights = np.concatenate(
    (np.full((1, OUTPUT_NEURON_COUNT), -0.1),
     np.full((train_data.shape[1] - 1, OUTPUT_NEURON_COUNT), 1)),
    axis=0)
"""Matrix of perceptron input weights with one perceptron per row.

The number of rows indicates dimensionality of input. (bias?) The number of
columns indicates the number of output perceptrons.

"""

h = np.empty((train_data.shape[0], weights.shape[1]))
"""Matrix of neuron outputs for each training example by row.

The number of rows indicates the number of training examples. The number of
columns indicates the number of outputs perceptrons, and thus classes.

"""

targets = np.array([[0],
                    [0],
                    [1],
                    [1]])
"""Matrix of correct output neurons per training example, by row.

The number of rows indicates the number of training examples. The number of
columns indicates the number of output neurons.

"""

h = np.dot(train_data, weights)

y = np.where(h > 0, 1, 0)

weights_new = weights - ETA * np.dot(np.transpose(train_data), y - targets)

h_new = np.dot(train_data, weights_new)

y_new = np.where(h > 0, 1, 0)

print(weights_new)
