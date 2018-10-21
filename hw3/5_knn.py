import random

import numpy as np


training_input = np.array([[-1, -1],
                           [-1, 0],
                           [0, 0],
                           [0, 1],
                           [1, 0],
                           [1, -1]])

test_input = [0.3, 0.2]


def noise_term():
    values = [-1, 0, 0, 1]
    return random.choice(values)


def f(x1, x2):
    return (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2


def t(x1, x2):
    return f(x1, x2) + noise_term()


def knn(training_input, training_targets, x1, x2, k=1):
    dist_and_target = []
    for i in range(len(training_input)):
        dist_and_target.append(((training_input[i,0] - x1) ** 2
                                + (training_input[i,1]) ** 2,
                                training_targets[i]))
    dist_and_target.sort(key=lambda x: x[0])

    return np.sum(np.array(dist_and_target)[0:k,1]) / k


def mean_sq_err(outputs, targets):
    return np.sum((outputs - targets) ** 2) / outputs.shape[0]


def force_of_brute():
    num_trials = 10000
    outputs = np.array(num_trials)
    targets = np.array(num_trials)
    for i in range(num_trials):
        import pdb; pdb.set_trace()
        training_targets = [t(j[0], j[1]) for j in training_input]
        outputs[i] = knn(training_input, training_targets, 0.3, 0.2)
        targets[i] = t(test_input[0], test_input[1])

    return mean_sq_err(outputs, targets)


print(force_of_brute())


