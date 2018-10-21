import numpy as np

train_data = np.array([[-1, x] for x in range(1, 5)])

targets = np.array([[0],
                    [0],
                    [1],
                    [1]])

param = np.dot(np.linalg.pinv(train_data), targets)

print(param)
