import pytest
import numpy as np

def countOnesLoop(array):
    if isinstance(array, np.ndarray):
        count = 0
        for i in array.flatten():
            if i == 1:
                count += 1
        return count
    else:
        raise TypeError('`array` is not a numpy array.')


def countOnesWhere(array):
    if isinstance(array, np.ndarray):
        return len(np.where(array == 1)[0])
    else:
        raise TypeError('`array` is not a numpy array.')

# Testing
test_cases = [
    (np.ones((3, 3)), 9),
    (np.eye(3), 3),
    (np.arange(4), 1),
    (np.zeros((2,3)), 0)
]

@pytest.mark.parametrize("array, exp_result", test_cases)
def test_countOnesLoop(array, exp_result):
    assert countOnesLoop(array) == exp_result

@pytest.mark.parametrize("array, exp_result", test_cases)
def test_countOnesWhere(array, exp_result):
    assert countOnesWhere(array) == exp_result
