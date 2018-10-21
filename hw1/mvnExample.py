#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl

SAMPLE_CNT = 1000

def multivariate_sample(mean_vc, covarience_matr):
    return np.random.multivariate_normal(mean_vc, covarience_matr, SAMPLE_CNT)


def bi_normal_sample(mean_vc, covarience_matr):
    if not isinstance(mean_vc, np.ndarray) or not isinstance(covarience_matr, np.ndarray):
        raise Exception("Only Numpy arrays are supported as arguments.")
    if (mean_vc.shape == (2,)) or (covarience_matr.shape == (2, 2)):
        pass
        # raise Exception("This function only supports 2 dimentional cases.")
    if covarience_matr[0, 1] != covarience_matr[1, 0]:
        raise Exception( "Covariance matrix is not valid: covarience_matr[0][1] != covarience_matr[1][0]")

    x1_mean = mean_vc[0]
    x2_mean = mean_vc[1]

    x1_stdev = covarience_matr[0, 0]
    x2_stdev = np.linalg.det(covarience_matr)

    # Consistency of covarient values established above
    x12_covar = covarience_matr[0, 1]

    result = np.empty((SAMPLE_CNT, 2))
    for n in range(SAMPLE_CNT):
        result[n, 0] = np.random.normal(x1_mean, x1_stdev)
        result[n, 1] = np.random.normal((result[n, 0] - x2_mean) * x12_covar, x2_stdev)

    return result

def graph_distr_2d(distr):
    pl.figure()
    pl.scatter(distr[:, 0], distr[:, 1])
    pl.xlabel('x1')
    pl.ylabel('x2')

def main():
    mean_vc = np.array([0, 0])
    covarience_matr = np.array([[1, 0.5], [0.5, 1]])

    sample_lib = multivariate_sample(mean_vc, covarience_matr)
    sample_roanmh = bi_normal_sample(mean_vc, covarience_matr)
    graph_distr_2d(sample_lib)
    graph_distr_2d(sample_roanmh)

    print("mean from lib: {}".format(np.mean(sample_lib, 0)))
    print("mean from student: {}".format(np.mean(sample_roanmh, 0)))
    print("Covariance Matrix from lib: {}".format(np.cov(np.transpose(sample_lib))))
    print("Covariance Matrix from student: {}".format(np.cov(np.transpose(sample_roanmh))))

    pl.show()


if __name__ == '__main__':
    main()
