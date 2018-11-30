import matplotlib.pyplot as pl
import numpy as np


from anomalyIdsEucDist import predictAnomalies
from sklearn.model_selection import KFold


def scatter_anom_scores(anom_scores):
    x = np.array(range(anom_scores.shape[0]))
    pl.figure()
    pl.scatter(x, anom_scores)
    pl.show()


features = np.genfromtxt('trainData.csv', delimiter=',', dtype=None,
                         encoding=None)

kf = KFold(n_splits=10, shuffle=True)

composite_scores = np.array([])
for train_index, test_index in kf.split(features):
    train_feat = features[train_index]
    test_feat = features[test_index]

    anom_scores = predictAnomalies(train_feat, test_feat)
    composite_scores = np.array(list(composite_scores) + list(anom_scores))

scatter_anom_scores(composite_scores)
