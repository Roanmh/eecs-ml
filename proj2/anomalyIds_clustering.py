# -*- coding: utf-8 -*-
"""
Anomaly detector using distance from nearest-valid centroid.
Based on code by Kevin Xu.

@author: Kevin S. Xu
"""

import numpy as np
from sklearn.feature_extraction import DictVectorizer


def predictAnomalies(trainFeatures, testFeatures):
    nAttr = len(trainFeatures.dtype)
    nCatAttr = 3
    nRealAttr = nAttr - nCatAttr
    nTrainSamples = np.size(trainFeatures)
    nTestSamples = np.size(testFeatures)

    # Get list of names of categorical attributes
    catAttrNames = list(trainFeatures.dtype.names[-nCatAttr:])

    # Convert categorical features to binary using 1-of-K representation
    trainCat = trainFeatures[catAttrNames]
    trainCatDict = catFeatureDict(trainCat, catAttrNames)
    dv = DictVectorizer()
    trainCatEncoded = dv.fit_transform(trainCatDict).toarray()
    testCat = testFeatures[catAttrNames]
    testCatDict = catFeatureDict(testCat, catAttrNames)
    testCatEncoded = dv.transform(testCatDict).toarray()

    # Extract real features and convert all to float type
    trainReal = np.zeros((nTrainSamples, nRealAttr))
    testReal = np.zeros((nTestSamples, nRealAttr))
    for attr in range(nRealAttr):
        trainReal[:, attr] = trainFeatures['f' + str(attr)].astype(float)
        testReal[:, attr] = testFeatures['f' + str(attr)].astype(float)

    # Combine real features and encoded categorical features (now all of type
    # float)
    trainAll = np.c_[trainReal, trainCatEncoded]
    testAll = np.c_[testReal, testCatEncoded]

    # Simple anomaly detector--compute distance of each test sample from mean
    # over all training samples


    return testDistFromMean


def catFeatureDict(catArray, catAttrNames):
    """
    Convert structured array of categorical variables (represented as byte
    strings) to a list of dictionaries with values as decoded strings
    """
    catDict = []
    for row in catArray:
        dictRow = {}
        for col in range(len(row)):
            dictRow[catAttrNames[col]] = row[col]
        catDict.append(dictRow)

    return catDict


if __name__ == "__main__":
    data = np.genfromtxt('trainData.csv', delimiter=',', dtype=None,
                         encoding=None)
    trainData = data[::2]
    testData = data[1::2]

    anomScores = predictAnomalies(trainData, testData)
