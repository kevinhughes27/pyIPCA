#!/usr/bin/env python

"""iris
pca and ccipca on the scikit-learn iris dataset

this code is modified from the scikit-learn example:
scikit-learn/examples/decomposition/plot_pca_vs_lda.py
"""
import cv2
from sklearn import datasets
from sklearn.decomposition import PCA
from pyIPCA import CCIPCA
import pylab as pl

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# PCA
k = 2
pca = PCA(n_components=k)
X_r = pca.fit(X).transform(X)

# CCIPCA
k = 2
ccipca = CCIPCA(n_components=k)   
X_r2 = ccipca.fit(X).transform(X)

# Plot
pl.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
  pl.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
pl.legend()
pl.title('PCA of IRIS dataset')

pl.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
  pl.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
pl.legend()
pl.title('CCIPCA of IRIS dataset')

pl.show()