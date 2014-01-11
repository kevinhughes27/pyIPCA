#!/usr/bin/env python

"""ellipse
inc pca performed on a set of points distributed like
an ellipse. Should return the correct principal
axis of the ellipse
"""
import math
import numpy as np
import pylab as pl  
from sklearn.decomposition import PCA
from pyIPCA import CCIPCA

# 2D point set in the shape of an ellipse
X = np.array([10*np.random.randn(1000), np.random.randn(1000)]).transpose()
rot = np.array( [[math.cos(math.radians(135)), -math.sin(math.radians(135))],
              [math.sin(math.radians(135)),  math.cos(math.radians(135))]] )
X = np.dot(X,rot)

print 'rot'
print rot

# CCIPCA
k = 2 
ccipca = CCIPCA(n_components=k).fit(X)
Xm = ccipca.mean_   
V = ccipca.components_

# Normalize V
for j in range(0,V.shape[0]):
    V[j,:] /= np.sqrt(np.dot(V[j,:],V[j,:]))

print 'V'
print V

# Plot
def plot_components(Xm, V, c):
  pl.arrow(Xm[0], Xm[1], 25*V[0,0], 25*V[0,1], 
     shape='full', lw=3,length_includes_head=True, head_width=1.25, color = c)

  pl.arrow(Xm[0], Xm[1], -5*V[1,0], -5*V[1,1], 
    shape='full', lw=3,length_includes_head=True, head_width=1.25, color = c) 

pl.figure()

# the point data set
pl.scatter( X[:,0], X[:,1] )

# plot eigenvectors
plot_components(Xm, V,'red')

pl.show()