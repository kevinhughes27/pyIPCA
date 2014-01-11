#!/usr/bin/env python

"""eigenface
finds the first k eigenfaces using pca and inc pca
the results are displayed side by side for visual
comparison
"""
import cv2
import numpy as np
from sklearn.decomposition import PCA
from pyIPCA import CCIPCA
from os.path import join, exists
import urllib
import zipfile

ORL_FACES_URL = "http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip"

# get the data
if not exists('orl_faces'):
  print 'downloading orl faces from %s' % ORL_FACES_URL
  urllib.urlretrieve(ORL_FACES_URL, 'orl_faces.zip')
  print 'extracing orl_faces.zip'
  zfile = zipfile.ZipFile("orl_faces.zip",)
  zfile.extractall('orl_faces')

pathToORL = 'orl_faces/'

faces = []
for i in range(1,10+1):
  for j in range(1,2+1):
      path =  pathToORL + 's' + str(i) + '/' + str(j) + '.pgm'
      #print path
      face = cv2.imread(path,0)
      faces.append(face)
      
      #cv2.imshow("face",face)
      #cv2.waitKey()
  
imgW = faces[0].shape[1]
imgH = faces[0].shape[0]

faces_rowvecs = []
for (i,face) in enumerate (faces):
  faces_rowvecs.append(face.flatten("C").copy())
          
faceData = np.vstack(faces_rowvecs)

# PCA
k = 4
pca = PCA(n_components=k)
pca.fit(faceData)

# display mean
mean = pca.mean_
mean = mean.reshape(imgH,imgW)
mean = cv2.normalize(mean,norm_type=cv2.NORM_MINMAX)

cv2.imshow("pca mean",mean)
cv2.moveWindow("pca mean",(1)*(imgW+10), 100);

# display eigenvectors
for i in range(k):

  eigenvector = pca.components_[i,:]
  eigenvector = eigenvector.reshape(imgH,imgW)
  eigenvector = cv2.normalize(eigenvector,norm_type=cv2.NORM_MINMAX)
  
  cv2.imshow("pca"+str(i),eigenvector)
  cv2.moveWindow("pca"+str(i),(i+3)*(imgW+10), 100);


# CCIPCA
k = 4 
ccipca = CCIPCA(n_components=k).fit(faceData)
Xm = ccipca.mean_   
V = ccipca.components_

# display mean
mean = Xm
mean = mean.reshape(imgH,imgW)
mean = cv2.normalize(mean,norm_type=cv2.NORM_MINMAX)

cv2.imshow("ccipca mean",mean)
cv2.moveWindow("ccipca mean",(1)*(imgW+10), 300);

# display eigenvectors
for i in range(k):

  eigenvector = V[i,:]
  eigenvector = eigenvector.reshape(imgH,imgW)
  eigenvector = cv2.normalize(eigenvector,norm_type=cv2.NORM_MINMAX)
  
  cv2.imshow("ccipca"+str(i),eigenvector)
  cv2.moveWindow("ccipca"+str(i),(i+3)*(imgW+10), 300);

cv2.waitKey()