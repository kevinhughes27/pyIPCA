""" Incremental Principal Component Analysis
"""

# Author: Kevin Hughes <kevinhughes27@gmail.com>
# License: BSD Style.

import numpy as np
from scipy import linalg as la

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import array2d, as_float_array
from sklearn.utils.extmath import safe_sparse_dot

from sklearn.decomposition import PCA

class Hall_IPCA(BaseEstimator, TransformerMixin):
    """Incremental principal component analysis
    
    Linear dimensionality reduction using an online incremental PCA algorithm.
    Components are updated sequentially as new observations are introduced. 
    Each new observation (u) is projected on the eigenspace spanned by
    the current components and the residual vector is used as a new 
    component. The new principal components are then rotated by a rotation 
    matrix (R) whose columns are the eigenvectors of the transformed covariance 
    matrix to yield p + 1 principal components. From those, only the first p are 
    selected.
     
    Parameters
    ----------
    n_components : int
        Number of components to keep.
        Must be set

    copy : bool
        If False, data passed to fit are overwritten

    Attributes
    ----------
    `components_` : array, [n_components, n_features]
        Components with maximum variance.

    `explained_variance_ratio_` : array, [n_components]
        Percentage of variance explained by each of the selected components. \
        k is not set then all components are stored and the sum of explained \
        variances is equal to 1.0
        
    Notes
    -----
    Calling fit(X) multiple times will update the components_ etc.
    
    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.decomposition import IncrementalPCA
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> ipca = IncrementalPCA(n_components=2)
    >>> ipca.fit(X)
    IncrementalPCA(copy=True, n_components=2)
    >>> print(ipca.explained_variance_ratio_) # doctest: +ELLIPSIS
    [ 0.99244...  0.00755...]

    See also
    --------
    ProbabilisticPCA
    RandomizedPCA
    KernelPCA
    SparsePCA
    CCIPCA
    """
    def __init__(self, n_components=2, copy=True):
        self.n_components = n_components
        self.copy = copy
        self.iteration = 0
     
    def fit(self, X, y=None, **params):
        """Fit the model with X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
            
        Notes
        -----
        Calling multiple times will update the components
        """
        
        X = array2d(X)
        n_samples, n_features = X.shape 
        X = as_float_array(X, copy=self.copy)
          
        if self.iteration != 0 and n_features != self.components_.shape[1]:
            raise ValueError('The dimensionality of the new data and the existing components_ does not match')   
        
        # incrementally fit the model
        for i in range(0,X.shape[0]):
            self.partial_fit(X[i,:])
        
        return self
      
    def partial_fit(self, u):
        """ Updates the mean and components to account for a new vector.

        Based on P.Hall, D. Marshall and R. Martin "Incremental Eigenalysis for 
        Classification" which appeared in British Machine Vision Conference, volume 1,
        pages 286-295, September 1998.
        
        implementation based on the python recipe (recipe-577213-1) 
        published by Micha Kalfon

        NOTE: Does not scale well with high dimensional data

        Parameters
        ----------
        u : array [1, n_features]
            a single new data sample
        """   

        # init
        if self.iteration == 0:  
            n_features = len(u)
            self.mean_ = np.zeros([n_features], np.float)
            self.covariance_ = np.zeros([n_features, n_features], np.float)
            self.components_ = np.zeros([self.n_components,n_features], np.float)
            self.explained_variance_ratio_ = np.zeros([self.n_components], np.float)

        n = float(self.iteration)
        C = self.covariance_
        V = self.components_
        E = self.explained_variance_ratio_

        # Update covariance matrix and mean vector and centralize input around
        # new mean
        oldmean = self.mean_.copy()
        self.mean_ = (n*self.mean_ + u) / (n + 1.0)
        C = (n*C + np.dot(np.asmatrix(u).T,np.asmatrix(u)) + n*np.dot(np.asmatrix(oldmean).T,np.asmatrix(oldmean)) - (n+1.0)*np.dot(np.asmatrix(self.mean_).T,np.asmatrix(self.mean_))) / (n + 1.0)
        u -= self.mean_    

        # Project new input on current subspace and calculate
        # the normalized residual vector
        g = np.dot(u,V.T)       
        r = u - (np.dot(g,V))
        if np.linalg.norm(r) > 1e-9:
            r = (r / np.linalg.norm(r))
        else:
            r = np.zeros_like(r)    

        # Extend the transformation matrix with the residual vector and find
        # the rotation matrix by solving the eigenproblem DR=RE
        V = np.vstack([V, r])
        D = np.dot(V,np.dot(C,V.T))
        (E, R) = np.linalg.eigh(D.T)

        # Sort eigenvalues and eigenvectors from largest to smallest to get the
        # rotation matrix R
        sorter = list(reversed(E.argsort(0)))
        E = E[sorter]
        R = R[:,sorter].T

        # Apply the rotation matrix
        V = np.dot(R,V) 

        # Select only self.n_components largest eigenvectors and values 
        # and update state
        self.iteration += 1
        self.covariance_ = C
        self.components_ = V[0:self.n_components,:]
        self.explained_variance_ratio_ = E[0:self.n_components]

        # normalize explained_variance_ratio_
        self.explained_variance_ratio_ = (self.explained_variance_ratio_ / self.explained_variance_ratio_.sum())

        return

    def transform(self, X):
        """Apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        X = array2d(X)
        X_transformed = X - self.mean_
        X_transformed = np.dot(X_transformed, self.components_.T)
        return np.asarray(X_transformed)

    def inverse_transform(self, X):
        """Transform data back to its original space, i.e.,
        return an input X_original whose transform would be X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples in the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)
        """
        return np.dot(X, self.components_) + self.mean_
