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

class Skocaj_IPCA(BaseEstimator, TransformerMixin):
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

    def partial_fit(self, x):
        """ Updates the mean and components to account for a new vector.

        An Implementation of the incremental pca technique (algorithm 1)
        described in:

        Skocaj, Danijel, and Ales Leonardis. "Weighted and robust incremental method
        for subspace learning." Computer Vision, 2003. Proceedings. Ninth IEEE
        International Conference on. IEEE, 2003.

        Parameters
        ----------
        x : array [1, n_features]
            a single new data sample
        """  

        # init
        if self.iteration == 0:  
            n_features = len(x)
            self.mean_ = x
            self.coeffs_ = np.zeros([1,1], np.float)
            self.components_ = np.zeros([1,n_features], np.float)
            self.explained_variance_ratio_ = np.zeros([1], np.float)

        xm = self.mean_
        U = self.components_
        D = self.explained_variance_ratio_
        A = self.coeffs_

        # 1: Project a new image x into the current eigenspacea = np.dot((u-xm),U.T)
        a = np.dot((x-xm),U.T)

        # 2: Reconstruct the new image
        y = np.dot(a,U) + xm

        # 3: Compute the residual vector
        r = x - y
        normr = np.linalg.norm(r) 
        if normr > 1e-9:
            r = (r / normr)
        else:
            r = np.zeros_like(r)

        # 4: Append r as a new basis vector
        U = np.vstack([U,r])

        # 5: Determine the coefficients in the new basis
        A = np.vstack([A,a])
        A = np.hstack([A,np.zeros([A.shape[0],1])])
        A[-1,-1] = normr

        # 6: Perform PCA on A. Obtain the mean value xm, the
        # eigenvectors U, and the eigenvalues D
        pca = PCA()
        pca.fit(A)

        # 7: Project the coefficient vectors to the new basis
        A = pca.transform(A)

        # 9: Update the mean
        xm = xm + np.dot(pca.mean_,U)

        # 8: Rotate the supspace
        U = np.dot(pca.components_,U)

        # 10: New eigenvalues
        D = pca.explained_variance_ratio_

        # Select only self.n_components largest eigenvectors and values 
        # and update state
        self.iteration += 1
        self.mean = xm
        self.components_ = U[0:self.n_components,:]
        self.coeffs_ = A[:,0:self.n_components]
        self.explained_variance_ratio_ = D[0:self.n_components]

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
