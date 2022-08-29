"""
This module is used to calculate the PCA transform and the joint distance.
"""



# Imports
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np



def pca_transform(data_scaled):
    """
    Compute the PCA transform from a scaled data.

    Parameters
    ----------
    data_scaled : numpy.ndarray
        A matrix with the scaled data to transform

    Returns
    -------
    pandas.core.frame.DataFrame
        a dataframe with the transformed data labelled
    """

    #compute the PCA transform
    pca = PCA(n_components=2).fit_transform(data_scaled)

    #return a dataframe with the transformed data
    return pd.DataFrame(data=pca, columns=['PC1', 'PC2'])



def dra_distance(real, synthetic):
    """
    Compute the proposed DRA distance, which is a distance metric that indicates the distance
    between two dimensionality dimension plots. The metric is the joint distance between the
    baricenters distance and spread distance.

    Parameters
    ----------
    real : pandas.core.frame.DataFrame
        A dataframe with the dimensionality results (PCA or ISOMAP) of the real data.

    synthetic: pandas.core.frame.DataFrame
        A dataframe with the dimensionality results (PCA or ISOMAP) of the synthetic data.

    Returns
    -------
    numpy.float64
        the computed DRA distance metric.
    """

    #compute baricenters distance
    bc_real = np.mean(real[['PC1', 'PC2']].values)
    bc_synth = np.mean(synthetic[['PC1', 'PC2']].values)
    dist_real_synth = np.linalg.norm(bc_real - bc_synth)

    #compute spread distance
    spread_real = np.std(real[['PC1', 'PC2']].values)
    spread_synth = np.std(synthetic[['PC1', 'PC2']].values)
    dist_spread_real_synth = np.abs(spread_real-spread_synth)

    #compute joint distance
    alpha = 0.05
    return np.round(alpha*dist_real_synth + (1-alpha)*dist_spread_real_synth, 4)
