"""
This
"""

#import libraries
from sklearn.manifold import Isomap
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


def pca_transform(data_scaled):
    """Compute the PCA transform from a scaled data.

    Parameters
    ----------
    data_scaled : numpy.ndarray
        A matrix with the scaled data to transform
    labels : numpy.ndarray
        Labels to assign to the transformed data (0 for real and 1 for synthetic)
    Returns
    -------
    pandas.core.frame.DataFrame
        a dataframe with the transformed data labelled
    """

    #compute the PCA transform
    pca = PCA(n_components=2).fit_transform(data_scaled)

    #return a dataframe with the transformed data
    return pd.DataFrame(data=pca, columns=['PC1', 'PC2'])


def batch(iterable, num=1):
    """Create iterable batches from a dataframe.

    Parameters
    ----------
    iterable : numpy.ndarray
        A matrix to be divided in batches
    n : int
        Length of the batches to create
    """

    #get length of the matrix to divide in batches
    length = len(iterable)

    #loop to divide the data into batches of length n
    for ndx in range(0, length, num):
        yield iterable[ndx:min(ndx + num, length)]


def isomap_transform_on_batch(data_scaled):
    """Compute the Isomap transform on batch from a scaled data.

    Parameters
    ----------
    data_scaled : numpy.ndarray
        A matrix with the scaled data to transform
    labels : numpy.ndarray
        Labels to assign to the transformed data (0 for real and 1 for synthetic)
    Returns
    -------
    pandas.core.frame.DataFrame
        a dataframe with the transformed data labelled
    """

    #initialize dataframe to save the values of the transformation
    iso_df = pd.DataFrame(columns=['PC1', 'PC2'])

    #loop to iterate over all batches of data
    for bat in batch(data_scaled, 10000):

        #transform the batch of data
        iso_transform = Isomap(n_components=2).fit_transform(bat)

        #append the transformation of the actual batch to the dataframe that contains the
        # transformation of all the batches
        iso = pd.DataFrame(data=iso_transform, columns=['PC1', 'PC2'])
        iso_df = iso_df.append(iso, ignore_index=True)

    #return a dataframe with the transformed data
    return iso_df


def dra_distance(real, synthetic):
    """Compute the proposed DRA distance, which is a distance metric that indicates the distance
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
