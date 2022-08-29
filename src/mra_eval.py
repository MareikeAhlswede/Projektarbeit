"""
This module is used to compute the normalized contingency table used for the MRA analysis.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def get_categorical_correlations(df):
    """
    Computes the normalized contingency table and its norm of categorical attributes of a dataframe.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        The dataframe to compute the normalized contingency table


    Returns
    -------
    pandas.core.frame.DataFrame
        A dataframe with the normalized contingency table of the categorical attributes
        of the dataframe
    """


    # Select the categorical columns of the dataframe
    df = df.select_dtypes(include=['category'])

    # Get each combination of attributes pairs.
    factors_paired = [(i, j) for i in df.columns.values for j in df.columns.values]

    # Initialize lists to save the chi2 and the p values
    chi2, p_values = [], []

    # For each attribute pair
    for factor in factors_paired:
        # If the factor pair is not identical
        if factor[0] != factor[1]:
            # Compute the contingency table of the attributes pair
            chitest = chi2_contingency(pd.crosstab(df[factor[0]], df[factor[1]]))
            # Add the chi2 value to the pre-defined list
            chi2.append(chitest[0])
            # Add the p-value to the pre-defined list
            p_values.append(chitest[1])

        # If the factor pair is identical
        else:
            # Add 0 to the pre-defined lists
            chi2.append(0)
            p_values.append(0)


    # Reshape the list containing the chi2 values to a matrix
    chi2 = np.array(chi2).reshape((df.shape[1], df.shape[1]))
    # Transfer the matrix to a dataFrame
    chi2 = pd.DataFrame(chi2, index=df.columns.values,columns=df.columns.values)
    # Normalize the contingency table
    normalized_chi2 = (chi2 - np.min(chi2))/np.ptp(chi2)

    # Return normalized contingency tables
    return normalized_chi2
