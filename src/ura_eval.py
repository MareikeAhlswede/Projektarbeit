"""
This module is used to perform the test of the Univariate Resemblance Analysis.
"""



# Imports
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, kstest, chi2_contingency, wasserstein_distance
from scipy.spatial import distance



def t_test(real_data, synthetic_data):
    """
    Performs a student t-tests to compare numerical attributes of real data and synthetic data.


    Parameters
    ----------
    real_data : pandas.core.frame.DataFrame
        The dataframe containing the real data.
    synthetic_data : pandas.core.frame.DataFrame
        The dataframe containing the synthetic data.


    Returns
    -------
    list
        A list of p-values containing the results of the statistical tests.
    """


    # Get a list of numerical column names
    num_cols = (real_data.select_dtypes(include=['int64'])).columns.tolist()

    # Initialize a list to save the p-values of the tests
    p_values = []

    # For each numerical column
    for col in num_cols:
        # Perform a t-test between real and synthetic data
        _, p_val = ttest_ind(real_data[col], synthetic_data[col])
        # Append the p value to the list
        p_values.append(p_val)

    # Return the obtained p-values
    return p_values



def mwu_test(real_data, synthetic_data):
    """
    Performs a Mann-Whitney-U test to compare numerical attributes of real and synthetic data.


    Parameters
    ----------
    real_data : pandas.core.frame.DataFrame
        The dataframe containing the real data
    synthetic_data : pandas.core.frame.DataFrame
        The dataframe containing the synthetic data


    Returns
    -------
    list
        A list of p-values containing the results of the statistical tests.
    """


    # Get a list of numerical column names
    num_cols = (real_data.select_dtypes(include=['int64'])).columns.tolist()

    # Initialize a list to save the p-values of the tests
    p_values = []

    # For each numerical column
    for col in num_cols:
        # Perform a Mann-Whitney U test between real and synthetic data
        _, p_val = mannwhitneyu(real_data[col], synthetic_data[col])
        # Append the p value to the list
        p_values.append(p_val)

    # Return the obtained p-values
    return p_values



def ks_test(real_data, synthetic_data):
    """
    Performs a Kolomogrov-Smirnov test to compare numerical attributes of real and synthetic data.


    Parameters
    ----------
    real_data : pandas.core.frame.DataFrame
        The dataframe containing the real data
    synthetic_data : pandas.core.frame.DataFrame
        The dataframe containing the synhetic data


    Returns
    -------
    list
        A list of p-values containing the results of the statistical tests.
    """


    # Get a list of numerical column names
    num_cols = (real_data.select_dtypes(include=['int64'])).columns.tolist()

    # Initialize a list to save the p-values of the tests
    p_values = []

    # For each numerical column
    for col in num_cols:
        # Perform a t-test between real and synthetic data
        _, p_val = kstest(real_data[col], synthetic_data[col])
        # Append the p value to the list
        p_values.append(p_val)

    # Return the obtained p-values
    return p_values



def chi2_test(real_data, synthetic_data):
    """
    Performs a Chi2 tests to compare categorical attributes of real data and synthetic data.


    Parameters
    ----------
    real_data : pandas.core.frame.DataFrame
        The dataframe containing the real data
    synthetic_data : pandas.core.frame.DataFrame
        The dataframe containing the synthetic data


    Returns
    -------
    list
        A list of p-values containing the results of the statistical tests.
    """


    # Get list of categorical column names
    cat_cols = (real_data.select_dtypes(include=['category'])).columns.tolist()

    # Initialize a list to save the p-values of the tests
    p_values = []

    # For each categorical column
    for col in cat_cols:
        # Create a contingency table
        contingency = pd.crosstab(real_data[col], synthetic_data[col])
        # Perform a chi2 test
        _, p_val, _, _ = chi2_contingency(contingency)
        # Append the p value to the list
        p_values.append(p_val)

    # Return the obtained p-values
    return p_values



def cosinus_dist(real_data, synthetic_data):
    """
    Calculates the cosine distance to compare numerical attributes of real and synthetic data.


    Parameters
    ----------
    real_data : pandas.core.frame.DataFrame
        The dataframe containing the scaled numerical attributes of the real data.
    synthetic_data : pandas.core.frame.DataFrame
        The dataframe containing the scaled numerical attributes of the synthetic data


    Returns
    -------
    list
        A list of distances containing the results of the statistical tests.
    """


    # Get list of numerical column names
    num_cols = real_data.columns.tolist()

    # Initialize a list to save the distance measures
    dists = []

    # For each numerical column
    for col in num_cols:
        # Calculate the distance between real and synthetic data
        dists.append(distance.cosine(real_data[col].values, synthetic_data[col].values))

    # Return the obtained distances
    return dists



def wasserstein_dist(real_data, synthetic_data):
    """
    Calculates the Wasserstein distance to compare numerical attributes of
    real and synthetic data.



    Parameters
    ----------
    real_data : pandas.core.frame.DataFrame
        The dataframe containing the scaled numerical attributes of the real data.
    synthetic_data : pandas.core.frame.DataFrame
        The dataframe containing the scaled numerical attributes of the synthetic data.


    Returns
    -------
    list
        A list of distances containing the results of the statistical tests.
    """


    # Get list of numerical column names
    num_cols = real_data.columns.tolist()

    # Initialize a list to save the distance measures
    dists = []

    # For each numerical column
    for col in num_cols:
        # Calculate the distance between real and synthetic data
        dists.append(wasserstein_distance(real_data[col].values, synthetic_data[col].values))

    # Return the obtained distances
    return dists
