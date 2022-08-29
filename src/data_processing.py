"""
This module creates a DataProcessor class used to transform the training data before GAN training
and for inverse transformation of the sampled data. Also, this module can be used to scale data.
"""



# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler, \
                                    LabelEncoder, OneHotEncoder



class DataProcessor:
    """
    A class used for processing of the real and synthetic data.


    Attributes:
    -----------
    int_cols: list
        A list containing the names of the integer columns.
    cat_cols: list
        A list containing the names of the categorical columns.
    train_data: pandas.core.frame.DataFrame
        A dataframe used for the training of the GANs.
    ordinal_encoder: dict
        An ordinal encoder used to transform the categorical columns.
    std_scaler: dict
        A scaler used to transform the columns to have mean zero and a standard deviation of one.
    label_encoder
        A label encoder used to transform the categorical columns.
    onehot_encoder
        An one-hot encoder used to transform the categorical columns.

    Methods:
    --------
    transform()
        Uses an encoder and scaler for the transformation of the training data.
    inverse_transform()
        Used the enocder and scaler to inverse transform the generated symples to
        obtain the synthetic data.
    scale_num()
        Used to scale numeric data between zero and one.
    scale_all()
        Used to scale numeric data with mean zero and standard deviation of one
        and one-hot encode categorical data.
    """



    def __init__(self, raw_data):
        """
        Initialization of the class.


        Parameters
        ----------
        raw_data: pandas.core.frame.DataFrame
            The raw data to be processed.
        """

        # Define the lists containing the names of the int and category columns.
        self.int_cols = raw_data.select_dtypes(include=['int64']).columns.tolist()
        self.cat_cols = raw_data.select_dtypes(include=['category']).columns.tolist()

        # Define the training data
        self.train_data = raw_data

        # Define categories for all categorical features
        gender_cat = ["Female", "Male", "Unknown/Invalid"]
        age_cat = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)",
                "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
        adtype_cat = ["1", "2", "3", "4", "5", "6", "7", "8"]
        disch_cat = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14",
                    "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27",
                    "28", "29"]
        adsrc_cat = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14",
                    "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26"]
        maxglu_cat = ["None", ">200", ">300", "Norm"]
        a1res_cat = ["Norm", "None", ">8", ">7"]
        change_cat = ["No", "Ch"]
        dimed_cat = ["No", "Yes"]
        read_cat = ['NO', '>30', '<30']

        # Define an ordinal encoder to transform the categorical columns with given categories
        self.ordinal_encoder = OrdinalEncoder(categories=[gender_cat, age_cat, adtype_cat,
                                                        disch_cat, adsrc_cat, maxglu_cat, a1res_cat,
                                                        change_cat, dimed_cat, read_cat],
                                                        handle_unknown = 'use_encoded_value',
                                                        unknown_value= -1)

        # Define a scaler scaling the data with mean zero and standard deviation of one
        self.std_scaler = StandardScaler()

        # Define a label encoder
        self.label_encoder = dict()

        # Initialize a dictionary to save the label encoders in
        categorical_vars = pd.DataFrame()

        # For each categorical variable
        for cat in self.cat_cols:
            # Fit an save a label encoder to the categorical data
            self.label_encoder[cat] = LabelEncoder().fit(raw_data[cat])
            # And encode the categorical data
            categorical_vars[cat] = self.label_encoder[cat].transform(raw_data[cat])

        # Use the label encoded data to fit the one-hot encoder
        self.onehot_encoder = OneHotEncoder().fit(categorical_vars)


    def transform(self):
        """
        Preprocesses the training data using an ordinal encoder and scaler.

        Returns
        -------
        pandas.core.frame.DataFrame
            A dataframe containing the pre-processed training data.
        """

        # Take a copy of the training data and use it for further transformation
        prepro_data = self.train_data.copy()

        # Apply ordinal encoding to the categorical attributes
        cat_data = pd.DataFrame(self.ordinal_encoder.fit_transform(prepro_data[self.cat_cols]),
                                    columns = self.cat_cols)

        # Apply scaling to the numerical attributres
        num_data = pd.DataFrame(self.std_scaler.fit_transform(prepro_data[self.int_cols]),
                                    columns = self.int_cols)

        # Concatinate the data of the categorical and numeric attributes
        prepro_data = pd.concat([num_data, cat_data] , axis=1, join='inner')

        # Set the correct datatypes of the preprocessed data
        prepro_data[self.cat_cols] = prepro_data[self.cat_cols].astype('category')
        prepro_data[self.int_cols] = prepro_data[self.int_cols].astype('float64')

        # Reindex the preprocessed data so that it has the original form of the training data
        prepro_data = prepro_data.reindex(self.train_data.columns.tolist(), axis=1)

        # Return the preprocessed training data.
        return prepro_data



    def inverse_transform(self, sample_data):
        """
        Use and ordinal enocder and scaler to inverse transform the synthetic samples to
        obtain the synthetic data.

        Parameters
        ----------
        sample_data : pandas.core.frame.DataFrame
            A dataframe with the synthetic data to be transformed.

        Returns
        -------
        pandas.core.frame.DataFrame
            A dataframe with the inverse transformed synthetic data.
        """

        # Take a copy of the sample data
        synthetic_data = sample_data.copy()

        # Inverse transform the ordinal encoding of categorical columns
        cat_data = pd.DataFrame(self.ordinal_encoder.inverse_transform(synthetic_data[self.cat_cols]),
                                    columns=self.cat_cols)

        # Rescale all values to their original mean and standard deviations
        num_data = pd.DataFrame(self.std_scaler.inverse_transform(synthetic_data[self.int_cols]),
                                columns=self.int_cols)

        # Concatinate the categorical and numeric attributes
        synth_data = pd.concat([num_data, cat_data], axis=1, join='inner')

        # Set the correct datatypes of the preprocessed data
        synth_data[self.cat_cols] = synth_data[self.cat_cols].astype('category')
        synth_data[self.int_cols] = synth_data[self.int_cols].astype('int64')

        # Reindex the preprocessed data so that it has the original form of the training data
        synth_data = synth_data.reindex(self.train_data.columns.tolist(), axis=1)

        # Return the transformed synthetic data.
        return synth_data



    def scale_num(self, dataframe):
        """
        Scale a dataframe so that all numeric attributes lie between 0 and 1.

        Parameters
        ----------
        dataframe : pandas.core.frame.DataFrame
            A dataframe to scale.

        Returns
        -------
        pandas.core.frame.DataFrame
            A dataframe with the scaled data.
        """

        # Initialize and fit the scaler
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(dataframe[self.int_cols])

        # Return the scaled dataframe
        return pd.DataFrame(scaled, columns=dataframe[self.int_cols].columns.tolist())


    def scale_all(self, dataframe):
        """
        Used to scale numeric data with mean zero and standard deviation of one
        and one-hot encode categorical data.

        Parameters
        ----------
        data : pandas.core.frame.DataFrame
            The data to be transformed.

        Returns
        -------
        numpy.ndarray
            The transformed data.
        """

        # Initialize a dataframe to save the encoded categorical attributes
        categorical_vars = pd.DataFrame()

        # For each categorical attribute
        for cat in self.cat_cols:
            # Transform the categorical data using a label encoder
            categorical_vars[cat] = self.label_encoder[cat].transform(dataframe[cat])

        # Apply one-hot encoding to the categorical data and transfer it to an array
        x_cat = self.onehot_encoder.transform(categorical_vars).toarray()

        # Scale all numerical attributes to have mean zero and a standard deviation of one
        x_num = self.std_scaler().transform(dataframe[self.int_cols])

        # Return the transformed data as an ndarray
        return np.column_stack((x_num, x_cat)).astype('float32')
