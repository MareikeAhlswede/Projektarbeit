"""
This module is the main script used to sample the data and evaluate the results
between real data and synthetic data sample from different GAN architectures.
"""



# Imports
import warnings
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import seaborn as sns
from matplotlib import pyplot as plt
from model_training import ModelTrainer
from data_processing import DataProcessor
from ura_eval import cosinus_dist, t_test, mwu_test, ks_test, chi2_test, cosinus_dist, \
       jensenshannon_dist, wasserstein_dist
from mra_eval import get_categorical_correlations
from dra_eval import isomap_transform_on_batch , pca_transform, dra_distance

from ydata_synthetic.synthesizers.regular import CRAMERGAN

warnings.filterwarnings("ignore")



# Read in the original data and only extract the twenty features necessary for further analysis
# along with their respective data type
data_orig = pd.read_csv(r'.\data\diabetic_data.csv', sep=',',
                   usecols=['encounter_id', 'patient_nbr', 'gender', 'age', 'admission_type_id',
                            'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
                            'num_lab_procedures', 'num_procedures', 'num_medications',
                            'number_outpatient', 'number_emergency', 'number_inpatient',
                            'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'change',
                            'diabetesMed', 'readmitted'],
                   dtype={'encounter_id': np.int64, 'patient_nbr': np.int64, 'gender': 'category',
                          'age': 'category', 'admission_type_id': 'category',
                          'discharge_disposition_id': 'category',
                          'admission_source_id': 'category', 'time_in_hospital': np.int64,
                          'num_lab_procedures': np.int64, 'num_procedures': np.int64,
                          'num_medications': np.int64, 'number_outpatient': np.int64,
                          'number_emergency': np.int64, 'number_inpatient': np.int64,
                          'number_diagnoses': np.int64, 'max_glu_serum': 'category',
                          'A1Cresult': 'category', 'change': 'category',
                          'diabetesMed': 'category', 'readmitted': 'category'})


# Print an overview of the data
# print(data_orig.head())

# Print out the number of missing values per column in %
# missing_data = data_orig.isnull().mean()*100
# missing_data = missing_data[missing_data!=0]
# print(missing_data.empty)
# --> True, no missing data in imported dataset



# Split the data into test and training data (20% test data, 80% training data)
data_train, data_test = train_test_split(data_orig, test_size=0.2, random_state=42)

# Initialize a DataProcessor class used to transform the training data before GAN training
# and for inverse transformation of the sampled data
processor = DataProcessor(data_train)

# Transforms the training data so that categorical values are ordinal encoded and then all
# all values are scaled with mean zero and standard deviation one.
prepro_data_train = processor.transform()

# Define the batch sizes used for training the different GAN algorithms
# batch_sizes = [1000]#, 100]  # 10000,

# # # If it does not exist
# # if not os.path.exists("./saved/gan"):
# #     # Create a path to save the trained GAN Generators (Synthesizers)
# #     os.makedirs("./saved/gan")

# For each batch size
# for bs in batch_sizes:

#     # Initialize a ModelTrainer with the preprocessed training data and the selected batch_size
#     trainer = ModelTrainer(prepro_data_train, batch_size=bs)

#     # Create and train a GAN
#     gan_synthesizer = trainer.gan_training()

#     # Create a path to save the Generator of the GAN model.
#     GAN_PATH = "./saved/gan/GAN_n5d30bs" + str(bs) + ".pkl"

#     # Save the GAN model.
#     gan_synthesizer.save(path=GAN_PATH)

#     # Sample synthetic data from the GAN model
#     gan_sample = gan_synthesizer.sample(110000).iloc[0:101766, :]

#     # Inverse transform the synthetic data so that it has the original form of the real data
#     gan_sample = processor.inverse_transform(gan_sample)

#     # Create a path to save the synthetic data of the GAN model.
#     GAN_PATH = "./saved/data/Sample_GAN_n5d30bs" + str(bs) + ".csv"

#     # Save the synthetic data of the GAN model.
#     gan_sample.to_csv(GAN_PATH)


    # # Do the same for the WGAN
    # wgan_synthesizer = trainer.wgan_training()
    # WGAN_PATH = "./saved/gan/WGAN_n5d30bs" + str(bs) + ".pkl"
    # wgan_synthesizer.save(path=WGAN_PATH)
    # wgan_sample = wgan_synthesizer.sample(110000).iloc[0:101766, :]
    # wgan_sample = processor.inverse_transform(wgan_sample)
    # WGAN_PATH = "./saved/data/Sample_WGAN_n5d30bs" + str(bs) + ".csv"
    # wgan_sample.to_csv(WGAN_PATH)


#     # Do the same for the WGAN-GP
#     wgangp_synthesizer = trainer.wgangp_training()
#     WGANGP_PATH = "./saved/gan/WGANGP_n5d30bs" + str(bs) + ".pkl"
#     wgangp_synthesizer.save(path=WGANGP_PATH)
#     wgangp_sample = wgangp_synthesizer.sample(110000).iloc[0:101766, :]
#     wgangp_sample = processor.inverse_transform(wgangp_sample)
#     WGANGP_PATH = "./saved/data/Sample_WGANGP_n5d30bs" + str(bs) + ".csv"
#     wgangp_sample.to_csv(WGANGP_PATH)


#     # Do the same for the CramerGAN
#     cramergan_synthesizer = trainer.cramergan_training()
#     CRAMERGAN_PATH = "./saved/gan/CramerGAN_n5d30bs" + str(bs) + ".pkl"
#     cramergan_synthesizer.save(path=CRAMERGAN_PATH)

    # # Select the CramerGAN model
    # model = CRAMERGAN
    # cramergan_synthesizer = model.load(path='./saved/gan/CramerGAN_n5d30bs1000.pkl')




    # cramergan_sample = cramergan_synthesizer.sample(110000).iloc[0:101766, :]
    # print(cramergan_sample)
    # # Print out the number of missing values per column in %
    # missing_data = cramergan_sample.isnull().mean()*100
    # #missing_data = missing_data[missing_data!=0]
    # print(missing_data)
    # cramergan_sample = processor.inverse_transform(cramergan_sample)
    # CRAMERGAN_PATH = "./saved/data/Sample_CramerGAN_n5d30bs" + \
    #     str(bs) + ".csv"
    # cramergan_sample.to_csv(CRAMERGAN_PATH)

# Define all GAN models
synthesizers = ['GAN', 'WGAN', 'WGANGP']

gan = pd.read_csv(r'.\saved\data\Sample_GAN_200bs1000.csv',
                        index_col=0, dtype={'encounter_id': np.int64, 'patient_nbr': np.int64,
                        'gender': 'category', 'age': 'category', 'admission_type_id': 'category',
                        'discharge_disposition_id': 'category','admission_source_id': 'category',
                        'time_in_hospital': np.int64, 'num_lab_procedures': np.int64,
                        'num_procedures': np.int64, 'num_medications': np.int64,
                        'number_outpatient': np.int64, 'number_emergency': np.int64,
                        'number_inpatient': np.int64, 'number_diagnoses': np.int64,
                        'max_glu_serum': 'category', 'A1Cresult': 'category', 'change': 'category',
                        'diabetesMed': 'category', 'readmitted': 'category'})#.iloc[0:81412, :]

wgan = pd.read_csv(r'.\saved\data\Sample_WGAN_200bs1000.csv',
                        index_col=0, dtype={'encounter_id': np.int64, 'patient_nbr': np.int64,
                        'gender': 'category', 'age': 'category', 'admission_type_id': 'category',
                        'discharge_disposition_id': 'category','admission_source_id': 'category',
                        'time_in_hospital': np.int64,'num_lab_procedures': np.int64,
                        'num_procedures': np.int64,'num_medications': np.int64,
                        'number_outpatient': np.int64,'number_emergency': np.int64,
                        'number_inpatient': np.int64,'number_diagnoses': np.int64,
                        'max_glu_serum': 'category','A1Cresult': 'category', 'change': 'category',
                        'diabetesMed': 'category', 'readmitted': 'category'})#.iloc[0:81412, :]

wgangp = pd.read_csv(r'.\saved\data\Sample_WGANGP_200bs1000.csv',
                        index_col=0, dtype={'encounter_id': np.int64, 'patient_nbr': np.int64,
                        'gender': 'category', 'age': 'category', 'admission_type_id': 'category',
                        'discharge_disposition_id': 'category', 'admission_source_id': 'category',
                        'time_in_hospital': np.int64, 'num_lab_procedures': np.int64,
                        'num_procedures': np.int64, 'num_medications': np.int64,
                        'number_outpatient': np.int64, 'number_emergency': np.int64,
                        'number_inpatient': np.int64, 'number_diagnoses': np.int64,
                        'max_glu_serum': 'category', 'A1Cresult': 'category', 'change': 'category',
                        'diabetesMed': 'category', 'readmitted': 'category'})#.iloc[0:81412, :]

# Create a dictionary containing all datasets to be used for the evaluation
data = {'Real': data_orig, 'GAN': gan, 'WGAN': wgan,'WGANGP': wgangp}

# Select all numerical columns
num_cols = (data['Real'].select_dtypes(include=['int64'])).columns

# Select all numerical columns
cat_cols = (data['Real'].select_dtypes(include=['category'])).columns

# Create a dictionary holding the scaled datasets
scaled_num_data = dict()

# For all datasets
for name in list(data.keys()):
    # Scale the datasets with values between zero and one
    scaled_num_data[name] = processor.scale_num(data[name])



#--------------------------------------------------------------------------------------------------
# Univariate Resemblance analysis (URA)

print("URA")

# Create a dictionary holding the results of the t-test between real and synthetic data
p_values_student = dict()

# For each sample of synthetic data
for name in synthesizers:
    # Calculate the t-test for each numerical column beteen real and synthetic data
    p_values_student[name] = t_test(data['Real'], data[name])

# Enter the results into a dictionary
df_student_test = pd.DataFrame(data=p_values_student,
                                index=(data['Real'].select_dtypes(include=['int64'])).columns)

# Save the results of the t-tests as a .csv file
df_student_test.to_csv(r'.\saved\evaluation\resemblance\URA\01_URA_ttest.csv')



# Create a dictionary holding the results of the Mann-Whitney U (MWU) test
# between real and synthetic data
p_values_mwu = dict()

# For each sample of synthetic data
for name in synthesizers:
    # Calculate the MWU test for each numerical column beteen real and synthetic data
    p_values_mwu[name] = mwu_test(data['Real'], data[name])

# Enter the results in a dictionary
df_mwu_test = pd.DataFrame(data=p_values_mwu,
                            index=(data['Real'].select_dtypes(include=['int64'])).columns)

# Save the results of the MWU test as a .csv file
df_mwu_test.to_csv(r'.\saved\evaluation\resemblance\URA\02_URA_mwutest.csv')



# Create a dictionary holding the results of the Kolmogrov Smirnov (KS) test
# between real and synthetic data
p_values_ks = dict()

# For each sample of synthetic data
for name in synthesizers:
    # Calculate the KS test of each numerical column beteen real and synthetic data
    p_values_ks[name] = ks_test(data['Real'], data[name])

# Enter the results in a dictionary
df_ks_test = pd.DataFrame(data=p_values_ks,
                            index=(data['Real'].select_dtypes(include=['int64'])).columns)

# Save the results of the KS test as a .csv file
df_ks_test.to_csv(r'.\saved\evaluation\resemblance\URA\03_URA_kstest.csv')



# Create a dictionary holding the results of the Chi2 test between real and synthetic data
p_values_chi = dict()

# For each sample of synthetic data
for name in synthesizers:
    # Calculate the Chi2 test beteen real and synthetic data
    p_values_chi[name] = chi2_test(data['Real'], data[name])

# Enter the results in a dictionary
df_chi_test = pd.DataFrame(data=p_values_chi,
                            index=(data['Real'].select_dtypes(include=['int64'])).columns)

# Save the results of the Chi2 test as a .csv file
df_chi_test.to_csv(r'.\saved\evaluation\resemblance\URA\04_URA_chi2test.csv')



# Create a dictionary holding the cosine distance measures between real and synthetic data
cos_dist = dict()

# For each sample of synthetic data
for name in synthesizers:
    # Calculate the cosine distance beteen real and synthetic data
    cos_dist[name] = cosinus_dist(scaled_num_data['Real'], scaled_num_data[name])

# Enter the results in a dictionary
df_cos_dist = pd.DataFrame(data=cos_dist,
                            index=(data['Real'].select_dtypes(include=['int64'])).columns)

# Save the cosine distances as a .csv file
df_cos_dist.to_csv(r'.\saved\evaluation\resemblance\URA\05_URA_cosdist.csv')



# Create a dictionary holding the Jensen-Shannon distance measures between real and synthetic data
js_dist = dict()

# For each sample of synthetic data
for name in synthesizers:
    # Calculate the Jensen-Shannon distance beteen real and synthetic data
    js_dist[name] = jensenshannon_dist(
        scaled_num_data['Real'], scaled_num_data[name])

# Enter the results in a dictionary
df_js_dist = pd.DataFrame(data=js_dist,
                            index=(data['Real'].select_dtypes(include=['int64'])).columns)

# Save the Jensen-Shannon distances as a .csv file
df_js_dist.to_csv(r'.\saved\evaluation\resemblance\URA\06_URA_jsdist.csv')



# Create a dictionary holding the Wasserstein distance measures between real and synthetic data
ws_dist = dict()

# For each sample of synthetic data
for name in synthesizers:
    # Calculate the Wasserstein distance beteen real and synthetic data
    ws_dist[name] = wasserstein_dist(
        scaled_num_data['Real'], scaled_num_data[name])

# Enter the results in a dictionary
df_ws_dist = pd.DataFrame(data=ws_dist,
                            index=(data['Real'].select_dtypes(include=['int64'])).columns)

# Save the Wasserstein distances as a .csv file
df_ws_dist.to_csv(r'.\saved\evaluation\resemblance\URA\07_URA_wsdist.csv')



# Get all columns names
columns = data['Real'].columns

# A dictionary holding the combined values of the same column from different datasets
hists_data = dict()

# For each column
for col in columns:
    # Add the values of the real data
    hists_data[col] = data['Real'][col]
    # For each GAN models
    for name in list(data.keys())[1:]:
        # Add the values of the GAN model
        hists_data[col] = np.column_stack((hists_data[col], data[name][col]))


# Create a figure with 20 subplots (4 rows and 5 columns)
fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(16, 10))

# Create a dictionary with indexes for the subplot from 0 to 19
idx = {0: [0, 0], 1: [0,1], 2:[0,2], 3:[0,3], 4:[0,4], 5:[1,0], 6:[1,1], 7:[1,2], 8:[1,3],
        9: [1, 4], 10: [2,0], 11:[2,1], 12:[2,2], 13:[2,3], 14:[2,4], 15:[3,0], 16:[3,1], 17:[3,2],
        18: [3, 3], 19: [3,4]}


# For each column
for i, col_name in enumerate(columns):

    # Select the respective subplot by using the pre-defined indexes
    ax = axs[idx[i][0], idx[i][1]]
    # Add the column name as the title
    ax.set_title(col_name)


    # If the column is numeric
    if i in [0,1,7,8,9,10,11,12,13,14]:
        # For each of the datasets
        for col in range(hists_data[col_name].shape[1]):
            # Sort the column
            arr_col = np.sort(hists_data[col_name][:, col])
            # Plot a distribution plot
            sns.kdeplot(arr_col[int(0.05 * len(arr_col)): int(0.95 * len(arr_col))],
                            ax=ax)


    # If the column is categorical
    else:
        # Plot a histogram containing all the column data of all datasets
        ax.hist(hists_data[col_name], density=False, histtype='bar', label=list(data.keys()),
                align='mid')

        # If the columns gender or readmitted was chosen
        if col_name in ['gender', 'readmitted']:
            # Define the position of the X-ticks
            ax.set_xticks([0.1, 1.1, 1.9])
            #ax.set_xticklabels([1,2,3,4])

        # If the column age was chosen
        elif col_name in ['age']:
            # Get the X-ticks
            ax.set_xticks(ax.get_xticks())
            # And add the categories as lables
            ax.set_xticklabels(data['Real']['age'].cat.categories.tolist(),rotation=90, ha="right")

        # If the column change or diabetesMed was chosen
        elif col_name in ['change', 'diabetesMed']:
            # Define the position of the X-ticks
            ax.set_xticks([0.05, 0.95])

        # If the column max_glu_serum or A1Cresult was chosen
        elif col_name in ['max_glu_serum', 'A1Cresult']:
            # Define the position of the X-ticks
            ax.set_xticks([0.15, 1.05, 1.95, 2.85])


    # Delete the y-ticks
    ax.set_yticks([])


# Format the plot so that all subplots fit the figure
fig.tight_layout()

# Add a legend
axs[idx[i][0], idx[i][1]].legend(ncol=5, bbox_to_anchor=(-0.8, -0.2))

# Adjust the subplot layout by changing the position of the top edge
fig.subplots_adjust(top=0.75)


# Save the figure
fig.savefig(r'.\saved\evaluation\resemblance\URA\08_URA_distplot.png', bbox_inches='tight')



# #--------------------------------------------------------------------------------------------------
# # MRA

print("MRA")
# Create dictionary to save computed Pearson correlation coefficient (PCC) matrixes of all datasets
cors_numerical = dict()

# For each dataset
for name in list(data.keys()):
    # Compute the PCC
    cors_numerical[name] = np.absolute(data[name].corr(method='pearson'))


# Create a figure with as many subplots as datasets
fig, axs = plt.subplots(nrows=1, ncols=len(data), figsize=(15, 2.5))

# Create an index for each subplot
axs_idxs = range(len(data)+1)

# Combine the dataset name and its subplot index
idx = dict(zip(list(data.keys()), axs_idxs))


# For each dataset
for name_idx, name in enumerate(list(data.keys())):

    # Select the respective axes
    ax = axs[idx[name]]

    # Select the respective correlation matrix
    matrix = cors_numerical[name]

    # Delete redundancy of the corelation matrix
    matrix = matrix.iloc[1:, 0:-1]

    # Compute the mask of the correlation matrix to plot only one side of it
    matrix_mask = np.triu(np.ones_like(matrix, dtype=bool)) - np.identity(len(matrix))

    # If the current correlation matrix is not the last one
    if name_idx != len(list(data.keys())) - 1:
        # Plot a heatmap with the correlation matrix values without a legend
        sns.heatmap(matrix,  linewidths=.3, ax=ax, mask=matrix_mask,
                        cbar=False, vmin=0, vmax=1, cmap='Blues')

    # If the current correlation matrix is the last one
    else:
        # Plot a heatmap with the correlation matrix values with a legend
        sns.heatmap(matrix,  linewidths=.3, ax=ax, mask=matrix_mask,
                    cbar=True, vmin=0, vmax=1, cmap='Blues')

    # For all plots except the first one
    if name_idx > 0:
        # Delete the y-ticks
        ax.set_yticks([])

    # If the correlation matrix of the real dataset is plotted
    if name == 'Real':
        # Name it 'Real'
        ax.set_title(name)

    # If the correlation matrix of a synthesized dataset is plotted
    else:
        # Get the correlations differences between real data and synthetic data
        diff = abs(cors_numerical['Real'] - matrix)

        # Cut of the upper triangle
        diffs = diff.values[np.triu_indices(len(diff), k=1)]

        # Find all differences with difference smaller than 0.1
        preserved_cors = len(diffs[diffs < 0.1])

        # Return the percentage of correlations preserved in synthetic data
        # (rounded to two decimals)
        score = np.round(preserved_cors/len(diffs), 2)

        # Set the title with name and percentage of preserved correlations
        ax.set_title(name + ' (' + str(score) + ')')

# Save the figure
fig.savefig(r'.\saved\evaluation\resemblance\MRA\01_MRA_numcorr.png', bbox_inches='tight')



# Create dictionary to save computed normalized contingency tables of all datasets
cors_categorical = dict()

# For each dataset
for name in list(data.keys()):
    # Get the categorical correlations
    cors_categorical[name] = get_categorical_correlations(data[name])


# Create a figure with as many subplots as datasets
fig, axs = plt.subplots(nrows=1, ncols=len(data), figsize=(15, 2.5))

# Create an index for each subplot
axs_idxs = range(6)

# Combine the dataset name and its subplot index
idx = dict(zip(list(data.keys()), axs_idxs))


# For each dataset
for name_idx, name in enumerate(list(data.keys())):

    # Select the respective axes
    ax = axs[idx[name]]

    # Select the respective correlation matrix
    matrix = cors_categorical[name]

    # Delete redundancy of the corelation matrix
    matrix = matrix.iloc[1:, 0:-1]

    # Compute the mask of the correlation matrix to plot only one side of it
    cors_mask = np.triu(np.ones_like(matrix, dtype=bool)) - \
        np.identity(len(matrix))

    # If the current correlation matrix is not the last one
    if name_idx != len(list(data.keys())) - 1:

        # Plot a heatmap with the correlation matrix values without a legend
        sns.heatmap(matrix,  linewidths=.3, ax=ax, mask=cors_mask,
                    cbar=False, vmin=0, vmax=1, cmap='Blues')

    # If the current correlation matrix is the last one
    else:
        # Plot a heatmap with the correlation matrix values with a legend
        sns.heatmap(matrix,  linewidths=.3, ax=ax, mask=cors_mask,
                cbar=True, vmin=0, vmax=1, cmap='Blues')

    # For all plots except the first one
    if name_idx > 0:
        # Delete the y-ticks
        ax.set_yticks([])

    # If the correlation matrix of the real dataset is plotted
    if name == 'Real':
        # Name it 'Real'
        ax.set_title(name)

    # If the correlation matrix of a synthesized dataset is plotted
    else:
        # Get the correlations differences between real data and synthetic data
        diff = abs(cors_categorical['Real'] - matrix)

        # Cut off the upper triangle
        diffs = diff.values[np.triu_indices(len(diff), k=1)]

        # Find all differences with difference smaller than 0.1
        preserved_cors = len(diffs[diffs < 0.1])

        # Return the percentage of correlations preserved in synthetic data
        # (rounded to two decimals)
        score = np.round(preserved_cors/len(diffs), 2)

        # Set the title with name and percentage of preserved correlations
        ax.set_title(name + ' (' + str(score) + ')')


# Save the figure
fig.savefig(r'.\saved\evaluation\resemblance\MRA\02_MRA_catcorr.png', bbox_inches='tight')



#--------------------------------------------------------------------------------------------------
# DRA

print("DRA")

scaled_all_data = dict()

for key, value in data.items():
    scaled_all_data[key] = processor.scale_all(value)

data_pca = dict()
for key, _ in data.items():
    data_pca[key] = pca_transform(scaled_all_data[key])

joint_dist_pca = pd.DataFrame(columns=['joint_dist'], index=synthesizers)

# Get the standard color map
cmap = plt.get_cmap("tab10")

# Create a new figure
fig, axes = plt.subplots(1, len(synthesizers), figsize=(12, 2))

# For each synthetically created patient dataset
for counter, key in enumerate(synthesizers):
    # Scatter the pca results of the real dataset
    axes[counter].scatter(data_pca['Real'].loc[:, 'PC1'], data_pca['Real'].loc[:, 'PC2'],
                            c=np.array([cmap(0)]), s=20, alpha=0.5)
    # Ontop, scatter the pca results of the synthetic dataset
    axes[counter].scatter(data_pca[key].loc[:, 'PC1'], data_pca[key].loc[:, 'PC2'],
                            c=np.array([cmap(counter+1)]), s=20, alpha=0.5)

    # Label the x and y axis
    axes[counter].set_xlabel('PC1')
    axes[counter].set_ylabel('PC2')

    # Set the name of the GAN used for synthetic data generation as the title
    axes[counter].set_title(key)
    axes[counter].set_xticks([])
    axes[counter].set_yticks([])

    joint_dist_pca.iloc[counter, 0] = dra_distance(data_pca['Real'], data_pca[key])



# # Save the figure containing the pca results
fig.savefig(r'.\saved\evaluation\resemblance\DRA\01_DRA_PCA.png',
            bbox_inches='tight')

# Save the joint distances as a .csv file
joint_dist_pca.to_csv(r'.\saved\evaluation\resemblance\DRA\02_DRA_PCA_jointdist.csv')






# data_iso = dict()
# for name in data.keys():
#     data_iso[name] = isomap_transform_on_batch(scaled_all_data[name])

# print(data_iso['Real'])

# joint_dist_iso = pd.DataFrame(columns=['joint_dist'], index=synthesizers)

# # Get the standard color map
# cmap = plt.get_cmap("tab10")

# # Create a new figure
# fig, axes = plt.subplots(1, len(synthesizers), figsize=(12, 2))

# # For each synthetically created patient dataset
# for counter, key in enumerate(synthesizers):
#     # Scatter the iso results of the real dataset
#     axes[counter].scatter(data_iso['Real'].loc[:, 'PC1'],
#                           data_iso['Real'].loc[:, 'PC2'], c=np.array([cmap(0)]), s=20, alpha=0.5)
#     # Ontop, scatter the iso results of the synthetic dataset
#     axes[counter].scatter(data_iso[key].loc[:, 'PC1'],
#                           data_iso[key].loc[:, 'PC2'], c=np.array([cmap(counter+1)]), s=20, alpha=0.5)

#     # Label the x and y axis
#     axes[counter].set_xlabel('PC1')
#     axes[counter].set_ylabel('PC2')

#     # Set the name of the GAN used for synthetic data generation as the title
#     axes[counter].set_title(key)
#     axes[counter].set_xticks([])
#     axes[counter].set_yticks([])

#     joint_dist_iso.iloc[counter, 0] = dra_distance(data_iso['Real'], data_iso[key])


# # Save the figure containing the iso results
# fig.savefig(r'.\saved\evaluation\resemblance\DRA\03_DRA_ISO.png',
#             bbox_inches='tight')

# # Save the joint distances as a .csv file
# joint_dist_iso.to_csv(
#     r'.\saved\evaluation\resemblance\DRA\04_DRA_ISO_jointdist.csv')


#------------------------------------------------   ----------------------------------------------
# 3.1.4 Data Labeling Analysis (DLA)
 # Encoding and scaling necessary?

print("DLA")

# For re-identification during classification, add a new target variable to the real dataset
scaled_real = scaled_all_data['Real']
scaled_real = pd.DataFrame(scaled_real)
scaled_real['identification'] = 0.0
scaled_all_data['Real'] = scaled_real


# Create a new figure with number of plots equal to the number of synthetic datasets
fig, axes = plt.subplots(1, len(synthesizers), figsize=(12, 2))

# For each synthetically created patient dataset
for counter, key in enumerate(synthesizers):

    # Create a new dataframe, containing the accuracy, precision, recall and f1 score of the
    # different classification algorithms
    dla_df = pd.DataFrame(index=['rf', 'knn', 'dt', 'svm', 'mlp'],
                            columns=['acc', 'prec', 'rec', 'f1'])

    # For re-identification during classification, add a new target variable to the current
    # synthetic dataset

    scaled_synth = scaled_all_data[key]
    scaled_synth = pd.DataFrame(scaled_synth)
    scaled_synth['identification'] = 1.0
    scaled_all_data[key] = scaled_synth

    # Concatinate the real and the synthetic datasets
    combi_data = pd.concat([scaled_all_data['Real'], scaled_all_data[key]], axis=0)

    # Create a randomly sampled test and a training dataset from the concatinated datasets
    X_train, X_test, y_train, y_test = train_test_split(combi_data.drop(['identification'],
                            axis=1), combi_data['identification'],
                            test_size=0.2, shuffle=True, random_state=42)

    # Create a random forst classifier with given parameters
    clf = RandomForestClassifier(n_estimators=100, random_state=9, verbose=True, n_jobs=3)
    # Build a forest of trees from the training set
    clf.fit(X_train, y_train)
    # Predict the class labels of the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy, precision, recall and f1 score of the random forest classifier
    # and write them into the pre-defined dataframe
    dla_df.loc['rf', 'acc'] = accuracy_score(y_test, y_pred)
    dla_df.loc['rf', 'prec'] = precision_score(y_test, y_pred)
    dla_df.loc['rf', 'rec'] = recall_score(y_test, y_pred)
    dla_df.loc['rf', 'f1'] = f1_score(y_test, y_pred)

    # Create a k-nearest neighbors classifier with given parameters
    clf = KNeighborsClassifier(n_neighbors=10, n_jobs=3)
    #Fit the k-nearest neighbors classifier from the training dataset
    clf.fit(X_train, y_train)
    # Predict the class labels of the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy, precision, recall and f1 score of the k-nearest neighbors
    # classifier and write them into the pre-defined dataframe
    dla_df.loc['knn', 'acc'] = accuracy_score(y_test, y_pred)
    dla_df.loc['knn', 'prec'] = precision_score(y_test, y_pred)
    dla_df.loc['knn', 'rec'] = recall_score(y_test, y_pred)
    dla_df.loc['knn', 'f1'] = f1_score(y_test, y_pred)

    #  Create a decision tree classifier with given parameters
    clf = DecisionTreeClassifier(random_state=9)
    # Build a decision tree classifier from the training set
    clf.fit(X_train, y_train)
    # Predict the class labels of the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy, precision, recall and f1 score of the decision tree
    # classifier and write them into the pre-defined dataframe
    dla_df.loc['dt', 'acc'] = accuracy_score(y_test, y_pred)
    dla_df.loc['dt', 'prec'] = precision_score(y_test, y_pred)
    dla_df.loc['dt', 'rec'] = recall_score(y_test, y_pred)
    dla_df.loc['dt', 'f1'] = f1_score(y_test, y_pred)

    # Create a C-Support Vector classification with given parameters
    clf = SVC(C=100, max_iter=300, kernel="linear", probability=True,
                random_state=9, verbose=1)
    # Fit the SVM model according to the given training data
    clf.fit(X_train, y_train)
    # Predict the class labels of the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy, precision, recall and f1 score of the C-Support Vector
    # classifier and write them into the pre-defined dataframe
    dla_df.loc['svm', 'acc'] = accuracy_score(y_test, y_pred)
    dla_df.loc['svm', 'prec'] = precision_score(y_test, y_pred)
    dla_df.loc['svm', 'rec'] = recall_score(y_test, y_pred)
    dla_df.loc['svm', 'f1'] = f1_score(y_test, y_pred)

    # Create a muli-layer perceptron classifier with given parameters
    clf = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                            max_iter=300, random_state=9, verbose=1)
    # Fit the model to data matrix X and target(s) y
    clf.fit(X_train, y_train)
    # Predict the class labels of the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy, precision, recall and f1 score of the multi-layer perceptron
    # classifier and write them into the pre-defined dataframe
    dla_df.loc['mlp', 'acc'] = accuracy_score(y_test, y_pred)
    dla_df.loc['mlp', 'prec'] = precision_score(y_test, y_pred)
    dla_df.loc['mlp', 'rec'] = recall_score(y_test, y_pred)
    dla_df.loc['mlp', 'f1'] = f1_score(y_test, y_pred)

    # Create a boxplot of the accuracy, precision, recall and f1 score
    axes[counter].boxplot(dla_df)
    # Set the name of the GAN used for synthetic data generation as the title
    axes[counter].set_title(key)
    # Select the current plot
    plt.sca(axes[counter])
    # And change the xticks accordingly
    plt.xticks(range(1, 5, 1), ["acc", "prec", "rec", "f1"])

# Save the figure containing the data labeling analysis
plt.savefig(r'.\saved\evaluation\resemblance\DLA\DLA_plot.png', bbox_inches='tight')
