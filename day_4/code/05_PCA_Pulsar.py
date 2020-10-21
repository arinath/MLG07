# -------------------------------------------------------------------------------------------------------------------- #
#                                                                                                                      #
#                                             Machine learning Guild Bootcamp                                          #
#                                                © Deloitte Consulting LLP                                             #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #

# Author list:
#    Name                                       Username                                     Email
#    Gert De Geyter                             gedgeyter                                    gedegeyter@deloitte.com

# -------------------------------------------------------------------------------------------------------------------- #
# ###                                               Program Description                                            ### #
# -------------------------------------------------------------------------------------------------------------------- #

# In this excercise we will use Principal Component Analysis on dataset of Pulsar stars.
# HTRU2 is a data set which describes a sample of pulsar candidates collected during the High Time Resolution Universe
# Survey . Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth. They are of
# considerable scientific interest as probes of space-time, the inter-stellar medium, and states of matter.
#
# As pulsars rotate, their emission beam sweeps across the sky, and when this crosses our line of sight, produces a
# detectable pattern of broadband radio emission. As pulsars rotate rapidly, this pattern repeats periodically. Thus
# pulsar search involves looking for periodic radio signals with large radio telescopes.
#
# Each pulsar produces a slightly different emission pattern, which varies slightly with each rotation. Thus a
# potential signal detection known as a 'candidate', is averaged over many rotations of the pulsar, as determined by the
# length of an observation. In the absence of additional info, each candidate could potentially describe a real pulsar.
# However in practice almost all detections are caused by radio frequency interference (RFI) and noise, making
# legitimate signals hard to find.
#
# Machine learning tools are now being used to automatically label pulsar candidates to facilitate rapid analysis.
# Classification systems in particular are being widely adopted, which treat the candidate data sets as binary
# classification problems. Here the legitimate pulsar examples are a minority positive class, and spurious examples the
# majority negative class.
#
# The data set shared here contains 16,259 spurious examples caused by RFI/noise, and 1,639 real pulsar examples.
# These examples have all been checked by human annotators. Each row lists the variables first, and the class label is
# the final entry. The class labels used are 0 (negative) and 1 (positive).
# -------------------------------------------------------------------------------------------------------------------- #
# ###                                            Includes and data load                                            ### #
# -------------------------------------------------------------------------------------------------------------------- #

# Make sure the working directory is set to project folder
# add this is if you are running a notebook
# %matplotlib notebook

import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------------------------------------------------------------------- #
# ###                                                    Program                                                   ### #
# -------------------------------------------------------------------------------------------------------------------- #

# DATA: Each candidate is described by 8 continuous variables, and a single class variable. The first four are simple
# statistics obtained from the integrated pulse profile (folded profile). This is an array of continuous variables that
# describe a longitude-resolved version of the signal that has been averaged in both time and frequency . The remaining
# four variables are similarly obtained from the DM-SNR curve . These are summarised below:
#
# Mean of the integrated profile.
# Standard deviation of the integrated profile.
# Excess kurtosis of the integrated profile.
# Skewness of the integrated profile.
# Mean of the DM-SNR curve.
# Standard deviation of the DM-SNR curve.
# Excess kurtosis of the DM-SNR curve.
# Skewness of the DM-SNR curve.
# Class

columnnames = ['Mean_ip', 'Std_ip', 'Kurtosis_ip', 'Skew_ip', 'Mean_SNR', 'Std_SNR', 'Kurtosis_SNR', 'Skew_SNR','Class']
filepath = os.path.abspath('../data/HTRU_2.csv')
# the data comes without colum names but he
df_pulsar = pd.read_csv(filepath, names=columnnames)

# we are rescaling our data so that it stays in the range 0 to 1.
scaler = MinMaxScaler(feature_range=[0, 1])
df_rescaled = scaler.fit_transform(df_pulsar.drop(columns=["Class"])) # the last column is the target

#Fitting the PCA algorithm with our Data
pca = PCA().fit(df_rescaled)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()

# show the eigenvectors
print(pca.components_)
print(pca.components_[0]) # it seems that its mostly 'Mean_SNR' and 'Std_SNR' that are driving the variance.
# IMPORTANT: this is NOT a model so variance does not tell you anything about feature importance!

pca = PCA(n_components=5)
dataset = pca.fit_transform(df_rescaled)


# dataset now has become your X variables to predict your target Y (data[:, 8:])

# -------------------------------------------------------------------------------------------------------------------- #
