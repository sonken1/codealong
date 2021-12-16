# Regression example, predicting Fuel Consumption:
# https://www.tensorflow.org/tutorials/keras/regression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()

######### 1: Data #########

# 1. Download the dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

# 2. Load the data as a pandas dataframe.
raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

# 3. Always copy df:s

# 4. Vizualize data (.head() shows the initial five rows and headers)

# 5. Checking for Nan and Cleaning the dataset from NaN

# 6. The "Origin" column is really categorical, not numeric. So convert that to a array with map and pd.get_dummies:

# 7. Visualize the head/tail

# 8. Split the data into training and testing parts, with .sample()

# 9. Inspect the data, Plot correlation (this is where SNS and pairplot is quite nice)

# 10. Visualize differences (this will show why we need some normalization)

# 11. Split features from labels
# Separate the target value, the "label", from the features (this is the .pop() in pandas)

# 12. Normalization
# The preprocessing.Normalization layer allows norm to be done inside the model, and can be adapted to the data


######### 2: Construct a Neural Network #########
# 1. Setup a "simple" Fully-Connected Neural Network (typically used for 'static data', i.e. non-dynamic/time-dependent)
# Keras Sequential Class (for 'simple' networks) rather than Keras Functional (used for more complex architectures)

# 2. View model

# 3. "Compile" model (prepare for training)

# loss = function to optimize against ('objective' - what)
# optimizer = optimization algorithm (how)
# metrics = value to monitor ('when are we good?')

# 4. In previous example, we had a batch size set, however as we have fewer data points for this example, we handle all
# training examples as one batch

# 5. Set a validation split-size (how much of the training data to be used for validation)

# 6. Train the model

# 7. Save the model (so we do not need to train again)

# 8. Plot history



######### 3: Test the implementation #########
# 1. By propagating the Neural network (model) with the Test Input (test_features) and compare result with the Test Target (test_labels)
# we get a "score" on the quality of the network. This can be done with keras' 'evaluate' method.

# 2. Predict some results


# 3. Plot some results (error distribution e.g.)

