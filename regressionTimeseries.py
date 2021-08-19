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
# Download the dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

# Set-up the variables of interest
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

# Load the data as a pandas dataframe.
raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

# 3. Always copy df:s (Keeping the original as is)
dataset = raw_dataset.copy()

# 4. Vizualize data (.head() shows the initial five rows and headers)
dataset.head()

# 5. Checking for Nan and Cleaning the dataset from NaN
dataset.isna().sum()
dataset = dataset.dropna()

# 6. The "Origin" column is really categorical, not numeric. So convert that to a array with map and pd.get_dummies:
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

# 7. Visualize the head/tail (to see difference from 5.)
dataset.tail()

# 8. Split the data into training and testing parts, with .sample()
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 9. Inspect the data, Plot correlation (this is where SNS and pairplot is quite nice - very useful to avoid using
# correlated features)
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')

# 10. Visualize differences (this will show why we need some normalization)
charac = train_dataset.describe().transpose()

# 11. Split features from labels
# Separate the target value, the "label", from the features (this is the .pop() in pandas)
# This label is the value that you will train the model to predict.
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# 12. Normalization
# In the table of statistics it's easy to see how different the ranges of each feature are (helps understanding the need
# for nomalization.)
print(charac[['mean', 'std']])

# The Normalization layer
# The preprocessing.Normalization layer is a clean and simple way to build that preprocessing into your model.
# (compare to MNIST normalization 1.7)
# The first step is to create the layer:
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

# This calculates the mean and variance, and stores them in the layer.
print(normalizer.mean.numpy())

######### 2: Construct a Neural Network #########
# 1. Setup a "simple" Fully-Connected Neural Network (typically used for 'static data', i.e. non-dynamic/time-dependent)
# Keras Sequential Class (for 'simple' networks) rather than Keras Functional (used for more complex architectures)
model = keras.Sequential()
model.add(normalizer)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

# 2. View model (allows for some transparency/logic check of the model)
model.summary()

# 3. "Compile" model (prepare for training)
model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001), metrics='mean_absolute_error')
# loss = function to optimize against ('objective' - what)
# optimizer = optimization algorithm (how)
# metrics = value to monitor ('when are we good?')

# In previous example, we had a batch size set, however as we have fewer data points for this example, we handle all
# training examples as one batch (see https://www.deeplearningbook.org/ for more)
epochs = 100

# 5. Set a validation split-size (how much of the training data to be used for validation)
validation_split = 0.2

# 6. Train the model (.fit = learning the parameters defined by step 2.1 for the specified data)
history = model.fit(
    train_features, train_labels,
    validation_split=validation_split,
    epochs=epochs)

# 7. Save the model (so we do not need to train again)
# How to use saved models: https://www.tensorflow.org/guide/keras/save_and_serialize
model.save('fuel_regression.h5')

# 8. Plot history (in order to understand how training went - Loss should decrease over time.
# Metrics should change according to definition)
plot_loss(history)

######### 3: Test the implementation #########
# 1. By propagating the Neural network (model) with the Test Input (test_features) and compare result with the Test Target (test_labels)
# we get a "score" on the quality of the network. This can be done with keras' 'evaluate' method.
score = model.evaluate(test_features, test_labels, verbose=0)

# 2. Predict some results
test_predictions = model.predict(test_features).flatten()

# 3. Plot some results (error distribution e.g.) - this comes in handy to compare different models (i.e. try to create
# a better model!)
plt.figure()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

plt.figure()
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')