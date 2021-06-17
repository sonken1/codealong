# MNIST CONV-NET from:
# https://keras.io/examples/vision/mnist_convnet/

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import multilabel_confusion_matrix
startTime = datetime.now()

######### 1: Data #########
# Keras (our "API" towards Tensorflow contains some typical datasets)
# Load the data as tuples (input, output)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Visualize the input data (important to know how the input looks)
plt.figure()
plt.title('Example input image')
plt.imshow(x_train[0])
plt.show()

# Visualize the output/target data
plt.figure()
plt.hist(y_train)
plt.hist(y_test)
plt.legend(['Train', 'Test'])
plt.title('Distribution over classes')
plt.show()

# Digits should be 0 - 9 (10 Classes)
test_classes = np.unique(y_test)
train_classes = np.unique(y_train)

# Number of classes is the number of unique classes in test/train:
num_classes = len(test_classes)

# Input shape (for the network) is gathered from input data
img_size = x_train[0].shape
input_shape = (x_train[0].shape + (1,))

# Data pre-processing:
# Images (such as loaded x_train) are typically represented so that each pixel lies between 0 - 255 (8 bits, one byte)
# We desire the input-data to be normalized from 0 - 1 (so conversion to floats necessary as well)
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# We need to make sure to provide nbr of channels (e.g. 3 for rgb, 1 for greyscaled)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# print("x_train shape:", x_train.shape)
# print(x_train.shape[0], "train samples")
# print(x_test.shape[0], "test samples")

# We want y (output/target) to be represented as a binary array (rather than just a int) for each example
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

######### 2: Construct a Neural Network #########
# Setup a "simple" Convolutional Network (typically used for Image Classification)
# Keras Sequential Class (for 'simple' networks) rather than Keras Functional (used for more complex architectures)
model = keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation="softmax"))

# Visualize Model (for understanding, debugging and parameters)
model.summary()

# "Compile" model (prepare for training)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# loss = function to optimize against ('objective' - what)
# optimizer = optimization algorithm (how)
# metrics = value to monitor ('when are we good?')

# Set batch size (number of examples per model-update) and epochs (how many times ALL examples are gone through)
batch_size = 128
epochs = 15

# Set a validation split-size (how much of the training data to be used for validation)
validation_split = 0.1

# Fit = train model (according to set parameters)
training_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
model.save('mnist_cnn.h5')
# Plot the 'history' object of the history (in order to understand how training went)
# Loss decrease over the epochs
plt.figure()
plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])
plt.legend(['Loss', 'Val_loss'])
plt.show()

# Accuracy increase over the epochs
plt.figure()
plt.plot(training_history.history['accuracy'])
plt.plot(training_history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val_accuracy'])
plt.show()

######### 3: Test the implementation #########
# By propagating the Neural network (model) with the Test Input (x_test) and compare result with the Test Target (y_test)
# we get a "score" on the quality of the network. This can be done with keras' 'evaluate' method.
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Prediction array:
predictions = model.predict(x_test)

# Display one test image
plt.figure()
idx = 1
plt.imshow(np.squeeze(x_test[idx]))
plt.title('True Label: {}. Predicted: {}'.format(np.argmax(y_test[idx]), np.argmax(predictions[idx])))
plt.show()

# Confusion Matrix (for all):
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)
multilabel_confusion_matrix(y_true, y_pred, labels=test_classes)

print('Time: ')
print(datetime.now() - startTime)

# Test on separate image (The size of this image makes it very bad)
path = r"C:\Users\elias\Documents\codealong\presentationmaterial\handwritten-2.jpg"
img = load_img(path, target_size=img_size, color_mode="grayscale")
img = np.expand_dims(img, 2)
img = np.expand_dims(img, 0)
pred = model.predict(img).argmax()
