# MNIST CONV-NET from:
# https://keras.io/examples/vision/mnist_convnet/

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

######### 1: Data #########
# Keras (our "API" towards Tensorflow contains some typical datasets)

# 1. Load the data as tuples (input, output)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Visualize the input data (important to know how the input looks)
plt.figure()
plt.imshow(x_train[0])
plt.show()

# 3. Visualize the output/target data
plt.figure()
plt.hist(y_train)
plt.hist(y_test)
plt.legend(['Train Target', 'Test Target'])
plt.show()

# 4. Digits should be 0 - 9 (10 Classes)
num_train_class = np.unique(y_train)
num_test_class = np.unique(y_test)

# 5. Number of classes is the number of unique classes in test/train
num_class = len(num_train_class)

# 6. Input shape (for the network) is gathered from input data


# 7. Data pre-processing:
# Images (such as loaded x_train) are typically represented so that each pixel lies between 0 - 255 (8 bits, one byte)
# We desire the input-data to be normalized from 0 - 1 (so conversion to floats necessary as well)
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# 8. We need to make sure to provide nbr of channels (e.g. 3 for rgb, 1 for greyscaled)
img_size = x_train[0].shape
input_shape = (x_train[0].shape + (1, ))

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# 9. We want y (output/target) to be represented as a binary array (rather than just a int) for each example
y_test = keras.utils.to_categorical(y_test, num_class)
y_train = keras.utils.to_categorical(y_train, num_class)

######### 2: Construct a Neural Network #########
# 1. Setup a "simple" Convolutional Network (typically used for Image Classification)
# Keras Sequential Class (for 'simple' networks) rather than Keras Functional (used for more complex architectures)
model = keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=input_shape, activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(num_class, activation="softmax"))

# 2. Visualize Model (for understanding, debugging and parameters)
model.summary()

# 3. "Compile" model (prepare for training)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
# loss = function to optimize against ('objective' - what)
# optimizer = optimization algorithm (how)
# metrics = value to monitor ('when are we good?')

# 4. Set batch size (number of examples per model-update) and epochs (how many times ALL examples are gone through)
batch_size = 128
epochs = 15

# 5. Set a validation split-size (how much of the training data to be used for validation)
validation_split = 0.1

# 6. Fit = train model (according to set parameters)
history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split)

# 7. Save the model (so we do not need to train again)
model.save('mnist_codealong.h5')

# 8. Plot the 'history' object of the history (in order to understand how training went)
# Loss decrease over the epochs
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Loss', 'Val_loss'])
plt.show()

# 9. Accuracy increase over the epochs
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val_Accuracy'])
plt.show()

######### 3: Test the implementation #########
# 1. By propagating the Neural network (model) with the Test Input (x_test) and compare result with the Test Target (y_test)
# we get a "score" on the quality of the network. This can be done with keras' 'evaluate' method.
score = model.evaluate(x_test, y_test, verbose=0)
print("Loss:", score[0])
print("Accuracy:", score[1])

# 2. Calculate Prediction array:
y_prediction = model.predict(x_test)

# 3. Display one test image

# 4. Confusion Matrix (for all):
y_pred = np.argmax(y_prediction, axis=1)
y_true = np.argmax(y_test, axis=1)
confusion_matrix(y_true, y_pred)

# 5. Test on separate image (The size of this image makes it very bad)
path = r"C:\Users\elias\Documents\codealong\presentationmaterial\elias_2.png"
img = load_img(path, color_mode="grayscale")
img = np.expand_dims(img, -1)
img = np.expand_dims(img, 0)
prediction = model.predict(img).argmax()

