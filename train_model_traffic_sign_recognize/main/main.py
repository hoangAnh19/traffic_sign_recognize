import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Reading the input images and putting them into a numpy array

currentPath = os.getcwd()

inputDirPath = currentPath + '/input/'
modelDirPath = currentPath + '/model/'
modelName = 'model'
epochs = 10
classes = 16

data = []
labels = []

height = 30
width = 30
channels = 3

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

n_inputs = height * width * channels

for i in range(classes):
    path = inputDirPath + "train/{0}/".format(i)
    print(path)
    Class = os.listdir(path)
    for a in Class:
        try:
            image = cv2.imread(path + a)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data.append(np.array(size_image))
            labels.append(i)
        except AttributeError:
            print(" ")

Cells = np.array(data)
labels = np.array(labels)

# Randomize the order of the input images
s = np.arange(Cells.shape[0])
np.random.seed(43)
np.random.shuffle(s)
Cells = Cells[s]
labels = labels[s]

# Splitting the images into train and validation sets
(X_train, X_val) = Cells[(int)(0.2 * len(labels)):], Cells[:(int)(0.2 * len(labels))]
X_train = X_train.astype('float32') / 255
X_val = X_val.astype('float32') / 255
(y_train, y_val) = labels[(int)(0.2 * len(labels)):], labels[:(int)(0.2 * len(labels))]

# Using one hot encoding for the train and validation labels
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,  # Random rotation
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Zoom transformation
    horizontal_flip=True,  # Horizontal flip
    fill_mode="nearest"  # Fill missing pixels after transformation
)

# Fit into training data (Augment images)
datagen.fit(X_train)

# Reduce learning rate when validation loss is not improving
lr_reduction = ReduceLROnPlateau(monitor='val_loss',  # Monitor validation loss
                                 patience=3,  # Wait for 3 epochs if no improvement
                                 verbose=1,  # Display message when LR is reduced
                                 factor=0.5,  # Reduce LR by a factor of 0.5
                                 min_lr=0.00001)  # Minimum LR value

# Definition of the DNN model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5),
                 activation='relu', input_shape=X_train.shape[1:]))
model.add(BatchNormalization())  # Batch Normalization to stabilize learning
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())  # Batch Normalization
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))  # Output layer with 43 classes for classification

# Compilation of the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model with Data Augmentation and LR Scheduling
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),  # Use Data Augmentation
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    callbacks=[lr_reduction])  # Use the LR reduction callback

# Display of the accuracy and the loss values
plt.figure(10)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# Save the model
model.save(modelDirPath + modelName + '.h5')
