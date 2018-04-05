import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#Linear neural network
def linearNetwork(input_shape,num_classes):
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(0.33))
    model.add(Dense(num_classes, activation='softmax'))
    return model

#Convolutional network without pooling layers as in a DCGAN system
def dcganStyleNetwork(input_shape,num_classes):
    model = Sequential()
    model.add(Conv2D(48, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(48, (5, 5), activation='relu')) #TODO: Fix layer sizes
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(0.33))
    model.add(Dense(num_classes, activation='softmax'))

#Convolutional network from Adam's original code
def convolutionalNetwork(input_shape,num_classes):
    model = Sequential()
    model.add(Conv2D(48, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.33))
    model.add(Flatten())
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(0.33))
    model.add(Dense(num_classes, activation='softmax'))
    return model