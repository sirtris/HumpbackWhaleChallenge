# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras import backend as K
#from keras.preprocessing.image import ImageDataGenerator


class LabelOneHotEncoder():
    def __init__(self):
        self.ohe = OneHotEncoder()
        self.le = LabelEncoder()

    def fit_transform(self, x):
        features = self.le.fit_transform(x)
        return self.ohe.fit_transform(features.reshape(-1,1))

    def transform(self, x):
        return self.ohe.transform(self.le.transform(x.reshape(-1,1)))

    def inverse_transform(self, x):
        return self.le.inverse_transform(self.le.inverse_transform(x))

    def inverse_labels(self, x):
        return self.le.inverse_transform(x)

batch_size = 128
num_classes = len(y_cat.toarray()[0])
epochs = 5

SIZE = 64
input_shape = (SIZE,SIZE,1)
lohe = LabelOneHotEncoder()
def fit():
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
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.summary()
    model.fit_generator(image_gen.flow(x_train, y_train.toarray(), batch_size=batch_size),
          steps_per_epoch=5,
          epochs=epochs,
          verbose=1)


#Returns image and top 5 predictions as a nested list
def predict():
    preds = []
    for image in test:
        img = augment_image(image)
        x = img.astype("float32")
        # apply preprocessing to test images
        x = image_gen.standardize(x.reshape(1, SIZE, SIZE))

        y = model.predict_proba(x.reshape(1, SIZE, SIZE, 1))
        predicted_args = np.argsort(y)[0][::-1][:5]
        predicted_tags = lohe.inverse_labels(predicted_args)
        image = split(image)[-1]
        predicted_tags = " ".join(predicted_tags)
        lst = [image, predicted_tags]
        preds.append(lst)
    return preds    
    
def extract(dataframe):
    print("test")
    imglist = []
    for fn in dataframe['Image'][0:1000]:
        img = Image.open(fn)
        img = img.resize((SIZE,SIZE)) # resize
        img = img.convert('LA') # greyscale
        imglist.append(np.array(img)[:, :, 0]) # append an array of size (SIZE,SIZE)
    npimglist = np.array(imglist).reshape((-1, SIZE, SIZE, 1))
    return npimglist.astype("float32")
