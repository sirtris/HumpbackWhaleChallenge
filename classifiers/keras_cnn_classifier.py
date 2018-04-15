import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from os.path import split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator


train_df = pd.read_csv('./IO/cropped.csv', sep=';')
train_df = train_df[0:100] #only for testing

# Get the list of train/test files
train = glob('cropped/*jpg')
train = train[0:100] #only for testing
#test = glob('data/test/*jpg')

# train/test directories
train_dir = 'cropped/'
#test_dir = 'data/test/'

# For resizing: can change later
#idealWidth = 64
#idealHeight = 64
SIZE = 64


# Augment a single image
def augment_image(file_name):
    # Open Image
    img = Image.open(file_name)

    # Augmentations
    img = img.convert('LA')
    img = img.resize((SIZE, SIZE))
    # image = shear(image)
    # image = shift(image)
    #  other transformations

    return np.array(img)[:, :, 0]


train_df["Image"] = train_df["Image"].map(lambda a: "cropped/"+a)
ImageToLabelDict = dict(zip(train_df["Image"], train_df["Id"]))

train_img = np.array([augment_image(img) for img in train])
x = train_img

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


read_the_comments_below #comment this (this is just to draw your attention)
#This list(map()) function doesnt work WORK???
y = list(map(ImageToLabelDict.get, train))
#y is here: [None, None, None, None, None, None, None, None, None, ..., None]

lohe = LabelOneHotEncoder()
print(y)
y_cat = lohe.fit_transform(y)


x = x.reshape((-1, SIZE, SIZE, 1))
input_shape = x[0].shape
x_train = x.astype("float32")
y_train = y_cat

image_gen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=15,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True)

# training the image preprocessing
image_gen.fit(x_train, augment=True)


batch_size = 128
num_classes = len(y_cat.toarray()[0])
epochs = 5

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

# Create model and add layers
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

score = model.evaluate(x_train, y_train, verbose=0)
print('Training loss: {0:.4f}\nTraining accuracy:  {1:.4f}'.format(*score))


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


#x = run()
#print(x)