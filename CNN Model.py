import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
import cv2
from keras.preprocessing.image import ImageDataGenerator


data_dir = 'E:\PYthon & AI\AI & ML\Cancer Radar\Dataset'

import pathlib
data_dir = pathlib.Path(data_dir)
list(data_dir.glob('*/*.jpg'))

image_count = len(list(data_dir.glob('*/*.jpg')))
objects = {
'Adeno' : list(data_dir.glob('Adeno/*')),
'Carcinoma' : list(data_dir.glob('Carcinoma/*')),
'Normal' : list(data_dir.glob('Normal/*'))
}

objects_labels = {
    'Adeno' : 0,
   'Carcinoma' : 1,
   'Normal' : 2
    }
print(image_count)

X, y = [], []

for name, images in objects.items():
    print("Printing...")
    for image in images:
          
        img = cv2.imread(str(image))
        
      
        try:
            resized = cv2.resize(img, (128,128))
            X.append(resized)
            y.append(objects_labels[name])
        except:
            print('skipped')
            break
        
X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1,random_state = 0)

print(len(X_train), len(X_test),len(y_train), len(y_test))

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

aug = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape = (128,128,3)),
    layers.experimental.preprocessing.RandomContrast(0.3),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(0.1)
    ])

model = keras.Sequential([
    aug,
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    
    
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),

    layers.Conv2D(250, (3,3), padding='same', activation='relu'),
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.AvgPool2D(2, 2),
    
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.AvgPool2D(2, 2),
    
    layers.Conv2D(256, (2, 2), padding="same", activation="relu"),
    layers.MaxPooling2D(2, 2),
    


    layers.Flatten(),
    layers.Dense(32,  activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(3, activation='softmax'),
       
    ])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
    )


hitory = model.fit(X_train_scaled, y_train, epochs = 30)

model.save('model2.h5')