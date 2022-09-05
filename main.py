from genericpath import isfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os

def train_model():
    ##Loading data
    mnist = keras.datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    print(" --------------- Data loaded!\n")

    ##Normalizing data
    X_train = keras.utils.normalize(X_train, axis = 1)
    X_test = keras.utils.normalize(X_test, axis = 1)

    ##Adding Layers
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape = (28,28)))
    model.add(keras.layers.Dense(128, activation = 'relu'))
    model.add(keras.layers.Dense(128, activation = 'relu'))
    model.add(keras.layers.Dense(128, activation = 'relu'))
    model.add(keras.layers.Dense(128, activation = 'relu'))

    model.add(keras.layers.Dense(10, activation = 'softmax'))
    #Compiling Model
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    #Training Model
    model.fit(X_train, Y_train, epochs = 20)
    #Saving Model
    model.save('handwritten.model')
    
    print("-------------- Model Trained!\n")
    
    
    #Evaluate model
    loss, accuracy = model.evaluate(X_test, Y_test)

### MAIN ###

## ---- > Train model 
#train_model()  # Uncomment First time use

#Load Model
model = tf.keras.models.load_model('handwritten.model')
print('-------------- Model loaded!') 

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        res = np.argmax(prediction)
        print("This digit is propably a", end = " ")
        print(res)
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        image_number += 1