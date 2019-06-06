"""
Using CNN
"""
import tensorflow as tf
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Conv2D
from keras.preprocessing.image import ImageDataGenerator as ig
from keras.layers import Dropout
from sklearn.externals import joblib   

#Making Sequential Object
model = Sequential()

#Feature detection and Pooling
model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28,28,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28,28,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))


# Flattening the 2D arrays for fully connected layers
model.add(Flatten())

#2 Hidden layer in the network
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

#compiling of layres
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

#image augmentation
train_datagen = ig(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ig(rescale=1./255)

#Defining path to access for keras library
train_data = train_datagen.flow_from_directory('test/trainingSet',
                                                target_size=(28, 28),
                                                batch_size=32,
                                                class_mode='sparse')


test_data = train_datagen.flow_from_directory('test/testSet',
                                                target_size=(28, 28),
                                                batch_size=32,
                                                class_mode='sparse')

#Fitting the model
#Change epochs according to your machine power
model.fit_generator(
        train_data,
        steps_per_epoch=41999,
        epochs=10,
        validation_data=test_data,
        validation_steps=27999)

# Save the model as a pickle in a file 
joblib.dump(model, 'filename.pkl') 

'''
   Use model.predict("Image address here")
'''    
