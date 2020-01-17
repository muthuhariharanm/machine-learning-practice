from keras.models import Sequential
from keras. layers import Dense 
from keras.utils import to_categorical
import numpy as np 
import mnist
import matplotlib.pyplot as plt

#Training set
train_images = mnist.train_images()
train_labels = mnist.train_labels()

#Test set
test_images = mnist. test_images()
test_labels = mnist.test_labels()

#Normalize the pixel values from [0, 255] to [-0.5 to 0.5] 
train_images = (train_images / 255) - 0.5
test_images = (test_images/ 255) - 0.5

#Flatten each 28 x 28 image into a 784 = 28 * 28
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1,784))

#print the new image shape
print(train_images.shape) #60,000 rows and 784 cols
print(test_images.shape)  #10,000 rows and 784 cols

#Define the model attributes
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=784))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

#Add optimizer
model.compile(
  optimizer= 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

#Train with dataset
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=5,
    batch_size = 3
)

#Save the model
model.save_weights('model.h5')
