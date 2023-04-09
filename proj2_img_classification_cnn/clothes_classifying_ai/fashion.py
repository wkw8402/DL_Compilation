import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

trainX = trainX / 255.0 # pre-process image data into 0-1 range
testX = testX / 255.0

trainX = trainX.reshape( (trainX.shape[0], 28, 28, 1) ) # change shape of numpy array data into 4d for Conv2D's 4d
testX = testX.reshape( (testX.shape[0], 28, 28, 1) ) # 1 used for black and white, 3 used for colored

# trainX.shape = (60000, 28, 28)

# plt.imshow( trainX[1] )
# plt.gray()
# plt.colorbar()
# plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D( filters=32, kernel_size=(3,3), padding="same", activation='relu', input_shape=(28,28,1) ),

    tf.keras.layers.MaxPooling2D( pool_size=(2,2) ), # 2x2 area used for each max 

    # tf.keras.layers.Dense(128, input_shape=(28,28), activation="relu"), # rectified linear unit = make all negative numbers to 0
    # specify input shape for summary in line28

    tf.keras.layers.Flatten(), # make matrix to 1d from 28x28 2d
    
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax") # num of nodes in last layer = num of categories

    # sigmoid is used for binary prediction, num of last layer node = 1
    # softmax is used for category prediction, num of last layer node = number of categories, sum of all categories' prediction = 1

])

model.summary()

model.compile( loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'] )

# label in integer = [1, 3, 0, 2], use sparse_categorical_crossentropy for loss
# label in one-hot encoding = [ [0,1,0,0] [0,0,0,1] [1,0,0,0] [0,0,1,0] ], use categorical_crossentropy for loss

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5)

# score = model.evaluate( testX, testY )
# print(score)

# convolution layer:
# feature extraction: make 20 duplicates, each having image's different important features
# deep learning based on feature extraction
# apply kernerl(ex. sharpen, gaussian blur)=filter to make layer

# pooling layer(down sampling):
# max pooling summarize areas with max values

# Convolutional Neural Network:
# Input image -> Filters -> Convolutional layer -> pooling -> Flattening -> Dense -> Output

# overfitting = memorize training dataset, happens when last epoch accuracy is greater than evaluation

# goal1 = increase val_accuracy by adding dense layer? adding conv+pooling?
# goal2 = prevent overfitting by stop learning when val_accuracy stops increasing