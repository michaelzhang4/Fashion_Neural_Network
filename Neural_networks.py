from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data() #load in dataset of fashion images

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']   # list of items indexed respectively

train_images = train_images/255.0   # shrink data so that it's within 0 and 1
test_images = test_images/255.0

model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),    # creating the neural network, with first layer flattening the data so data can be passed to individual neurons
                          keras.layers.Dense(128, activation="relu"),   # a layer of fully connected neurons with activation function rectifying linear unit (Fast)
                          keras.layers.Dense(10, activation="softmax")])    # another layer of fully connected neurons with activation function softmax (All neurons in this layer add up to 1)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])   # setting model parameters

model.fit(train_images, train_labels, epochs=5) # training the model with 5 laps

prediction = model.predict(test_images) # predict names of test images

for x in range(len(test_images)):   # plot and show images with prediction and actual labels
    plt.grid(False)
    plt.imshow(test_images[x], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[x]])
    plt.title("Prediction: " +class_names[np.argmax(prediction[x])])
    plt.show()