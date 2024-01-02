from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data() #load in dataset of fashion images

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']   # list of items indexed respectively

train_images = train_images/255.0   # shrink data so that it's within 0 and 1
test_images = test_images/255.0

# model

# model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),    # creating the neural network, with first layer flattening the data so data can be passed to individual neurons
#                           keras.layers.Dense(128, activation="relu"),   # a layer of fully connected neurons with activation function rectifying linear unit (Fast)
#                           keras.layers.Dense(10, activation="softmax")])    # another layer of fully connected neurons with activation function softmax (All neurons in this layer add up to 1)

# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])   # setting model parameters

# model.fit(train_images, train_labels, epochs=5) # training the model with 5 laps

# model.save("model.h5")

model = keras.models.load_model("model.h5")

prediction = model.predict(test_images) # predict names of test images

def multiPrint(array,total):
    temp=[]
    for i,x in enumerate(array):
        if x>0.2:
            temp.append((round((x/total)*100,2),class_names[i]))
    string=""
    for (x,y) in temp:
        string+=" "+str(y)+":"+str(x)+"%"
    return string

current_image = 0  # Keep track of which image is currently displayed

def plot_image(index):
    plt.clf()  # Clear the current figure
    plt.grid(False)
    plt.imshow(test_images[index], cmap=plt.cm.binary)
    plt.title("Prediction: " + class_names[np.argmax(prediction[index])] + "\nActual: " + class_names[test_labels[index]])
    total = sum(prediction[index])
    maxCertainty = round(max(prediction[index]) / total * 100, 2)
    
    if maxCertainty > 80:
        plt.xlabel("Model Certainty: " + str(maxCertainty) + "%")
    else:
        plt.xlabel("Model Certainties: " + str(multiPrint(prediction[index], total)))
    plt.draw()  # Redraw the current figure

def on_key(event):
    global current_image
    if event.key == 'right':
        current_image = (current_image + 1) % len(test_images)  # Move to the next image
        plot_image(current_image)
    elif event.key == 'left':
        current_image = (current_image - 1) % len(test_images)  # Move to the previous image
        plot_image(current_image)

# Connect the key event to on_key function
fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_key)

# Initial plot
plot_image(current_image)
plt.show()
