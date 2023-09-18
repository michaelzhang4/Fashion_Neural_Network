# Fashion Neural Network #

This project uses a Neural Network to identify pieces of fashion given an image.
The Keras fashion dataset is loaded in as training and test data. The model contains 3 layers, one layer flattening the data then two dense layers which looks at the pixels in the image and returns percentage of likelihood of each possible item.
The model is trained with the training data and then when the project is run you can see how accurate the model is by validating what the model predicts the test data to be in comparison to the actual value. 
